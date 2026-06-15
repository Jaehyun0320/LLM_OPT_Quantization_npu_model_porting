import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer


def load_tokenizer(model_id):
    if model_id == "google/gemma-4-E2B":
        return AutoTokenizer.from_pretrained(
            model_id,
            extra_special_tokens={"video_token": "<|video|>"},
        )
    return AutoTokenizer.from_pretrained(model_id)


def choose_providers(provider):
    available = ort.get_available_providers()

    if provider == "auto":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available. "
                f"Available providers: {available}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if provider == "cpu":
        return ["CPUExecutionProvider"]

    raise ValueError(f"Unknown provider: {provider}")


def numpy_dtype_from_onnx_type(onnx_type):
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }
    if onnx_type == "tensor(bfloat16)":
        raise RuntimeError(
            "ONNX input expects bfloat16, but NumPy has no native bfloat16 dtype. "
            "For validation, export with --dtype fp16 or use an ONNX Runtime path "
            "that supports bfloat16 OrtValue inputs."
        )
    if onnx_type not in mapping:
        raise RuntimeError(f"Unsupported ONNX input type: {onnx_type}")
    return mapping[onnx_type]


def input_type_map(session):
    return {input_info.name: input_info.type for input_info in session.get_inputs()}


def output_names(session):
    return [output_info.name for output_info in session.get_outputs()]


def prepare_no_cache_inputs(tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="np")
    return {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }


def prepare_with_cache_inputs(tokenizer, prompt, cache_shape, type_map):
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"][:, -1:].astype(np.int64)
    batch_size = input_ids.shape[0]
    current_seq_len = input_ids.shape[1]
    past_seq_len = int(cache_shape["past_seq_len"])
    num_layers = int(cache_shape["num_layers"])

    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": np.ones(
            (batch_size, past_seq_len + current_seq_len),
            dtype=np.int64,
        ),
        "position_ids": np.arange(
            past_seq_len,
            past_seq_len + current_seq_len,
            dtype=np.int64,
        )[None, :],
    }

    for layer_idx in range(num_layers):
        key_name = f"past_key_{layer_idx}"
        value_name = f"past_value_{layer_idx}"
        key_dtype = numpy_dtype_from_onnx_type(type_map[key_name])
        value_dtype = numpy_dtype_from_onnx_type(type_map[value_name])

        cache_tensor_shape = (
            int(cache_shape["batch_size"]),
            int(cache_shape["num_key_value_heads"]),
            past_seq_len,
            int(cache_shape["head_dim"]),
        )
        ort_inputs[key_name] = np.zeros(cache_tensor_shape, dtype=key_dtype)
        ort_inputs[value_name] = np.zeros(cache_tensor_shape, dtype=value_dtype)

    return ort_inputs


def load_reference_logits(path):
    logits = torch.load(path, map_location="cpu")
    if isinstance(logits, torch.Tensor):
        return logits.detach().float().cpu().numpy()
    return np.asarray(logits, dtype=np.float32)


def compare_logits(onnx_logits, reference_logits, tolerance):
    onnx_last_logits = onnx_logits[:, -1, :].astype(np.float32)
    reference_logits = reference_logits.astype(np.float32)

    diff = np.abs(onnx_last_logits - reference_logits)
    max_abs_diff = float(diff.max())
    mean_abs_diff = float(diff.mean())

    return {
        "onnx_last_logits_shape": list(onnx_last_logits.shape),
        "reference_logits_shape": list(reference_logits.shape),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "tolerance": tolerance,
        "passed": max_abs_diff <= tolerance,
    }


def validate_cache_outputs(outputs, output_names_list, cache_shape):
    output_map = dict(zip(output_names_list, outputs))
    num_layers = int(cache_shape["num_layers"])
    expected_total_seq_len = int(cache_shape["past_seq_len"]) + 1

    rows = []
    all_shapes_ok = True
    for layer_idx in range(num_layers):
        key_name = f"present_key_{layer_idx}"
        value_name = f"present_value_{layer_idx}"
        key_shape = list(output_map[key_name].shape)
        value_shape = list(output_map[value_name].shape)

        key_ok = key_shape[2] == expected_total_seq_len
        value_ok = value_shape[2] == expected_total_seq_len
        all_shapes_ok = all_shapes_ok and key_ok and value_ok

        rows.append(
            {
                "layer": layer_idx,
                "present_key_shape": key_shape,
                "present_value_shape": value_shape,
                "expected_total_seq_len": expected_total_seq_len,
                "key_sequence_ok": key_ok,
                "value_sequence_ok": value_ok,
            }
        )

    return {
        "expected_present_cache_count": 2 * num_layers,
        "observed_present_cache_count": len(output_names_list) - 1,
        "expected_total_seq_len": expected_total_seq_len,
        "all_shapes_ok": all_shapes_ok,
        "layers": rows,
    }


def greedy_loop_with_cache(session, tokenizer, initial_inputs, output_names_list, steps):
    if steps <= 0:
        return None

    ort_inputs = dict(initial_inputs)
    generated_token_ids = []
    shape_trace = []

    for step in range(steps):
        outputs = session.run(None, ort_inputs)
        output_map = dict(zip(output_names_list, outputs))
        logits = output_map["logits"]
        next_token = logits[:, -1, :].argmax(axis=-1).astype(np.int64)
        generated_token_ids.append(int(next_token[0]))

        present_names = [name for name in output_names_list if name.startswith("present_")]
        next_inputs = {
            "input_ids": next_token.reshape(1, 1),
        }

        previous_attention_mask = ort_inputs["attention_mask"]
        next_total_seq_len = previous_attention_mask.shape[1] + 1
        next_inputs["attention_mask"] = np.ones(
            (previous_attention_mask.shape[0], next_total_seq_len),
            dtype=np.int64,
        )
        next_inputs["position_ids"] = np.array(
            [[next_total_seq_len - 1]],
            dtype=np.int64,
        )

        for present_name in present_names:
            past_name = present_name.replace("present_", "past_")
            next_inputs[past_name] = output_map[present_name]

        shape_trace.append(
            {
                "step": step,
                "next_token_id": int(next_token[0]),
                "input_ids_shape": list(ort_inputs["input_ids"].shape),
                "attention_mask_shape": list(ort_inputs["attention_mask"].shape),
                "first_present_key_shape": list(output_map[present_names[0]].shape)
                if present_names
                else None,
            }
        )

        ort_inputs = next_inputs

        if tokenizer.eos_token_id is not None and int(next_token[0]) == tokenizer.eos_token_id:
            break

    return {
        "steps_requested": steps,
        "steps_completed": len(generated_token_ids),
        "generated_token_ids": generated_token_ids,
        "decoded_text": tokenizer.decode(generated_token_ids, skip_special_tokens=True),
        "shape_trace": shape_trace,
        "note": (
            "This is a mechanical with-cache loop using dummy zero initial cache. "
            "It checks recurrent ONNX I/O wiring, not meaningful text quality."
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="results/onnx_export.json")
    parser.add_argument("--onnx", type=str, default=None)
    parser.add_argument("--reference-logits", type=str, default=None)
    parser.add_argument("--provider", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--loop-steps", type=int, default=0)
    parser.add_argument("--output", type=str, default="results/onnx_validation.json")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = json.loads(Path(args.metadata).read_text())

    onnx_path = args.onnx or metadata["onnx_path"]
    reference_logits_path = (
        args.reference_logits
        or metadata.get("reference_logits", {}).get("path")
    )
    if reference_logits_path is None:
        raise ValueError(
            "Reference logits path was not provided and was not found in metadata."
        )

    providers = choose_providers(args.provider)
    session = ort.InferenceSession(onnx_path, providers=providers)
    type_map = input_type_map(session)
    output_names_list = output_names(session)

    tokenizer = load_tokenizer(metadata["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    export_mode = metadata["export_mode"]
    if export_mode == "no_cache":
        ort_inputs = prepare_no_cache_inputs(tokenizer, metadata["prompt"])
    elif export_mode == "with_cache":
        ort_inputs = prepare_with_cache_inputs(
            tokenizer=tokenizer,
            prompt=metadata["prompt"],
            cache_shape=metadata["cache_shape"],
            type_map=type_map,
        )
    else:
        raise ValueError(f"Unsupported export mode: {export_mode}")

    outputs = session.run(None, ort_inputs)
    logits = outputs[0]
    reference_logits = load_reference_logits(reference_logits_path)
    logits_comparison = compare_logits(
        onnx_logits=logits,
        reference_logits=reference_logits,
        tolerance=args.tolerance,
    )

    cache_validation = None
    loop_validation = None
    if export_mode == "with_cache":
        cache_validation = validate_cache_outputs(
            outputs=outputs,
            output_names_list=output_names_list,
            cache_shape=metadata["cache_shape"],
        )
        loop_validation = greedy_loop_with_cache(
            session=session,
            tokenizer=tokenizer,
            initial_inputs=ort_inputs,
            output_names_list=output_names_list,
            steps=args.loop_steps,
        )

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata_path": args.metadata,
        "onnx_path": onnx_path,
        "reference_logits_path": reference_logits_path,
        "model_id": metadata["model_id"],
        "export_mode": export_mode,
        "providers_requested": providers,
        "providers_active": session.get_providers(),
        "input_names": [input_info.name for input_info in session.get_inputs()],
        "output_names": output_names_list,
        "logits_comparison": logits_comparison,
        "cache_validation": cache_validation,
        "loop_validation": loop_validation,
        "passed": (
            logits_comparison["passed"]
            and (
                cache_validation is None
                or cache_validation["all_shapes_ok"]
            )
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print(f"Wrote ONNX validation result to {args.output}")
    print(f"Passed: {result['passed']}")


if __name__ == "__main__":
    main()
