import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(device_name):
    if device_name != "auto":
        return device_name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype_name, device):
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32

    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_tokenizer(model_id):
    if model_id == "google/gemma-4-E2B":
        return AutoTokenizer.from_pretrained(
            model_id,
            extra_special_tokens={"video_token": "<|video|>"},
        )
    return AutoTokenizer.from_pretrained(model_id)


def load_model(model_id, dtype, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model


def get_config_value(config, names, default=None): 
    # like get_config_value(config, ["num_hidden_layers", "n_layer"])
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def cache_shape_from_config(config, batch_size, past_seq_len):
    num_layers = get_config_value(config, ["num_hidden_layers", "n_layer"])
    num_attention_heads = get_config_value(
        config,
        ["num_attention_heads", "n_head"],
    )
    num_key_value_heads = get_config_value(
        config,
        ["num_key_value_heads", "num_kv_heads"],
        num_attention_heads,
    )
    head_dim = get_config_value(config, ["head_dim"])
    hidden_size = get_config_value(config, ["hidden_size", "n_embd"])

    if head_dim is None:
        if hidden_size is None or num_attention_heads is None:
            raise ValueError("Cannot infer head_dim from model config.")
        head_dim = hidden_size // num_attention_heads

    if num_layers is None or num_key_value_heads is None:
        raise ValueError("Cannot infer KV-cache shape from model config.")

    return {
        "num_layers": int(num_layers),
        "num_attention_heads": int(num_attention_heads),
        "num_key_value_heads": int(num_key_value_heads),
        "head_dim": int(head_dim),
        "batch_size": int(batch_size),
        "past_seq_len": int(past_seq_len),
    }


def make_dummy_past_key_values(cache_shape, dtype, device):
    flat_past = []
    tensor_shape = (
        cache_shape["batch_size"],
        cache_shape["num_key_value_heads"],
        cache_shape["past_seq_len"],
        cache_shape["head_dim"],
    )
    for _ in range(cache_shape["num_layers"]):
        key = torch.zeros(tensor_shape, dtype=dtype, device=device)
        value = torch.zeros(tensor_shape, dtype=dtype, device=device)
        flat_past.extend([key, value])
    return tuple(flat_past)


def flatten_cache(cache):
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()

    flat = []
    for layer_cache in cache:
        flat.append(layer_cache[0])
        flat.append(layer_cache[1])
    return tuple(flat)


def build_dynamic_axes_no_cache():
    return {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    }


def build_names_and_dynamic_axes_with_cache(num_layers):
    input_names = ["input_ids", "attention_mask", "position_ids"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "current_sequence"},
        "attention_mask": {0: "batch", 1: "total_sequence"},
        "position_ids": {0: "batch", 1: "current_sequence"},
        "logits": {0: "batch", 1: "current_sequence"},
    }

    for layer_idx in range(num_layers):
        past_key_name = f"past_key_{layer_idx}"
        past_value_name = f"past_value_{layer_idx}"
        present_key_name = f"present_key_{layer_idx}"
        present_value_name = f"present_value_{layer_idx}"

        input_names.extend([past_key_name, past_value_name])
        output_names.extend([present_key_name, present_value_name])

        dynamic_axes[past_key_name] = {0: "batch", 2: "past_sequence"}
        dynamic_axes[past_value_name] = {0: "batch", 2: "past_sequence"}
        dynamic_axes[present_key_name] = {0: "batch", 2: "total_sequence"}
        dynamic_axes[present_value_name] = {0: "batch", 2: "total_sequence"}

    return input_names, output_names, dynamic_axes


class NoCacheWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits


class WithCacheWrapper(torch.nn.Module):
    def __init__(self, model, num_layers):
        super().__init__()
        self.model = model
        self.num_layers = num_layers

    def forward(self, input_ids, attention_mask, position_ids, *flat_past_key_values):
        past_key_values = []
        for layer_idx in range(self.num_layers):
            key = flat_past_key_values[2 * layer_idx]
            value = flat_past_key_values[2 * layer_idx + 1]
            past_key_values.append((key, value))

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=tuple(past_key_values),
            use_cache=True,
            return_dict=True,
        )
        return (outputs.logits, *flatten_cache(outputs.past_key_values))


def prepare_no_cache_inputs(tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    return input_ids, attention_mask


def prepare_with_cache_inputs(tokenizer, prompt, past_seq_len, device):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"][:, -1:].to(device)
    current_seq_len = input_ids.shape[1]
    batch_size = input_ids.shape[0]

    attention_mask = torch.ones(
        (batch_size, past_seq_len + current_seq_len),
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(
        past_seq_len,
        past_seq_len + current_seq_len,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    return input_ids, attention_mask, position_ids


def save_reference_logits(wrapper, wrapper_inputs, output_path):
    with torch.inference_mode():
        outputs = wrapper(*wrapper_inputs)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        last_token_logits = logits[:, -1, :].detach().float().cpu()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(last_token_logits, output_path)
    return {
        "path": str(output_path),
        "shape": list(last_token_logits.shape),
        "dtype": str(last_token_logits.dtype),
    }


def export_no_cache(model, tokenizer, args, dtype, device):
    input_ids, attention_mask = prepare_no_cache_inputs(
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
    )
    wrapper = NoCacheWrapper(model).eval()
    wrapper_inputs = (input_ids, attention_mask)
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = build_dynamic_axes_no_cache()

    reference = save_reference_logits(
        wrapper=wrapper,
        wrapper_inputs=wrapper_inputs,
        output_path=Path(args.reference_logits_output),
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        wrapper_inputs,
        args.output,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )

    return {
        "export_mode": "no_cache",
        "onnx_path": args.output,
        "opset": args.opset,
        "dtype": str(dtype),
        "device": device,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "reference_logits": reference,
        "dummy_inputs": {
            "input_ids_shape": list(input_ids.shape),
            "attention_mask_shape": list(attention_mask.shape),
        },
    }


def export_with_cache(model, tokenizer, args, dtype, device):
    input_ids, attention_mask, position_ids = prepare_with_cache_inputs(
        tokenizer=tokenizer,
        prompt=args.prompt,
        past_seq_len=args.past_seq_len,
        device=device,
    )

    cache_shape = cache_shape_from_config(
        config=model.config,
        batch_size=input_ids.shape[0],
        past_seq_len=args.past_seq_len,
    )
    flat_past = make_dummy_past_key_values(
        cache_shape=cache_shape,
        dtype=dtype,
        device=device,
    )

    input_names, output_names, dynamic_axes = build_names_and_dynamic_axes_with_cache(
        num_layers=cache_shape["num_layers"],
    )

    wrapper = WithCacheWrapper(model, num_layers=cache_shape["num_layers"]).eval()
    wrapper_inputs = (input_ids, attention_mask, position_ids, *flat_past)

    reference = save_reference_logits(
        wrapper=wrapper,
        wrapper_inputs=wrapper_inputs,
        output_path=Path(args.reference_logits_output),
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        wrapper_inputs,
        args.output,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )

    return {
        "export_mode": "with_cache",
        "onnx_path": args.output,
        "opset": args.opset,
        "dtype": str(dtype),
        "device": device,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "cache_shape": cache_shape,
        "reference_logits": reference,
        "dummy_inputs": {
            "input_ids_shape": list(input_ids.shape),
            "attention_mask_shape": list(attention_mask.shape),
            "position_ids_shape": list(position_ids.shape),
            "flat_past_count": len(flat_past),
            "flat_past_shape_each": list(flat_past[0].shape) if flat_past else None,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="google/gemma-4-E2B")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--prompt", type=str, default="Deep learning is")
    parser.add_argument(
        "--export-mode",
        type=str,
        default="no_cache",
        choices=["no_cache", "with_cache"],
    )
    parser.add_argument("--past-seq-len", type=int, default=8)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output", type=str, default="models/gemma4_e2b_no_cache.onnx")
    parser.add_argument(
        "--metadata-output",
        type=str,
        default="results/onnx_export.json",
    )
    parser.add_argument(
        "--reference-logits-output",
        type=str,
        default="results/onnx_reference_logits.pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "prompt": args.prompt,
        "success": False,
    }

    try:
        tokenizer = load_tokenizer(args.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = load_model(args.model_id, dtype, device)

        if args.export_mode == "no_cache":
            export_metadata = export_no_cache(
                model=model,
                tokenizer=tokenizer,
                args=args,
                dtype=dtype,
                device=device,
            )
        else:
            export_metadata = export_with_cache(
                model=model,
                tokenizer=tokenizer,
                args=args,
                dtype=dtype,
                device=device,
            )

        metadata.update(export_metadata)
        metadata["success"] = True

    except Exception as exc:
        metadata.update(
            {
                "success": False,
                "export_mode": args.export_mode,
                "onnx_path": args.output,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "notes": (
                    "ONNX export can fail for large decoder-only LLMs, especially "
                    "when exposing KV-cache tensors. Keep this metadata for the "
                    "conversion notes section."
                ),
            }
        )

    metadata_output = Path(args.metadata_output)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")

    if metadata["success"]:
        print(f"Wrote ONNX model to {args.output}")
    else:
        print(f"ONNX export failed. Wrote metadata to {args.metadata_output}")


if __name__ == "__main__":
    main()
