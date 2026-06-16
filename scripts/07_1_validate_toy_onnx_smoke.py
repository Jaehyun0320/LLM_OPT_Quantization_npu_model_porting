import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_providers(provider):
    available = ort.get_available_providers()
    if provider == "auto":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available. "
                f"Available providers: {available}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    raise ValueError(f"Unknown provider: {provider}")


def ensure_onnx_exists(path, producer_script):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ONNX file: {path}. Run {producer_script} first to create it."
        )


def validate_no_cache(args, providers):
    onnx_path = Path(args.no_cache_onnx)
    ensure_onnx_exists(
        onnx_path,
        "python3 scripts/06_1_toy_onnx_smoke.py",
    )

    toy = load_module(
        REPO_ROOT / "scripts" / "06_1_toy_onnx_smoke.py",
        "toy_onnx_smoke",
    )
    torch.manual_seed(args.seed)

    model = toy.ToyCausalLM(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
    ).eval()
    input_ids, attention_mask = toy.build_inputs(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
    )

    with torch.inference_mode():
        torch_logits = model(input_ids, attention_mask)

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    ort_logits = session.run(
        ["logits"],
        {
            "input_ids": input_ids.cpu().numpy().astype(np.int64),
            "attention_mask": attention_mask.cpu().numpy().astype(np.int64),
        },
    )[0]

    comparison = toy.compare_outputs(
        torch_logits=torch_logits,
        ort_logits=ort_logits,
        tolerance=args.tolerance,
    )

    return {
        "success": comparison["passed"],
        "onnx_path": str(onnx_path),
        "model": {
            "type": "ToyCausalLM",
            "vocab_size": args.vocab_size,
            "hidden_size": args.hidden_size,
        },
        "inputs": {
            "input_ids_shape": list(input_ids.shape),
            "attention_mask_shape": list(attention_mask.shape),
        },
        "comparison": comparison,
    }


def validate_with_cache(args, providers):
    onnx_path = Path(args.kv_cache_onnx)
    ensure_onnx_exists(
        onnx_path,
        "python3 scripts/06_2_toy_kv_cache_onnx_smoke.py",
    )
    if args.hidden_size != args.num_heads * args.head_dim:
        raise ValueError("--hidden-size must equal --num-heads * --head-dim.")

    toy_kv = load_module(
        REPO_ROOT / "scripts" / "06_2_toy_kv_cache_onnx_smoke.py",
        "toy_kv_cache_onnx_smoke",
    )
    torch.manual_seed(args.seed)

    model = toy_kv.ToyKVCachedCausalLM(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
    ).eval()
    input_ids, attention_mask, past_key, past_value = toy_kv.build_inputs(
        batch_size=args.batch_size,
        current_seq_len=args.current_seq_len,
        past_seq_len=args.past_seq_len,
        vocab_size=args.vocab_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
    )

    torch_logits, torch_present_key, torch_present_value = toy_kv.run_torch_step(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key=past_key,
        past_value=past_value,
    )

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    ort_logits, ort_present_key, ort_present_value = toy_kv.run_ort_step(
        session=session,
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key=past_key,
        past_value=past_value,
    )

    single_step = {
        "logits": toy_kv.compare_array(torch_logits, ort_logits, args.tolerance),
        "present_key": toy_kv.compare_array(
            torch_present_key,
            ort_present_key,
            args.tolerance,
        ),
        "present_value": toy_kv.compare_array(
            torch_present_value,
            ort_present_value,
            args.tolerance,
        ),
        "cache_sequence_grew": (
            list(past_key.shape)[2] + args.current_seq_len
            == list(torch_present_key.shape)[2]
        ),
    }
    single_step["passed"] = (
        single_step["logits"]["passed"]
        and single_step["present_key"]["passed"]
        and single_step["present_value"]["passed"]
        and single_step["cache_sequence_grew"]
    )

    loop = toy_kv.run_loop(
        session=session,
        model=model,
        args=args,
        initial_present_key=torch_present_key,
        initial_present_value=torch_present_value,
    )

    return {
        "success": bool(single_step["passed"] and loop["passed"]),
        "onnx_path": str(onnx_path),
        "model": {
            "type": "ToyKVCachedCausalLM",
            "vocab_size": args.vocab_size,
            "hidden_size": args.hidden_size,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
        },
        "inputs": {
            "input_ids_shape": list(input_ids.shape),
            "attention_mask_shape": list(attention_mask.shape),
            "past_key_shape": list(past_key.shape),
            "past_value_shape": list(past_value.shape),
        },
        "single_step": single_step,
        "loop": loop,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["no_cache", "with_cache", "both"],
    )
    parser.add_argument("--provider", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--current-seq-len", type=int, default=1)
    parser.add_argument("--past-seq-len", type=int, default=4)
    parser.add_argument("--loop-steps", type=int, default=4)
    parser.add_argument(
        "--no-cache-onnx",
        type=str,
        default="models/toy_causal_lm_smoke.onnx",
    )
    parser.add_argument(
        "--kv-cache-onnx",
        type=str,
        default="models/toy_kv_cache_causal_lm_smoke.onnx",
    )
    parser.add_argument("--output", type=str, default="results/toy_onnx_validation.json")
    return parser.parse_args()


def main():
    args = parse_args()
    providers = choose_providers(args.provider)

    validations = {}
    if args.mode in ("no_cache", "both"):
        validations["no_cache"] = validate_no_cache(args, providers)
    if args.mode in ("with_cache", "both"):
        validations["with_cache"] = validate_with_cache(args, providers)

    success = all(row["success"] for row in validations.values())
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Standalone toy ONNX Runtime validation smoke test. This consumes "
            "ONNX files produced by scripts/06_1_toy_onnx_smoke.py and "
            "scripts/06_2_toy_kv_cache_onnx_smoke.py, then re-computes the "
            "PyTorch references and compares them with ORT outputs."
        ),
        "success": bool(success),
        "mode": args.mode,
        "runtime": {
            "requested_provider": args.provider,
            "providers": providers,
            "available_providers": ort.get_available_providers(),
        },
        "validations": validations,
        "note": (
            "This validates the ONNX/ORT pipeline on ONNX-friendly toy graphs. "
            "It does not prove that Gemma4/tiny-gpt2 HF export works, because "
            "those failed earlier in the Transformers causal-mask exporter path."
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print(f"Wrote toy ONNX validation result to {args.output}")
    print(f"Success: {result['success']}")

    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
