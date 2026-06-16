import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


class ToyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask):
        hidden_states = self.embed(input_ids)
        hidden_size = hidden_states.shape[-1]

        scores = torch.matmul(
            hidden_states,
            hidden_states.transpose(-1, -2),
        ) / math.sqrt(hidden_size)

        seq_len = input_ids.shape[1]
        causal_mask = torch.tril(
            torch.ones(
                (seq_len, seq_len),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        )
        padding_mask = attention_mask[:, None, :].to(hidden_states.dtype)
        combined_mask = causal_mask[None, :, :] * padding_mask
        masked_scores = scores + (1.0 - combined_mask) * -10000.0

        probs = torch.softmax(masked_scores, dim=-1)
        context = torch.matmul(probs, hidden_states)
        return self.lm_head(context)


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


def build_inputs(batch_size, seq_len, vocab_size):
    input_ids = torch.arange(batch_size * seq_len, dtype=torch.long).reshape(
        batch_size,
        seq_len,
    )
    input_ids = input_ids % vocab_size
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    if seq_len >= 2:
        attention_mask[:, -1] = 0
    return input_ids, attention_mask


def compare_outputs(torch_logits, ort_logits, tolerance):
    torch_np = torch_logits.detach().cpu().numpy().astype(np.float32)
    ort_np = ort_logits.astype(np.float32)
    diff = np.abs(torch_np - ort_np)
    return {
        "torch_logits_shape": list(torch_np.shape),
        "onnx_logits_shape": list(ort_np.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "tolerance": tolerance,
        "passed": bool(float(diff.max()) <= tolerance),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provider", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--onnx-output", type=str, default="models/toy_causal_lm_smoke.onnx")
    parser.add_argument("--output", type=str, default="results/toy_onnx_smoke.json")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model = ToyCausalLM(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
    ).eval()
    input_ids, attention_mask = build_inputs(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
    )

    with torch.inference_mode():
        torch_logits = model(input_ids, attention_mask)

    onnx_path = Path(args.onnx_output)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=False,
    )

    providers = choose_providers(args.provider)
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    ort_logits = session.run(
        ["logits"],
        {
            "input_ids": input_ids.cpu().numpy().astype(np.int64),
            "attention_mask": attention_mask.cpu().numpy().astype(np.int64),
        },
    )[0]

    comparison = compare_outputs(
        torch_logits=torch_logits,
        ort_logits=ort_logits,
        tolerance=args.tolerance,
    )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Toy causal-LM smoke test for ONNX export and ONNX Runtime "
            "validation. This avoids Hugging Face Transformers masking_utils "
            "so it checks the local ONNX pipeline separately from Gemma4 or "
            "tiny-gpt2 exporter compatibility."
        ),
        "success": comparison["passed"],
        "model": {
            "type": "ToyCausalLM",
            "vocab_size": args.vocab_size,
            "hidden_size": args.hidden_size,
            "uses_attention_mask": True,
            "uses_causal_mask": True,
        },
        "export": {
            "onnx_path": str(onnx_path),
            "opset": args.opset,
            "input_names": ["input_ids", "attention_mask"],
            "output_names": ["logits"],
        },
        "runtime": {
            "requested_provider": args.provider,
            "providers": providers,
            "available_providers": ort.get_available_providers(),
        },
        "inputs": {
            "input_ids_shape": list(input_ids.shape),
            "attention_mask_shape": list(attention_mask.shape),
            "attention_mask": attention_mask.cpu().tolist(),
        },
        "comparison": comparison,
        "note": (
            "Passing this smoke test does not mean Gemma4 exports successfully. "
            "It only verifies that a small ONNX-friendly causal LM graph can be "
            "exported and numerically validated with ONNX Runtime."
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print(f"Wrote toy ONNX smoke result to {args.output}")
    print(f"Success: {result['success']}")
    print(f"Max abs diff: {comparison['max_abs_diff']}")

    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
