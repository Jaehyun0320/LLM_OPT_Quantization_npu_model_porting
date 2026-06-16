import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


class ToyKVCachedCausalLM(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, head_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.out_proj = torch.nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def split_heads(self, tensor):
        batch_size, seq_len, _ = tensor.shape
        return tensor.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def merge_heads(self, tensor):
        batch_size, _, seq_len, _ = tensor.shape
        return tensor.transpose(1, 2).reshape(
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        )

    def forward(self, input_ids, attention_mask, past_key, past_value):
        hidden_states = self.embed(input_ids)
        query = self.split_heads(self.q_proj(hidden_states))
        current_key = self.split_heads(self.k_proj(hidden_states))
        current_value = self.split_heads(self.v_proj(hidden_states))

        present_key = torch.cat([past_key, current_key], dim=2)
        present_value = torch.cat([past_value, current_value], dim=2)

        scores = torch.matmul(query, present_key.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)

        mask = attention_mask[:, None, None, :].to(scores.dtype)
        masked_scores = scores + (1.0 - mask) * -10000.0
        probs = torch.softmax(masked_scores, dim=-1)
        context = torch.matmul(probs, present_value)

        merged = self.merge_heads(context)
        logits = self.lm_head(self.out_proj(merged))
        return logits, present_key, present_value


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


def build_inputs(batch_size, current_seq_len, past_seq_len, vocab_size, num_heads, head_dim):
    input_ids = torch.arange(batch_size * current_seq_len, dtype=torch.long).reshape(
        batch_size,
        current_seq_len,
    )
    input_ids = input_ids % vocab_size
    total_seq_len = past_seq_len + current_seq_len
    attention_mask = torch.ones((batch_size, total_seq_len), dtype=torch.long)
    past_key = torch.zeros(
        (batch_size, num_heads, past_seq_len, head_dim),
        dtype=torch.float32,
    )
    past_value = torch.zeros_like(past_key)
    return input_ids, attention_mask, past_key, past_value


def compare_array(torch_tensor, ort_array, tolerance):
    torch_np = torch_tensor.detach().cpu().numpy().astype(np.float32)
    ort_np = ort_array.astype(np.float32)
    diff = np.abs(torch_np - ort_np)
    return {
        "torch_shape": list(torch_np.shape),
        "onnx_shape": list(ort_np.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "tolerance": tolerance,
        "passed": bool(float(diff.max()) <= tolerance),
    }


def run_torch_step(model, input_ids, attention_mask, past_key, past_value):
    with torch.inference_mode():
        return model(input_ids, attention_mask, past_key, past_value)


def run_ort_step(session, input_ids, attention_mask, past_key, past_value):
    return session.run(
        ["logits", "present_key", "present_value"],
        {
            "input_ids": input_ids.cpu().numpy().astype(np.int64),
            "attention_mask": attention_mask.cpu().numpy().astype(np.int64),
            "past_key": past_key.cpu().numpy().astype(np.float32),
            "past_value": past_value.cpu().numpy().astype(np.float32),
        },
    )


def run_loop(session, model, args, initial_present_key, initial_present_value):
    torch_past_key = initial_present_key.detach()
    torch_past_value = initial_present_value.detach()
    ort_past_key = initial_present_key.detach().cpu().numpy().astype(np.float32)
    ort_past_value = initial_present_value.detach().cpu().numpy().astype(np.float32)
    shape_trace = []
    comparisons = []

    for step in range(args.loop_steps):
        token_id = torch.tensor([[step % args.vocab_size]], dtype=torch.long)
        total_seq_len = torch_past_key.shape[2] + 1
        attention_mask = torch.ones((args.batch_size, total_seq_len), dtype=torch.long)

        torch_logits, torch_present_key, torch_present_value = run_torch_step(
            model=model,
            input_ids=token_id,
            attention_mask=attention_mask,
            past_key=torch_past_key,
            past_value=torch_past_value,
        )
        ort_logits, ort_present_key, ort_present_value = session.run(
            ["logits", "present_key", "present_value"],
            {
                "input_ids": token_id.numpy().astype(np.int64),
                "attention_mask": attention_mask.numpy().astype(np.int64),
                "past_key": ort_past_key,
                "past_value": ort_past_value,
            },
        )

        comparisons.append(
            {
                "step": step,
                "logits": compare_array(torch_logits, ort_logits, args.tolerance),
                "present_key": compare_array(
                    torch_present_key,
                    ort_present_key,
                    args.tolerance,
                ),
                "present_value": compare_array(
                    torch_present_value,
                    ort_present_value,
                    args.tolerance,
                ),
            }
        )
        shape_trace.append(
            {
                "step": step,
                "past_key_shape": list(torch_past_key.shape),
                "present_key_shape": list(torch_present_key.shape),
                "cache_sequence_grew_by": int(
                    torch_present_key.shape[2] - torch_past_key.shape[2]
                ),
            }
        )

        torch_past_key = torch_present_key.detach()
        torch_past_value = torch_present_value.detach()
        ort_past_key = ort_present_key
        ort_past_value = ort_present_value

    all_passed = all(
        row["logits"]["passed"]
        and row["present_key"]["passed"]
        and row["present_value"]["passed"]
        for row in comparisons
    )

    return {
        "steps": args.loop_steps,
        "passed": all_passed,
        "shape_trace": shape_trace,
        "comparisons": comparisons,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--current-seq-len", type=int, default=1)
    parser.add_argument("--past-seq-len", type=int, default=4)
    parser.add_argument("--loop-steps", type=int, default=4)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provider", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--onnx-output",
        type=str,
        default="models/toy_kv_cache_causal_lm_smoke.onnx",
    )
    parser.add_argument("--output", type=str, default="results/toy_kv_cache_onnx_smoke.json")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.hidden_size != args.num_heads * args.head_dim:
        raise ValueError("--hidden-size must equal --num-heads * --head-dim.")

    model = ToyKVCachedCausalLM(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
    ).eval()
    input_ids, attention_mask, past_key, past_value = build_inputs(
        batch_size=args.batch_size,
        current_seq_len=args.current_seq_len,
        past_seq_len=args.past_seq_len,
        vocab_size=args.vocab_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
    )

    torch_logits, torch_present_key, torch_present_value = run_torch_step(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key=past_key,
        past_value=past_value,
    )

    onnx_path = Path(args.onnx_output)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (input_ids, attention_mask, past_key, past_value),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "past_key", "past_value"],
        output_names=["logits", "present_key", "present_value"],
        opset_version=args.opset,
        do_constant_folding=False,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "current_sequence"},
            "attention_mask": {0: "batch", 1: "total_sequence"},
            "past_key": {0: "batch", 2: "past_sequence"},
            "past_value": {0: "batch", 2: "past_sequence"},
            "logits": {0: "batch", 1: "current_sequence"},
            "present_key": {0: "batch", 2: "total_sequence"},
            "present_value": {0: "batch", 2: "total_sequence"},
        },
    )

    providers = choose_providers(args.provider)
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    ort_logits, ort_present_key, ort_present_value = run_ort_step(
        session=session,
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key=past_key,
        past_value=past_value,
    )

    single_step = {
        "logits": compare_array(torch_logits, ort_logits, args.tolerance),
        "present_key": compare_array(torch_present_key, ort_present_key, args.tolerance),
        "present_value": compare_array(
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

    loop = run_loop(
        session=session,
        model=model,
        args=args,
        initial_present_key=torch_present_key,
        initial_present_value=torch_present_value,
    )

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Toy KV-cache causal-LM smoke test for ONNX export and ONNX Runtime "
            "validation. This checks explicit past_key/past_value inputs, "
            "present_key/present_value outputs, and recurrent ORT cache reuse."
        ),
        "success": bool(single_step["passed"] and loop["passed"]),
        "model": {
            "type": "ToyKVCachedCausalLM",
            "vocab_size": args.vocab_size,
            "hidden_size": args.hidden_size,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
            "uses_attention_mask": True,
            "uses_past_present_kv_cache": True,
        },
        "export": {
            "onnx_path": str(onnx_path),
            "opset": args.opset,
            "input_names": ["input_ids", "attention_mask", "past_key", "past_value"],
            "output_names": ["logits", "present_key", "present_value"],
        },
        "runtime": {
            "requested_provider": args.provider,
            "providers": providers,
            "available_providers": ort.get_available_providers(),
        },
        "inputs": {
            "input_ids_shape": list(input_ids.shape),
            "attention_mask_shape": list(attention_mask.shape),
            "past_key_shape": list(past_key.shape),
            "past_value_shape": list(past_value.shape),
        },
        "single_step": single_step,
        "loop": loop,
        "note": (
            "Passing this smoke test does not mean Gemma4 KV-cache export works. "
            "It verifies the explicit ONNX past/present cache I/O pattern and "
            "ORT loop mechanics on an ONNX-friendly toy graph."
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print(f"Wrote toy KV-cache ONNX smoke result to {args.output}")
    print(f"Success: {result['success']}")
    print(f"Single-step max logit diff: {single_step['logits']['max_abs_diff']}")

    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
