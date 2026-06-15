import argparse
import contextlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import candidate_generator as hf_candidate_generator


DEFAULT_ASSISTANT_MODELS = [
    "google/gemma-3-270m-it",
    "google/gemma-3-270m",
    "google/gemma-3-1b-it",
]

PROBE_TEXTS = [
    "Explain quantization in one sentence:",
    "Why does KV-cache improve autoregressive decoding?",
    "A real-time chatbot should optimize for",
]


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


def resolve_device(device_name):
    if device_name != "auto":
        return device_name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    model = model.to(device)
    model.eval()
    return model


def safe_model_slug(model_id):
    return model_id.replace("/", "__").replace(":", "_")


def prepare_inputs(tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def decode_new_text(tokenizer, output_ids, input_token_count):
    return tokenizer.decode(output_ids[0][input_token_count:], skip_special_tokens=True)


def summarize_tokenizer_compatibility(target_tokenizer, assistant_tokenizer):
    target_encodings = {text: target_tokenizer.encode(text) for text in PROBE_TEXTS}
    assistant_encodings = {text: assistant_tokenizer.encode(text) for text in PROBE_TEXTS}
    return {
        "target_tokenizer_class": target_tokenizer.__class__.__name__,
        "assistant_tokenizer_class": assistant_tokenizer.__class__.__name__,
        "target_vocab_size": getattr(target_tokenizer, "vocab_size", None),
        "assistant_vocab_size": getattr(assistant_tokenizer, "vocab_size", None),
        "target_len": len(target_tokenizer),
        "assistant_len": len(assistant_tokenizer),
        "same_probe_encodings": target_encodings == assistant_encodings,
        "target_probe_encodings": target_encodings,
        "assistant_probe_encodings": assistant_encodings,
    }


@contextlib.contextmanager
def capture_assisted_decoding_stats():
    stats = {
        "iterations": [],
        "proposed_tokens": 0,
        "accepted_tokens": 0,
        "rejected_tokens": 0,
    }
    original_update = hf_candidate_generator.AssistedCandidateGenerator.update_candidate_strategy

    def wrapped_update(self, input_ids, scores, num_matches):
        proposed = max(int(scores.shape[1]) - 1, 0)
        accepted = int(num_matches)
        rejected = max(proposed - accepted, 0)
        stats["iterations"].append(
            {
                "proposed_tokens": proposed,
                "accepted_tokens": accepted,
                "rejected_tokens": rejected,
            }
        )
        stats["proposed_tokens"] += proposed
        stats["accepted_tokens"] += accepted
        stats["rejected_tokens"] += rejected
        return original_update(self, input_ids, scores, num_matches)

    hf_candidate_generator.AssistedCandidateGenerator.update_candidate_strategy = wrapped_update
    try:
        yield stats
    finally:
        hf_candidate_generator.AssistedCandidateGenerator.update_candidate_strategy = original_update


def run_generation(
    model,
    tokenizer,
    inputs,
    max_new_tokens,
    use_cache,
    assistant_model=None,
    assistant_tokenizer=None,
):
    generate_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": use_cache,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if assistant_model is not None:
        generate_kwargs["assistant_model"] = assistant_model
        if assistant_tokenizer is not None:
            generate_kwargs["tokenizer"] = tokenizer
            generate_kwargs["assistant_tokenizer"] = assistant_tokenizer

    with torch.inference_mode():
        return model.generate(**generate_kwargs)


def benchmark_generation(
    model,
    tokenizer,
    inputs,
    device,
    max_new_tokens,
    warmup_runs,
    num_runs,
    use_cache,
    assistant_model=None,
    assistant_tokenizer=None,
):
    input_token_count = inputs["input_ids"].shape[-1]

    for _ in range(warmup_runs):
        _ = run_generation(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            assistant_model=assistant_model,
            assistant_tokenizer=assistant_tokenizer,
        )

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    latencies_ms = []
    tokens_per_second = []
    generated_token_counts = []
    outputs = []
    assisted_stats_by_run = []

    for _ in range(num_runs):
        if assistant_model is not None:
            stats_context = capture_assisted_decoding_stats()
        else:
            stats_context = contextlib.nullcontext(None)

        with stats_context as assisted_stats:
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            output_ids = run_generation(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                assistant_model=assistant_model,
                assistant_tokenizer=assistant_tokenizer,
            )
            if device == "cuda":
                torch.cuda.synchronize()

            elapsed_sec = time.perf_counter() - start

        output_token_count = output_ids.shape[-1]
        generated_token_count = output_token_count - input_token_count

        latencies_ms.append(elapsed_sec * 1000)
        tokens_per_second.append(generated_token_count / elapsed_sec)
        generated_token_counts.append(generated_token_count)
        outputs.append(decode_new_text(tokenizer, output_ids, input_token_count))

        if assisted_stats is not None:
            assisted_stats_by_run.append(
                {
                    "proposed_tokens": assisted_stats["proposed_tokens"],
                    "accepted_tokens": assisted_stats["accepted_tokens"],
                    "rejected_tokens": assisted_stats["rejected_tokens"],
                    "reject_rate": (
                        assisted_stats["rejected_tokens"] / assisted_stats["proposed_tokens"]
                        if assisted_stats["proposed_tokens"]
                        else None
                    ),
                    "accept_rate": (
                        assisted_stats["accepted_tokens"] / assisted_stats["proposed_tokens"]
                        if assisted_stats["proposed_tokens"]
                        else None
                    ),
                    "iterations": assisted_stats["iterations"],
                }
            )

    aggregate_assisted_stats = None
    if assisted_stats_by_run:
        proposed = sum(row["proposed_tokens"] for row in assisted_stats_by_run)
        accepted = sum(row["accepted_tokens"] for row in assisted_stats_by_run)
        rejected = sum(row["rejected_tokens"] for row in assisted_stats_by_run)
        aggregate_assisted_stats = {
            "proposed_tokens": proposed,
            "accepted_tokens": accepted,
            "rejected_tokens": rejected,
            "reject_rate": rejected / proposed if proposed else None,
            "accept_rate": accepted / proposed if proposed else None,
            "raw_by_run": assisted_stats_by_run,
            "note": (
                "Rates are captured from Transformers AssistedCandidateGenerator. "
                "They count draft tokens proposed by the assistant and matched by "
                "the target during greedy assisted generation."
            ),
        }

    return {
        "use_cache": use_cache,
        "latency_ms_mean": statistics.mean(latencies_ms),
        "latency_ms_std": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "tokens_per_second_mean": statistics.mean(tokens_per_second),
        "tokens_per_second_std": statistics.stdev(tokens_per_second)
        if len(tokens_per_second) > 1
        else 0.0,
        "generated_tokens_mean": statistics.mean(generated_token_counts),
        "peak_cuda_memory_mb": (
            torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else None
        ),
        "raw_latency_ms": latencies_ms,
        "raw_tokens_per_second": tokens_per_second,
        "raw_generated_token_counts": generated_token_counts,
        "sample_output": outputs[-1] if outputs else "",
        "assisted_decoding": aggregate_assisted_stats,
    }


def compute_speedup(reference, candidate):
    return {
        "latency_speedup": reference["latency_ms_mean"] / candidate["latency_ms_mean"],
        "tps_speedup": candidate["tokens_per_second_mean"]
        / reference["tokens_per_second_mean"],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="google/gemma-4-E2B")
    parser.add_argument(
        "--assistant-models",
        type=str,
        default=",".join(DEFAULT_ASSISTANT_MODELS),
        help="Comma-separated assistant model ids. Used when --run-speculative is set.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--benchmark-prompt", type=str, default="Deep learning is")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--run-kv-cache", action="store_true")
    parser.add_argument("--run-speculative", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/optimize")
    return parser.parse_args()


args = parse_args()
device = resolve_device(args.device)
resolved_dtype = resolve_dtype(args.dtype, device)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

target_tokenizer = load_tokenizer(args.model_id)
if target_tokenizer.pad_token is None:
    target_tokenizer.pad_token = target_tokenizer.eos_token

target_model = load_model(args.model_id, resolved_dtype, device)
target_inputs = prepare_inputs(target_tokenizer, args.benchmark_prompt, device)

if not args.run_kv_cache and not args.run_speculative:
    args.run_kv_cache = True

base_metadata = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "model_id": args.model_id,
    "device": device,
    "dtype": str(resolved_dtype),
    "benchmark_prompt": args.benchmark_prompt,
    "generation_config": {
        "max_new_tokens": args.max_new_tokens,
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "do_sample": False,
    },
}

kv_cache_reference = None
if args.run_kv_cache or args.run_speculative:
    without_cache = benchmark_generation(
        model=target_model,
        tokenizer=target_tokenizer,
        inputs=target_inputs,
        device=device,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        num_runs=args.num_runs,
        use_cache=False,
    )
    with_cache = benchmark_generation(
        model=target_model,
        tokenizer=target_tokenizer,
        inputs=target_inputs,
        device=device,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        num_runs=args.num_runs,
        use_cache=True,
    )
    kv_cache_reference = with_cache

    if args.run_kv_cache:
        kv_result = {
            **base_metadata,
            "optimization": {
                "method": "kv_cache",
                "baseline": "use_cache_false",
                "optimized": "use_cache_true",
            },
            "results": {
                "without_cache": without_cache,
                "with_cache": with_cache,
            },
            "impact": compute_speedup(without_cache, with_cache),
        }
        kv_output_path = output_dir / "kv_cache_on_off.json"
        kv_output_path.write_text(json.dumps(kv_result, indent=2) + "\n")

if args.run_speculative:
    assistant_model_ids = [
        item.strip() for item in args.assistant_models.split(",") if item.strip()
    ]

    for assistant_model_id in assistant_model_ids:
        assistant_tokenizer = load_tokenizer(assistant_model_id)
        if assistant_tokenizer.pad_token is None:
            assistant_tokenizer.pad_token = assistant_tokenizer.eos_token

        compatibility = summarize_tokenizer_compatibility(
            target_tokenizer,
            assistant_tokenizer,
        )

        assistant_model = load_model(assistant_model_id, resolved_dtype, device)

        speculative_with_cache = benchmark_generation(
            model=target_model,
            tokenizer=target_tokenizer,
            inputs=target_inputs,
            device=device,
            max_new_tokens=args.max_new_tokens,
            warmup_runs=args.warmup_runs,
            num_runs=args.num_runs,
            use_cache=True,
            assistant_model=assistant_model,
            assistant_tokenizer=assistant_tokenizer,
        )

        speculative_without_cache = benchmark_generation(
            model=target_model,
            tokenizer=target_tokenizer,
            inputs=target_inputs,
            device=device,
            max_new_tokens=args.max_new_tokens,
            warmup_runs=args.warmup_runs,
            num_runs=args.num_runs,
            use_cache=False,
            assistant_model=assistant_model,
            assistant_tokenizer=assistant_tokenizer,
        )

        result = {
            **base_metadata,
            "optimization": {
                "method": "speculative_decoding",
                "assistant_model_id": assistant_model_id,
                "assistant_model_note": (
                    "Speculative decoding is run with greedy decoding and target/"
                    "assistant tokenizers passed explicitly."
                ),
            },
            "tokenizer_compatibility": compatibility,
            "results": {
                "target_without_cache": without_cache,
                "target_with_cache": kv_cache_reference,
                "speculative_without_cache": speculative_without_cache,
                "speculative_with_cache": speculative_with_cache,
            },
            "impact_vs_target_with_cache": compute_speedup(
                kv_cache_reference,
                speculative_with_cache,
            ),
            "impact_vs_target_without_cache": compute_speedup(
                without_cache,
                speculative_with_cache,
            ),
        }

        assistant_output_path = (
            output_dir / f"speculative__{safe_model_slug(assistant_model_id)}.json"
        )
        assistant_output_path.write_text(json.dumps(result, indent=2) + "\n")

        del assistant_model
        if device == "cuda":
            torch.cuda.empty_cache()
