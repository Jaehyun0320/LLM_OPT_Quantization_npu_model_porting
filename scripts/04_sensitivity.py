import argparse
import csv
import gc
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SENSITIVITY_EXPERIMENTS = {
    "full_quant": {
        "component": "all_linear_layers",
        "skip_modules": [],
        "notes": "All bitsandbytes-supported Linear layers are quantized.",
    },
    "keep_attention_fp": {
        "component": "attention_projections",
        "skip_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "notes": "Attention projection modules are kept in the requested floating dtype.",
    },
    "keep_mlp_fp": {
        "component": "mlp_projections",
        "skip_modules": ["gate_proj", "up_proj", "down_proj"],
        "notes": "MLP projection modules are kept in the requested floating dtype.",
    },
    "keep_lm_head_fp": {
        "component": "lm_head",
        "skip_modules": ["lm_head"],
        "notes": "Language modeling head is kept in the requested floating dtype.",
    },
    "keep_embeddings_fp": {
        "component": "embeddings",
        "skip_modules": ["embed_tokens"],
        "notes": (
            "bitsandbytes replaces Linear layers, so embedding layers may already "
            "remain unquantized; this row is kept as a documented control."
        ),
    },
}


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


def cuda_device_map(device):
    if device == "cuda":
        return {"": 0}
    if device.startswith("cuda:"):
        return {"": int(device.split(":", 1)[1])}
    raise ValueError(f"Expected CUDA device, got {device}")


def build_quantization_config(quantization, resolved_dtype, skip_modules):
    skip_modules = skip_modules or None

    if quantization == "int8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=skip_modules,
        )

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=resolved_dtype,
        bnb_4bit_use_double_quant=True,
        # Despite the name, the installed transformers 4-bit quantizer also
        # consumes this list as modules_to_not_convert.
        llm_int8_skip_modules=skip_modules,
    )


def load_quantized_model(
    model_id,
    quantization,
    resolved_dtype,
    skip_modules,
    device,
    attn_implementation,
):
    quantization_config = build_quantization_config(
        quantization=quantization,
        resolved_dtype=resolved_dtype,
        skip_modules=skip_modules,
    )
    load_kwargs = {
        "quantization_config": quantization_config,
        "torch_dtype": resolved_dtype,
        "device_map": cuda_device_map(device),
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if attn_implementation != "auto":
        load_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return model


def run_generation(model, tokenizer, inputs, max_new_tokens):
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return output_ids


def benchmark_generation(model, tokenizer, inputs, max_new_tokens, warmup_runs, num_runs):
    for _ in range(warmup_runs):
        _ = run_generation(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    latencies_ms = []
    tokens_per_second = []

    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output_ids = run_generation(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
        )
        torch.cuda.synchronize()
        elapsed_sec = time.perf_counter() - start

        input_token_count = inputs["input_ids"].shape[-1]
        output_token_count = output_ids.shape[-1]
        generated_token_count = output_token_count - input_token_count

        latencies_ms.append(elapsed_sec * 1000)
        tokens_per_second.append(generated_token_count / elapsed_sec)

    return {
        "latency_ms_mean": statistics.mean(latencies_ms),
        "latency_ms_std": statistics.stdev(latencies_ms)
        if len(latencies_ms) > 1
        else 0.0,
        "tps_mean": statistics.mean(tokens_per_second),
        "tps_std": statistics.stdev(tokens_per_second)
        if len(tokens_per_second) > 1
        else 0.0,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def compute_perplexity(model, tokenizer, eval_texts, device, max_length=256):
    losses = []
    for text in eval_texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=inputs["input_ids"],
            )

        losses.append(outputs.loss.detach().float().cpu())

    mean_loss = torch.stack(losses).mean()
    return torch.exp(mean_loss).item()


def summarize_quantized_modules(model):
    summary = {
        "linear8bit_count": 0,
        "linear4bit_count": 0,
        "torch_linear_count": 0,
    }

    for module in model.modules():
        class_name = module.__class__.__name__
        if class_name == "Linear8bitLt":
            summary["linear8bit_count"] += 1
        elif class_name == "Linear4bit":
            summary["linear4bit_count"] += 1
        elif isinstance(module, torch.nn.Linear):
            summary["torch_linear_count"] += 1

    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="google/gemma-4-E2B")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--quantization", type=str, default="int8", choices=["int8", "int4"])
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=",".join(SENSITIVITY_EXPERIMENTS.keys()),
        help="Comma-separated experiment names to run.",
    )
    parser.add_argument("--output", type=str, default="results/sensitivity.csv")
    return parser.parse_args()


args = parse_args()
device = resolve_device(args.device)
resolved_dtype = resolve_dtype(args.dtype, device)

if device != "cuda":
    raise RuntimeError(
        "04_sensitivity.py should be run on an NVIDIA CUDA GPU because it relies "
        "on bitsandbytes quantization and CUDA peak-memory measurements."
    )

experiment_names = [name.strip() for name in args.experiments.split(",") if name.strip()]
unknown_experiments = [
    name for name in experiment_names if name not in SENSITIVITY_EXPERIMENTS
]
if unknown_experiments:
    raise ValueError(
        f"Unknown experiments: {unknown_experiments}. "
        f"Available: {list(SENSITIVITY_EXPERIMENTS)}"
    )

tokenizer = load_tokenizer(args.model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

benchmark_prompt = "Deep learning is"
benchmark_inputs = tokenizer(benchmark_prompt, return_tensors="pt")
benchmark_inputs = {k: v.to(device) for k, v in benchmark_inputs.items()}

eval_texts = [
    "Deep learning models learn useful representations from data.",
    "Quantization reduces memory usage by representing weights with fewer bits.",
    "Autoregressive language models generate text one token at a time.",
]

rows = []

for experiment_name in experiment_names:
    experiment = SENSITIVITY_EXPERIMENTS[experiment_name]
    skip_modules = experiment["skip_modules"]

    model = load_quantized_model(
        model_id=args.model_id,
        quantization=args.quantization,
        resolved_dtype=resolved_dtype,
        skip_modules=skip_modules,
        device=device,
        attn_implementation=args.attn_implementation,
    )

    module_summary = summarize_quantized_modules(model)
    benchmark = benchmark_generation(
        model=model,
        tokenizer=tokenizer,
        inputs=benchmark_inputs,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        num_runs=args.num_runs,
    )
    perplexity = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        eval_texts=eval_texts,
        device=device,
    )

    rows.append(
        {
            "experiment": experiment_name,
            "component": experiment["component"],
            "quantization": args.quantization,
            "dtype_for_skipped_modules": str(resolved_dtype),
            "attn_implementation": args.attn_implementation,
            "skip_modules": ";".join(skip_modules),
            "perplexity": perplexity,
            "latency_ms_mean": benchmark["latency_ms_mean"],
            "latency_ms_std": benchmark["latency_ms_std"],
            "tps_mean": benchmark["tps_mean"],
            "tps_std": benchmark["tps_std"],
            "peak_cuda_memory_mb": benchmark["peak_cuda_memory_mb"],
            "linear8bit_count": module_summary["linear8bit_count"],
            "linear4bit_count": module_summary["linear4bit_count"],
            "torch_linear_count": module_summary["torch_linear_count"],
            "notes": experiment["notes"],
        }
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

fieldnames = [
    "experiment",
    "component",
    "quantization",
    "dtype_for_skipped_modules",
    "attn_implementation",
    "skip_modules",
    "perplexity",
    "latency_ms_mean",
    "latency_ms_std",
    "tps_mean",
    "tps_std",
    "peak_cuda_memory_mb",
    "linear8bit_count",
    "linear4bit_count",
    "torch_linear_count",
    "notes",
]

output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
