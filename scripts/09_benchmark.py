import argparse
import csv
import gc
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_CONFIGS = [
    "fp16:1:64:32",
    "fp16:1:64:128",
    "fp16:1:512:32",
    "fp16:1:512:128",
    "int8:1:64:32",
    "int8:1:64:128",
    "int8:1:512:32",
    "int8:1:512:128",
    "int4:1:64:32",
    "int4:1:64:128",
    "int4:1:512:32",
    "int4:1:512:128",
]

SYNTHETIC_PROMPT_SEEDS = [
    (
        "Deep learning systems optimize numerical representations, memory "
        "movement, and autoregressive decoding efficiency. "
    )
]

DIVERSE_PROMPT_SEEDS = [
    "Explain why the sky appears blue to a middle school student.",
    "Write a polite email asking for an extension on a project deadline.",
    "Summarize the following meeting notes into three action items.",
    "Compare electric vehicles and gasoline cars in terms of maintenance costs.",
    "A real-time chatbot should balance latency, accuracy, and memory usage.",
    "In a small coastal town, an engineer noticed that the old lighthouse signal had changed.",
    "Describe how a transformer model uses attention to combine information across tokens.",
    "List practical risks that can appear when deploying machine learning models on edge devices.",
]


def resolve_device(device_name):
    if device_name != "auto":
        return device_name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_from_precision(precision, device):
    if precision in {"int8", "int4"}:
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if device in {"cuda", "mps"} else torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unknown precision: {precision}")


def load_tokenizer(model_id):
    if model_id == "google/gemma-4-E2B":
        return AutoTokenizer.from_pretrained(
            model_id,
            extra_special_tokens={"video_token": "<|video|>"},
        )
    return AutoTokenizer.from_pretrained(model_id)


def build_quantization_config(precision, compute_dtype):
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if precision == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    return None


def load_model(model_id, precision, device):
    dtype = dtype_from_precision(precision, device)
    quantization_config = build_quantization_config(precision, dtype)

    if precision in {"int8", "int4"}:
        if device != "cuda":
            raise RuntimeError(
                f"{precision} benchmark uses bitsandbytes and requires CUDA."
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to(device)

    model.eval()
    return model, dtype


def hardware_string(device):
    if device == "cuda" and torch.cuda.is_available():
        parts = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            parts.append(
                f"cuda:{idx} {props.name} {props.total_memory / 1024**3:.1f}GB"
            )
        return "; ".join(parts)

    if device == "mps":
        return f"Apple Silicon MPS ({platform.machine()})"

    return f"CPU {platform.processor() or platform.machine()}"


def parse_config(config_string):
    precision, batch_size, input_seq_len, gen_len = config_string.split(":")
    return {
        "precision": precision,
        "batch_size": int(batch_size),
        "input_seq_len": int(input_seq_len),
        "gen_len": int(gen_len),
    }


def make_prompt_from_seed(tokenizer, seed_text, input_seq_len):
    token_ids = tokenizer(seed_text, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise ValueError("Tokenizer produced no tokens for benchmark seed text.")

    repeated = []
    while len(repeated) < input_seq_len:
        repeated.extend(token_ids)
    repeated = repeated[:input_seq_len]
    return tokenizer.decode(repeated, skip_special_tokens=True)


def select_prompt_seeds(prompt_mode, prompt_variants):
    if prompt_mode == "synthetic":
        seeds = SYNTHETIC_PROMPT_SEEDS
    elif prompt_mode == "diverse":
        seeds = DIVERSE_PROMPT_SEEDS
    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")

    if prompt_variants < 1:
        raise ValueError("--prompt-variants must be at least 1.")

    return seeds[: min(prompt_variants, len(seeds))]


def make_prompts(tokenizer, input_seq_len, prompt_mode, prompt_variants):
    seeds = select_prompt_seeds(prompt_mode, prompt_variants)
    return [
        make_prompt_from_seed(tokenizer, seed_text, input_seq_len)
        for seed_text in seeds
    ]


def prepare_inputs(tokenizer, prompt, batch_size, device):
    prompts = [prompt] * batch_size
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return {key: value.to(device) for key, value in inputs.items()}


def synchronize_if_needed(device):
    if device == "cuda":
        torch.cuda.synchronize()


def reset_peak_memory_if_needed(device):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device):
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    return None


def run_prefill(model, inputs, use_cache=True):
    with torch.inference_mode():
        return model(
            **inputs,
            use_cache=use_cache,
        )


def select_next_token(outputs):
    return outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)


def measure_ttft(model, inputs, device):
    synchronize_if_needed(device)
    start = time.perf_counter()
    prefill_outputs = run_prefill(model, inputs, use_cache=True)
    first_token_ids = select_next_token(prefill_outputs)
    synchronize_if_needed(device)
    ttft_sec = time.perf_counter() - start
    return prefill_outputs, first_token_ids, ttft_sec


def run_decode_from_prefill(
    model,
    prefill_outputs,
    first_token_ids,
    inputs,
    decode_steps,
    device,
):
    past_key_values = prefill_outputs.past_key_values
    next_input_ids = first_token_ids
    attention_mask = inputs["attention_mask"]
    generated_tokens = 0

    for _ in range(decode_steps):
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

        with torch.inference_mode():
            outputs = model(
                input_ids=next_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        past_key_values = outputs.past_key_values
        next_input_ids = select_next_token(outputs)
        generated_tokens += int(next_input_ids.numel())

    return generated_tokens


def benchmark_config(
    model,
    tokenizer,
    config,
    device,
    warmup_runs,
    num_runs,
    prompt_mode,
    prompt_variants,
):
    prompts = make_prompts(
        tokenizer=tokenizer,
        input_seq_len=config["input_seq_len"],
        prompt_mode=prompt_mode,
        prompt_variants=prompt_variants,
    )

    prepared_inputs = [
        prepare_inputs(
            tokenizer=tokenizer,
            prompt=prompt,
            batch_size=config["batch_size"],
            device=device,
        )
        for prompt in prompts
    ]

    for inputs in prepared_inputs:
        for _ in range(warmup_runs):
            prefill_outputs, first_token_ids, _ = measure_ttft(model, inputs, device)
            _ = run_decode_from_prefill(
                model=model,
                prefill_outputs=prefill_outputs,
                first_token_ids=first_token_ids,
                inputs=inputs,
                decode_steps=min(max(config["gen_len"] - 1, 0), 4),
                device=device,
            )

    reset_peak_memory_if_needed(device)

    ttft_ms = []
    tps_values = []

    for inputs in prepared_inputs:
        for _ in range(num_runs):
            prefill_outputs, first_token_ids, ttft_sec = measure_ttft(
                model=model,
                inputs=inputs,
                device=device,
            )

            synchronize_if_needed(device)
            decode_start = time.perf_counter()
            generated_tokens = run_decode_from_prefill(
                model=model,
                prefill_outputs=prefill_outputs,
                first_token_ids=first_token_ids,
                inputs=inputs,
                decode_steps=max(config["gen_len"] - 1, 0),
                device=device,
            )
            synchronize_if_needed(device)
            decode_sec = time.perf_counter() - decode_start

            ttft_ms.append(ttft_sec * 1000)
            tps_values.append(generated_tokens / decode_sec if decode_sec > 0 else 0.0)

    return {
        "ttft_ms_mean": statistics.mean(ttft_ms),
        "ttft_ms_std": statistics.stdev(ttft_ms) if len(ttft_ms) > 1 else 0.0,
        "tps_mean": statistics.mean(tps_values),
        "tps_std": statistics.stdev(tps_values) if len(tps_values) > 1 else 0.0,
        "peak_mem_mb": peak_memory_mb(device),
        "raw_ttft_ms": ";".join(f"{value:.4f}" for value in ttft_ms),
        "raw_tps": ";".join(f"{value:.4f}" for value in tps_values),
        "prompt_count": len(prompts),
        "measurement_count": len(ttft_ms),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="google/gemma-4-E2B")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="synthetic",
        choices=["synthetic", "diverse"],
        help=(
            "synthetic uses one technical seed. diverse uses multiple ordinary "
            "prompt seeds, each repeated/truncated to input_seq_len."
        ),
    )
    parser.add_argument(
        "--prompt-variants",
        type=int,
        default=1,
        help=(
            "Number of prompt seeds to use per config. In diverse mode, values "
            "larger than 1 increase runtime proportionally."
        ),
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=",".join(DEFAULT_CONFIGS),
        help=(
            "Comma-separated precision:batch_size:input_seq_len:gen_len entries. "
            "Example: fp16:1:64:32,int8:1:512:128"
        ),
    )
    parser.add_argument("--output", type=str, default="results/benchmark.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    tokenizer = load_tokenizer(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    configs = [parse_config(item.strip()) for item in args.configs.split(",") if item.strip()]
    hardware = hardware_string(device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "precision",
        "batch_size",
        "input_seq_len",
        "gen_len",
        "prompt_mode",
        "prompt_variants",
        "measurement_count",
        "ttft_ms_mean",
        "ttft_ms_std",
        "tps_mean",
        "tps_std",
        "peak_mem_mb",
        "hardware",
        "notes",
        "created_at",
        "raw_ttft_ms",
        "raw_tps",
    ]

    rows = []
    precisions = []
    for config in configs:
        if config["precision"] not in precisions:
            precisions.append(config["precision"])

    for precision in precisions:
        model = None
        try:
            model, dtype = load_model(args.model_id, precision, device)
            precision_configs = [
                config for config in configs if config["precision"] == precision
            ]

            for config in precision_configs:
                try:
                    metrics = benchmark_config(
                        model=model,
                        tokenizer=tokenizer,
                        config=config,
                        device=device,
                        warmup_runs=args.warmup_runs,
                        num_runs=args.num_runs,
                        prompt_mode=args.prompt_mode,
                        prompt_variants=args.prompt_variants,
                    )
                    notes = (
                        f"dtype={dtype}; greedy_manual_decode; use_cache=True; "
                        "length_controlled_prompts"
                    )
                    row = {
                        "model": args.model_id,
                        "precision": precision,
                        "batch_size": config["batch_size"],
                        "input_seq_len": config["input_seq_len"],
                        "gen_len": config["gen_len"],
                        "prompt_mode": args.prompt_mode,
                        "prompt_variants": metrics["prompt_count"],
                        "measurement_count": metrics["measurement_count"],
                        "ttft_ms_mean": metrics["ttft_ms_mean"],
                        "ttft_ms_std": metrics["ttft_ms_std"],
                        "tps_mean": metrics["tps_mean"],
                        "tps_std": metrics["tps_std"],
                        "peak_mem_mb": metrics["peak_mem_mb"],
                        "hardware": hardware,
                        "notes": notes,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "raw_ttft_ms": metrics["raw_ttft_ms"],
                        "raw_tps": metrics["raw_tps"],
                    }
                except Exception as exc:
                    row = {
                        "model": args.model_id,
                        "precision": precision,
                        "batch_size": config["batch_size"],
                        "input_seq_len": config["input_seq_len"],
                        "gen_len": config["gen_len"],
                        "prompt_mode": args.prompt_mode,
                        "prompt_variants": args.prompt_variants,
                        "measurement_count": "",
                        "ttft_ms_mean": "",
                        "ttft_ms_std": "",
                        "tps_mean": "",
                        "tps_std": "",
                        "peak_mem_mb": "",
                        "hardware": hardware,
                        "notes": f"FAILED: {type(exc).__name__}: {exc}",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "raw_ttft_ms": "",
                        "raw_tps": "",
                    }

                rows.append(row)
                with output_path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

        finally:
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    print(f"Wrote benchmark CSV to {args.output}")


if __name__ == "__main__":
    main()
