import argparse
import json
import statistics
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from datetime import datetime, timezone

def resolve_dtype(dtype_name, device):
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32

    # auto
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

def run_generation(model, tokenizer, inputs, max_new_tokens):
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample = False,
            pad_token_id = tokenizer.eos_token_id,
        )
    return output_ids

def compute_perplexity(model, tokenizer, eval_texts, device, max_length = 256):
    losses = []
    for text in eval_texts:
        inputs = tokenizer(
            text,
            return_tensors = "pt",
            truncation = True,
            max_length = max_length,
        )
        inputs = {k : v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(
                input_ids = inputs["input_ids"],
                attention_mask = inputs.get("attention_mask"),
                labels = inputs["input_ids"],
            )
        
        losses.append(outputs.loss.detach().float().cpu())
    
    mean_loss = torch.stack(losses).mean()
    perplexity = torch.exp(mean_loss).item()
    return perplexity

""" Parsing arguments """
parser = argparse.ArgumentParser()

parser.add_argument('--model-id', type=str, default = "google/gemma-4-E2B")
parser.add_argument('--device', type=str, default="auto")
parser.add_argument('--dtype', type=str, default="auto")
parser.add_argument('--max-new-tokens', type=int, default=30)
parser.add_argument('--num-runs', type=int, default=5)
parser.add_argument('--warmup-runs', type=int, default=1)
parser.add_argument('--output', type=str, default="results/baseline.json")

args = parser.parse_args()
device = resolve_device(args.device)
resolved_dtype = resolve_dtype(args.dtype, device)

model_id = args.model_id  ### google/gemma-4-E2B is defalut

tokenizer = load_tokenizer(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype = resolved_dtype,
    trust_remote_code = True,
)
model = model.to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

""" For warm-up, latency check"""
benchmark_prompt = "Deep learning is"
inputs = tokenizer(benchmark_prompt, return_tensors = "pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

""" Warm up """
for _ in range(args.warmup_runs):
    _ = run_generation(
        model = model,
        tokenizer = tokenizer,
        inputs = inputs,
        max_new_tokens = args.max_new_tokens,
    )

if device == "cuda":
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

""" Time check """
latencies_ms = []
tokens_per_second = []

for _ in range(args.num_runs):
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    output_ids = run_generation(
        model = model,
        tokenizer = tokenizer,
        inputs = inputs,
        max_new_tokens = args.max_new_tokens,
    )
    if device == "cuda":
        torch.cuda.synchronize()

    elapsed_sec = time.perf_counter() - start

    input_token_count = inputs["input_ids"].shape[-1]
    output_token_count = output_ids.shape[-1]
    generated_token_count = output_token_count - input_token_count

    latency_ms = elapsed_sec * 1000
    tps = generated_token_count / elapsed_sec
    
    latencies_ms.append(latency_ms)
    tokens_per_second.append(tps)

latency_ms_mean = statistics.mean(latencies_ms)
latency_ms_std = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0

tps_mean = statistics.mean(tokens_per_second)
tps_std = statistics.stdev(tokens_per_second) if len(tokens_per_second) > 1 else 0.0

peak_cuda_memory_mb = (
    torch.cuda.max_memory_allocated() / 1024 ** 2
    if device == "cuda"
    else None
)

"""model size calculation"""
total_params = sum(p.numel() for p in model.parameters())
param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2

#""" Generated text """
#generated_text = tokenizer.decode(output_ids[0], skip_special_tokens = True)

""" Perplexity """
eval_texts = [
    "Deep learning models learn useful representations from data.",
    "Quantization reduces memory usage by representing weights with fewer bits.",
    "Autoregressive language models generate text one token at a time.",
]
perplexity = compute_perplexity(
    model = model,
    tokenizer = tokenizer,
    eval_texts = eval_texts,
    device = device,
)

""" Human_readable sanity check """
sample_prompts = [
    "Explain quantization in one sentence:",
    "Why does KV-cache improve autoregressive decoding?",
    "A real-time chatbot should optimize for",
]

sample_outputs = []

for sample_prompt in sample_prompts:
    sample_inputs = tokenizer(sample_prompt, return_tensors="pt")
    sample_inputs = {k : v.to(device) for k, v in sample_inputs.items()}

    sample_output_ids = run_generation(
        model=model,
        tokenizer=tokenizer,
        inputs=sample_inputs,
        max_new_tokens=args.max_new_tokens,
    )

    sample_outputs.append({
        "prompt": sample_prompt,
        "output": tokenizer.decode(sample_output_ids[0], skip_special_tokens=True),
    })

""" Result format """
result = {
    "created_at" : datetime.now(timezone.utc).isoformat(),
    "model_id" : args.model_id,
    "device" : device, 
    "dtype" : str(resolved_dtype),
    "benchmark_prompt" : benchmark_prompt,
    "generation_config" : {
        "max_new_tokens" : args.max_new_tokens,
        "num_runs" : args.num_runs,
        "warmup_runs" : args.warmup_runs,
        "do_sample" : False,
    },
    "model_size" : {
        "total_params" : total_params,
        "param_size_mb": param_size_mb,
    },
    "memory": {
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
    },
    "latency": {
        "latency_ms_mean": latency_ms_mean,
        "latency_ms_std": latency_ms_std,
        "tokens_per_second_mean": tps_mean,
        "tokens_per_second_std": tps_std,
        "raw_latency_ms": latencies_ms,
        "raw_tokens_per_second": tokens_per_second,
    },
    "quality" : {
        "metric": "perplexity",
        "perplexity": perplexity,
        "eval_num_texts": len(eval_texts),
        "eval_max_length": 256,
    },
    "sample_outputs": sample_outputs,
}

""" Store """
output_path = Path(args.output)
output_path.parent.mkdir(parents = True, exist_ok = True)
output_path.write_text(json.dumps(result, indent = 2) + "\n")
