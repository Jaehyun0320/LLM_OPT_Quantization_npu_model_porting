import argparse
import json
import statistics
import time
import os
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

def cuda_device_map(device):
    if device == "cuda":
        return {"": 0}
    if device.startswith("cuda:"):
        return {"": int(device.split(":", 1)[1])}
    raise ValueError(f"Expected CUDA device, got {device}")

def cuda_device_index(device):
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"Expected CUDA device, got {device}")

def build_max_memory(device, max_gpu_memory, max_cpu_memory):
    return {
        cuda_device_index(device): max_gpu_memory,
        "cpu": max_cpu_memory,
    }

def load_model(
    model_id,
    dtype,
    device,
    attn_implementation,
    load_device_map,
    max_gpu_memory,
    max_cpu_memory,
    offload_folder,
):
    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation != "auto":
        load_kwargs["attn_implementation"] = attn_implementation

    if device.startswith("cuda"):
        if load_device_map == "auto":
            offload_path = Path(offload_folder)
            offload_path.mkdir(parents=True, exist_ok=True)
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = build_max_memory(
                device=device,
                max_gpu_memory=max_gpu_memory,
                max_cpu_memory=max_cpu_memory,
            )
            load_kwargs["offload_folder"] = str(offload_path)
            load_kwargs["offload_state_dict"] = True
        else:
            load_kwargs["device_map"] = cuda_device_map(device)
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model = model.to(device)

    model.eval()
    return model

def get_process_memory_mb():
    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024**2,
            "vms_mb": memory_info.vms / 1024**2,
        }
    except Exception:
        return {
            "rss_mb": None,
            "vms_mb": None,
        }

def get_cuda_memory_mb(device):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {
            "cuda_allocated_mb": None,
            "cuda_reserved_mb": None,
            "cuda_max_allocated_mb": None,
        }

    return {
        "cuda_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "cuda_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "cuda_max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }

def memory_snapshot(stage, device):
    snapshot = {
        "stage": stage,
        "time": datetime.now(timezone.utc).isoformat(),
        **get_process_memory_mb(),
        **get_cuda_memory_mb(device),
    }
    return snapshot

def log_memory(stage, device, memory_trace, enabled):
    snapshot = memory_snapshot(stage, device)
    memory_trace.append(snapshot)

    if enabled:
        printable = ", ".join(
            f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in snapshot.items()
        )
        print(f"[memory] {printable}", flush=True)

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
parser.add_argument(
    "--attn-implementation",
    type=str,
    default="eager",
    choices=["auto", "eager", "sdpa", "flash_attention_2"],
    help=(
        "Attention backend passed to Transformers. The default eager path avoids "
        "Gemma4 SDPA enable_gqa compatibility issues on older PyTorch builds."
    ),
)
parser.add_argument(
    "--log-memory",
    action="store_true",
    help="Print CPU RSS and CUDA memory checkpoints during baseline execution.",
)
parser.add_argument(
    "--load-device-map",
    type=str,
    default="auto",
    choices=["cuda", "auto"],
    help=(
        "CUDA loading strategy. auto uses Accelerate device_map with memory "
        "budgets and optional CPU/disk offload; cuda loads the full model "
        "directly on one CUDA device."
    ),
)
parser.add_argument(
    "--max-gpu-memory",
    type=str,
    default="70GiB",
    help="GPU memory budget used when --load-device-map auto is selected.",
)
parser.add_argument(
    "--max-cpu-memory",
    type=str,
    default="80GiB",
    help="CPU memory budget used when --load-device-map auto is selected.",
)
parser.add_argument(
    "--offload-folder",
    type=str,
    default="models/offload/baseline",
    help="Disk offload directory used when --load-device-map auto is selected.",
)

args = parser.parse_args()
device = resolve_device(args.device)
resolved_dtype = resolve_dtype(args.dtype, device)
memory_trace = []
log_memory("start", device, memory_trace, args.log_memory)

model_id = args.model_id  ### google/gemma-4-E2B is defalut

tokenizer = load_tokenizer(model_id)
log_memory("after_tokenizer_load", device, memory_trace, args.log_memory)
model = load_model(
    model_id=model_id,
    dtype=resolved_dtype,
    device=device,
    attn_implementation=args.attn_implementation,
    load_device_map=args.load_device_map,
    max_gpu_memory=args.max_gpu_memory,
    max_cpu_memory=args.max_cpu_memory,
    offload_folder=args.offload_folder,
)
log_memory("after_model_load", device, memory_trace, args.log_memory)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
log_memory("after_pad_token_setup", device, memory_trace, args.log_memory)

""" For warm-up, latency check"""
benchmark_prompt = "Deep learning is"
inputs = tokenizer(benchmark_prompt, return_tensors = "pt")
log_memory("after_prompt_tokenize_cpu", device, memory_trace, args.log_memory)
inputs = {k: v.to(device) for k, v in inputs.items()}
log_memory("after_prompt_inputs_to_device", device, memory_trace, args.log_memory)

""" Warm up """
for _ in range(args.warmup_runs):
    log_memory("before_warmup_generate", device, memory_trace, args.log_memory)
    _ = run_generation(
        model = model,
        tokenizer = tokenizer,
        inputs = inputs,
        max_new_tokens = args.max_new_tokens,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    log_memory("after_warmup_generate", device, memory_trace, args.log_memory)

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
    "attn_implementation": args.attn_implementation,
    "model_loading": {
        "load_device_map": args.load_device_map,
        "max_gpu_memory": args.max_gpu_memory,
        "max_cpu_memory": args.max_cpu_memory,
        "offload_folder": args.offload_folder,
        "dtype": str(resolved_dtype),
        "attn_implementation": args.attn_implementation,
    },
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
        "trace": memory_trace,
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
