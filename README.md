# LLM Optimization, Quantization, and Edge Porting

This repository contains an assignment project for evaluating `google/gemma-4-E2B` across quantization, decoding optimization, ONNX export/validation, Apple MLX edge conversion, and throughput benchmarking. The code is organized as standalone scripts so that each experiment can be run independently on the appropriate hardware.

## Project Scope

- Baseline inference and environment checks.
- INT8 and INT4 quantization with bitsandbytes.
- Quantization sensitivity analysis through skip-module policies.
- KV-cache and speculative decoding benchmarks.
- ONNX export/validation attempts, plus toy ONNX smoke tests for no-cache and KV-cache paths.
- Apple MLX conversion attempts for edge/mobile deployment.
- Throughput benchmarking for TTFT, decode TPS, and peak memory.

## Hardware Requirements

The full target experiments are designed for a CUDA GPU server, ideally an A100-class NVIDIA GPU, because Gemma 4 E2B and bitsandbytes INT8/INT4 quantization require substantial GPU memory and CUDA support.

Apple Silicon is only required for the MLX edge-format experiments. The MLX retry environment uses Python 3.12 with `mlx-lm==0.31.3` and `mlx==0.31.2`. An 8GB MacBook may still fail on Gemma 4 conversion because conversion can require more unified memory than the final 4-bit model.

CPU or Apple MPS local runs are useful only as smoke tests with tiny models. They are not representative of Gemma 4 E2B latency, memory, or quantization behavior.

## Repository Layout

```text
scripts/00_check_env.py              Environment and package checks
scripts/00_prime_model_cache.py      Hugging Face model cache warm-up
scripts/01_baseline.py               Baseline inference benchmark
scripts/02_quantize_int8.py          bitsandbytes INT8 inference benchmark
scripts/03_quantize_int4.py          bitsandbytes INT4 NF4 inference benchmark
scripts/04_sensitivity.py            Quantization skip-module sensitivity
scripts/05_optimize_compile.py       KV-cache and speculative decoding experiments
scripts/06_export_onnx.py            Hugging Face ONNX export attempt
scripts/06_1_toy_onnx_smoke.py       Toy no-cache ONNX export + ORT smoke test
scripts/06_2_toy_kv_cache_onnx_smoke.py Toy KV-cache ONNX export + ORT smoke test
scripts/07_validate_onnx.py          Hugging Face ONNX Runtime validation
scripts/07_1_validate_toy_onnx_smoke.py Toy ONNX validation-only smoke test
scripts/08_convert_edge.py           Apple MLX conversion/generation wrapper
scripts/08_1_convert_edge_smoke.py   Gemma 3 MLX smoke test
scripts/09_benchmark.py              TTFT/TPS/peak-memory benchmark harness
```

The main reports are:

```text
report/quantization_report.md
report/onnx_edgeformat_report.md
report/benchmark_report.md
```

## Setup: CUDA / ONNX / Quantization

Create a regular Python environment and install the main dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For Gemma models, authenticate with a Hugging Face account that has accepted the gated model terms:

```bash
hf auth login
hf auth whoami
```

or:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

The token must belong to the Hugging Face account with Gemma access.

## Setup: Apple MLX

MLX is split into a separate requirements file because the Gemma 4 retry path needs Python 3.12+ and recent MLX packages. Do not commit the `.venv-mlx` directory.

```bash
/opt/homebrew/bin/python3.12 -m venv .venv-mlx
source .venv-mlx/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-mlx.txt
```

Expected versions:

```text
Python 3.12.x
mlx-lm 0.31.3
mlx 0.31.2
```

The previous system Python environment used `mlx-lm 0.29.1`, which did not include `mlx_lm.models.gemma4`. The new MLX environment includes the Gemma 4 backend files, but Gemma 4 E2B conversion may still fail due to model/weight compatibility or memory limits.

## Running Experiments

For quantization and sensitivity experiments, see:

```text
RUN_01_04_QUANTIZATION.md
```

For KV-cache and speculative decoding experiments, see:

```text
RUN_05_OPTIMIZATION.md
```

For ONNX export and validation, see:

```text
RUN_06_07_ONNX.md
```

For MLX Gemma 4 retry commands, see:

```text
RUN_08_MLX_0313.md
```

For throughput benchmarking, see:

```text
RUN_09_BENCHMARK.md
```

## Quick Local Smoke Tests

Baseline smoke test with a tiny model:

```bash
python3 scripts/01_baseline.py \
  --model-id sshleifer/tiny-gpt2 \
  --device cpu \
  --dtype fp32 \
  --max-new-tokens 8 \
  --warmup-runs 0 \
  --num-runs 1 \
  --output results/smoke_baseline_tiny_cpu.json
```

Toy ONNX smoke tests:

```bash
python3 scripts/06_1_toy_onnx_smoke.py
python3 scripts/06_2_toy_kv_cache_onnx_smoke.py
python3 scripts/07_1_validate_toy_onnx_smoke.py \
  --mode both \
  --provider cpu \
  --output results/toy_onnx_validation.json
```

These smoke tests verify local pipeline mechanics only. They do not replace the target Gemma 4 E2B CUDA experiments.

## Known Issues

The full Gemma 4 E2B CUDA/A100 experiments repeatedly hit exit code 137 during or near model loading. I tried mitigations such as pre-downloading model shards, direct CUDA/Accelerate loading, `low_cpu_mem_usage=True`, bf16 defaults, eager attention, memory logging, and memory-budget/offload options. The issue appears to be a system-level memory or job-limit kill rather than a normal Python exception.

Gemma 4 E2B with some torch/transformers combinations can hit an SDPA `enable_gqa` incompatibility. The scripts default to `attn_implementation="eager"` to avoid that path.

`02_quantize_int8.py`, `03_quantize_int4.py`, `04_sensitivity.py`, and INT8/INT4 rows in `09_benchmark.py` require NVIDIA CUDA because they use bitsandbytes. They are not expected to run on CPU or Apple MPS.

The Hugging Face ONNX export path failed locally for Gemma 4/tiny-gpt2 because the legacy PyTorch ONNX exporter could not trace the current Transformers causal-mask path. Toy ONNX no-cache and KV-cache graphs were added to validate the ONNX Runtime pipeline separately.

The original MLX conversion result in `results/edge_mlx_conversion.json` failed because `mlx-lm 0.29.1` did not include a Gemma 4 backend. A newer MLX environment with `mlx-lm 0.31.3` resolves that missing-backend issue, but `results/edge_mlx_conversion_0313_convert_only.json` shows a later failure while loading Gemma 4 E2B weights: MLX-LM received extra parameters such as later-layer `self_attn.k_proj`, `self_attn.v_proj`, and `self_attn.k_norm` weights that did not match the local model class. This means Gemma 4 conversion still needs further MLX-LM/model compatibility investigation.

## Generated Artifacts

Large generated artifacts are intentionally not committed:

```text
models/
*.onnx
*.pt
*.safetensors
.venv/
.venv-mlx/
```

Result JSON/CSV files under `results/` are lightweight experiment records. Some included results are smoke-test outputs rather than final Gemma 4 E2B benchmark results.
