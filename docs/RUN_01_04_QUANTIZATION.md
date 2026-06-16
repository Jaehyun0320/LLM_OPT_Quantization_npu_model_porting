# 01-04 Quantization Benchmark Commands

This file lists the commands to run `scripts/01_baseline.py` through
`scripts/04_sensitivity.py` on a CUDA GPU server with `google/gemma-4-E2B`.

## 1. Authenticate Hugging Face

Gemma models are gated. Log in on the GPU server before running:

```bash
hf auth login
```

If `hf` is unavailable:

```bash
huggingface-cli login
```

Check login:

```bash
hf auth whoami
```

The logged-in Hugging Face account must be the same account that was granted
access to the gated Gemma model pages. A random Hugging Face token will not
work unless that token belongs to an account with Gemma access.

If you use an environment variable instead of interactive login:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

The token must still be created from the access-approved Hugging Face account.
If the wrong account or token is used, model download usually fails with a
401/403 error or a gated-repo access message.

## 2. Install Requirements

```bash
python3 -m pip install -r requirements.txt
```

## 3. Optional Environment Check

```bash
python3 scripts/00_check_env.py \
  --check-model-access \
  --output results/env.json
```

Output:

```text
results/env.json
```

## 4. Baseline FP16/BF16 Run

Use this as the unquantized reference. On most CUDA GPUs, `--dtype auto`
selects BF16 if supported, otherwise FP16.

```bash
python3 scripts/01_baseline.py \
  --model-id google/gemma-4-E2B \
  --device cuda \
  --dtype auto \
  --max-new-tokens 64 \
  --num-runs 5 \
  --warmup-runs 1 \
  --output results/baseline_gemma4_e2b.json
```

Output:

```text
results/baseline_gemma4_e2b.json
```

The result includes model parameter storage size, latency mean/std, TPS
mean/std, CUDA peak memory, perplexity, and sample generations.

## 5. INT8 Quantization Run

```bash
python3 scripts/02_quantize_int8.py \
  --model-id google/gemma-4-E2B \
  --device cuda \
  --dtype auto \
  --max-new-tokens 64 \
  --num-runs 5 \
  --warmup-runs 1 \
  --output results/quant_int8_gemma4_e2b.json
```

Output:

```text
results/quant_int8_gemma4_e2b.json
```

The result includes approximate quantized parameter storage size, latency
mean/std, TPS mean/std, CUDA peak memory, perplexity, and sample generations.

## 6. INT4 NF4 Quantization Run

```bash
python3 scripts/03_quantize_int4.py \
  --model-id google/gemma-4-E2B \
  --device cuda \
  --dtype auto \
  --max-new-tokens 64 \
  --num-runs 5 \
  --warmup-runs 1 \
  --output results/quant_int4_gemma4_e2b.json
```

Output:

```text
results/quant_int4_gemma4_e2b.json
```

The result uses bitsandbytes 4-bit NF4 with double quantization.

## 7. Sensitivity Analysis: All Skip Experiments in One CSV

Yes: `04_sensitivity.py` can run all predefined skip-component experiments
for one quantization mode with a single command. By default, `--experiments`
already includes:

- `full_quant`
- `keep_attention_fp`
- `keep_mlp_fp`
- `keep_lm_head_fp`
- `keep_embeddings_fp`

Run all INT8 sensitivity rows into one CSV:

```bash
python3 scripts/04_sensitivity.py \
  --model-id google/gemma-4-E2B \
  --device cuda \
  --dtype auto \
  --quantization int8 \
  --max-new-tokens 64 \
  --num-runs 5 \
  --warmup-runs 1 \
  --output results/sensitivity_int8_gemma4_e2b.csv
```

Output:

```text
results/sensitivity_int8_gemma4_e2b.csv
```

Run all INT4 sensitivity rows into one CSV:

```bash
python3 scripts/04_sensitivity.py \
  --model-id google/gemma-4-E2B \
  --device cuda \
  --dtype auto \
  --quantization int4 \
  --max-new-tokens 64 \
  --num-runs 5 \
  --warmup-runs 1 \
  --output results/sensitivity_int4_gemma4_e2b.csv
```

Output:

```text
results/sensitivity_int4_gemma4_e2b.csv
```

Each CSV row records the experiment name, skipped modules, perplexity, latency
mean/std, TPS mean/std, CUDA peak memory, and quantized module counts.

## 8. Run a Subset of Sensitivity Experiments

Use `--experiments` when you only want selected rows:

```bash
python3 scripts/04_sensitivity.py \
  --model-id google/gemma-4-E2B \
  --device cuda \
  --dtype auto \
  --quantization int8 \
  --experiments full_quant,keep_attention_fp,keep_mlp_fp \
  --max-new-tokens 64 \
  --num-runs 5 \
  --warmup-runs 1 \
  --output results/sensitivity_int8_subset.csv
```

## Notes

- `02_quantize_int8.py`, `03_quantize_int4.py`, and `04_sensitivity.py`
  require an NVIDIA CUDA GPU because they use bitsandbytes quantization.
- `--dtype auto` selects BF16 on CUDA GPUs that support it, otherwise FP16.
- The scripts use greedy decoding with `do_sample=False`.
- CUDA timing uses `torch.cuda.synchronize()` before and after timed
  generation so latency/TPS measurements wait for GPU work to finish.
- `04_sensitivity.py` writes all selected skip experiments for one
  quantization mode into one CSV. To compare INT8 and INT4 sensitivity, run
  the two commands above and compare the two CSV files.
