# 05 Optimization Benchmark Commands

This file lists the commands to run `scripts/05_optimize_compile.py` on a CUDA GPU server.

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

## 3. Quick Smoke Test

Runs only KV-cache on/off with short generation:

```bash
python3 scripts/05_optimize_compile.py \
  --model-id google/gemma-4-E2B \
  --device cuda \
  --dtype fp16 \
  --run-kv-cache \
  --max-new-tokens 16 \
  --num-runs 1 \
  --warmup-runs 0 \
  --output-dir results/optimize_smoke
```

Output:

```text
results/optimize_smoke/kv_cache_on_off.json
```

## 4. KV-Cache Benchmark

```bash
python3 scripts/05_optimize_compile.py \
  --model-id google/gemma-4-E2B \
  --device cuda \
  --dtype fp16 \
  --run-kv-cache \
  --max-new-tokens 64 \
  --num-runs 5 \
  --warmup-runs 1 \
  --output-dir results/optimize_kv_cache
```

Output:

```text
results/optimize_kv_cache/kv_cache_on_off.json
```

## 5. Speculative Decoding: All Assistant Candidates

This compares the three Gemma assistant candidates:

- `google/gemma-3-270m-it`
- `google/gemma-3-270m`
- `google/gemma-3-1b-it`

```bash
python3 scripts/05_optimize_compile.py \
  --model-id google/gemma-4-E2B \
  --device cuda \
  --dtype fp16 \
  --run-speculative \
  --max-new-tokens 64 \
  --num-runs 5 \
  --warmup-runs 1 \
  --output-dir results/optimize_speculative
```

Outputs:

```text
results/optimize_speculative/speculative__google__gemma-3-270m-it.json
results/optimize_speculative/speculative__google__gemma-3-270m.json
results/optimize_speculative/speculative__google__gemma-3-1b-it.json
```

Each file includes:

- target latency/TPS with `use_cache=False`
- target latency/TPS with `use_cache=True`
- speculative latency/TPS with `use_cache=False`
- speculative latency/TPS with `use_cache=True`
- CUDA peak memory
- assistant proposed/accepted/rejected draft token counts
- reject rate and accept rate
- tokenizer compatibility probe results

## 6. Full 8-Case Benchmark

Use this when you want to compare the full matrix:

| Case | KV cache | Speculative decoding | Assistant |
| --- | --- | --- | --- |
| 1 | off | off | none |
| 2 | on | off | none |
| 3 | off | on | `google/gemma-3-270m-it` |
| 4 | on | on | `google/gemma-3-270m-it` |
| 5 | off | on | `google/gemma-3-270m` |
| 6 | on | on | `google/gemma-3-270m` |
| 7 | off | on | `google/gemma-3-1b-it` |
| 8 | on | on | `google/gemma-3-1b-it` |

```bash
python3 scripts/05_optimize_compile.py \
  --model-id google/gemma-4-E2B \
  --device cuda \
  --dtype fp16 \
  --run-kv-cache \
  --run-speculative \
  --max-new-tokens 64 \
  --num-runs 5 \
  --warmup-runs 1 \
  --output-dir results/optimize_full_8_cases
```

Outputs:

```text
results/optimize_full_8_cases/kv_cache_on_off.json
results/optimize_full_8_cases/speculative__google__gemma-3-270m-it.json
results/optimize_full_8_cases/speculative__google__gemma-3-270m.json
results/optimize_full_8_cases/speculative__google__gemma-3-1b-it.json
```

`kv_cache_on_off.json` contains cases 1 and 2. Each speculative result file contains the two speculative cases for that assistant, so the three assistant files contain cases 3 through 8.

## 7. Speculative Decoding: One Assistant

Use this when you want to test only one candidate first:

```bash
python3 scripts/05_optimize_compile.py \
  --model-id google/gemma-4-E2B \
  --assistant-models google/gemma-3-270m-it \
  --device cuda \
  --dtype fp16 \
  --run-speculative \
  --max-new-tokens 64 \
  --num-runs 5 \
  --warmup-runs 1 \
  --output-dir results/optimize_speculative_270m_it
```

## 8. Larger GPU With BF16

On A100/H100/L4-class GPUs, you can try:

```bash
python3 scripts/05_optimize_compile.py \
  --model-id google/gemma-4-E2B \
  --device cuda \
  --dtype bf16 \
  --run-kv-cache \
  --run-speculative \
  --max-new-tokens 128 \
  --num-runs 5 \
  --warmup-runs 1 \
  --output-dir results/optimize_full_bf16
```

## Notes

- The script uses greedy decoding: `do_sample=False`.
- Speculative decoding results are assistant-specific and are written to separate JSON files.
- Reject rate is captured by instrumenting Transformers' `AssistedCandidateGenerator` and counting proposed draft tokens that were not accepted by the target model.
- If a tokenizer compatibility issue appears, inspect the `tokenizer_compatibility` block in each speculative result JSON.
