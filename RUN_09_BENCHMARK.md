# 09 Throughput Benchmark Commands

This file lists commands for `scripts/09_benchmark.py`, the Challenge 3
benchmark harness.

The script writes the required CSV-style metrics:

```text
model, precision, batch_size, input_seq_len, gen_len,
ttft_ms_mean, ttft_ms_std, tps_mean, tps_std,
peak_mem_mb, hardware, notes
```

Additional raw run values are also included for auditability. The CSV also
records `prompt_mode`, `prompt_variants`, and `measurement_count`.

## 1. Authenticate Hugging Face

Gemma models are gated. Log in with the Hugging Face account that has access to
the Gemma model page:

```bash
hf auth login
hf auth whoami
```

Or set a token from the access-approved account:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

## 2. Install Requirements

```bash
python3 -m pip install -r requirements.txt
```

For INT8/INT4 rows, use an NVIDIA CUDA GPU because bitsandbytes quantization
requires CUDA.

## 3. Default 12-Configuration Benchmark

The default configuration runs 12 representative rows:

- precision: `fp16`, `int8`, `int4`
- input sequence length: `64`, `512`
- generation length: `32`, `128`
- batch size: `1`

```bash
python3 scripts/09_benchmark.py \
  --model-id google/gemma-4-E2B \
  --device cuda \
  --warmup-runs 1 \
  --num-runs 5 \
  --output results/benchmark.csv
```

Output:

```text
results/benchmark.csv
```

## 4. Larger Batch Benchmark

Use this if the GPU has enough memory and you want batch-size coverage:

```bash
python3 scripts/09_benchmark.py \
  --model-id google/gemma-4-E2B \
  --device cuda \
  --warmup-runs 1 \
  --num-runs 5 \
  --configs fp16:1:64:32,fp16:2:64:32,fp16:1:512:128,fp16:2:512:128,int8:1:64:32,int8:2:64:32,int8:1:512:128,int8:2:512:128,int4:1:64:32,int4:2:64:32,int4:1:512:128,int4:2:512:128 \
  --output results/benchmark_batch.csv
```

## 5. Diverse Prompt Benchmark

The default benchmark uses one synthetic technical seed repeated/truncated to
the requested `input_seq_len`. To include multiple ordinary prompt styles while
still controlling token length, use `--prompt-mode diverse`.

With `--prompt-variants 4`, each config runs four different prompt seeds. With
`--num-runs 5`, each CSV row aggregates:

```text
4 prompt variants * 5 runs = 20 measurements
```

This increases runtime proportionally.

```bash
python3 scripts/09_benchmark.py \
  --model-id google/gemma-4-E2B \
  --device cuda \
  --warmup-runs 1 \
  --num-runs 5 \
  --prompt-mode diverse \
  --prompt-variants 4 \
  --output results/benchmark_diverse.csv
```

Output:

```text
results/benchmark_diverse.csv
```

Use this as a robustness check for latency/TPS trends, not as an answer-quality
evaluation. The prompts are still repeated/truncated to fixed token lengths.

## 6. Quick Smoke Test

Use a tiny model locally or a short Gemma run on the server:

```bash
python3 scripts/09_benchmark.py \
  --model-id sshleifer/tiny-gpt2 \
  --device cpu \
  --warmup-runs 0 \
  --num-runs 1 \
  --configs fp32:1:16:4 \
  --output results/benchmark_smoke.csv
```

## Notes

- `gen_len` is interpreted as the total requested generated-token count,
  including the first generated token.
- TTFT is measured from the prompt prefill start through first-token selection.
- TPS is measured during the remaining manual greedy decode loop using the
  returned KV-cache, excluding the first token already counted by TTFT.
- `--prompt-mode synthetic` uses one technical seed text.
- `--prompt-mode diverse` uses several ordinary prompt seeds, each repeated or
  truncated to the requested `input_seq_len` so length remains controlled.
- CUDA timing uses `torch.cuda.synchronize()` before and after timed regions.
- INT8/INT4 configs are skipped as failed rows if run without CUDA.
- The default benchmark is intentionally separate from the short smoke prompts
  used in scripts `01` through `08`.
