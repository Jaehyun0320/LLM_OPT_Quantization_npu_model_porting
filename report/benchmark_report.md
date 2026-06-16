# Benchmark Report

## Summary
- `scripts/09_benchmark.py` was prepared as the Challenge 3 throughput benchmark harness.
- The intended criteria were TTFT, decode TPS, peak memory, precision mode, prompt length, generation length, batch size, and prompt diversity.
- The benchmark was designed to compare fp16/bf16, INT8, and INT4 on Gemma 4 E2B.
- The full server benchmark could not be completed because CUDA/A100 runs were repeatedly killed with exit code 137 during or near model loading.

## Benchmark Criteria
The script writes CSV rows with model name, precision, batch size, input sequence length, generation length, prompt mode, prompt variants, measurement count, TTFT mean/std, TPS mean/std, peak memory, hardware, and notes. This format was intended to make quantization and sequence-length tradeoffs easy to compare.

TTFT is measured from prompt prefill start through first-token selection, which approximates user-visible response delay. Decode TPS is measured after the first token using a manual greedy decode loop with `past_key_values`, so it focuses on steady-state autoregressive generation with KV-cache enabled. CUDA peak memory is recorded when running on GPU. INT8 and INT4 rows require NVIDIA CUDA because they use bitsandbytes.

## Planned Experiment Matrix
The default matrix contains 12 configurations: three precision modes (`fp16`, `int8`, `int4`), two input lengths (`64`, `512`), two generation lengths (`32`, `128`), and batch size `1`. I also added optional larger-batch configs and a diverse-prompt mode.

The diverse-prompt mode uses ordinary prompt seeds such as explanation, email, summarization, comparison, deployment-risk, and transformer-attention prompts. Each seed is repeated or truncated to the requested token length so latency remains length-controlled.

## Server Execution Issue
The full Gemma 4 E2B benchmark was attempted on a CUDA/A100 server, but the process repeatedly terminated with exit code 137. This appeared to be a system-level memory or job-limit kill, not a normal Python exception. Because the process died during or near model loading, I could not produce reliable fp16, INT8, or INT4 Gemma 4 E2B benchmark results.

Mitigations included pre-downloading model shards, direct CUDA/Accelerate loading, `low_cpu_mem_usage=True`, bf16 defaults, `eager` attention, memory logging, and memory-budget/offload options. These improved diagnosability but did not avoid the server-side kill.

## Limitations
This report documents the benchmark design and attempted criteria rather than final performance conclusions. The next step is to rerun `scripts/09_benchmark.py` in a CUDA environment where Gemma 4 E2B can be loaded reliably, then compare TTFT, TPS, and peak memory across fp16/bf16, INT8, and INT4 rows.
