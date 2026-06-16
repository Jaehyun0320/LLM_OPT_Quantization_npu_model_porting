# Notice

## Server Execution Limitation

The full Gemma 4 E2B experiments were repeatedly attempted on a CUDA/A100 server, but the process was terminated with exit code 137 during or near model weight loading. I treated this as a system-level memory or job-limit issue rather than a normal Python exception.

## Mitigation Attempts

I tried several mitigations to reduce model-loading peaks and improve diagnosability:

- Pre-downloaded Hugging Face model shards into the disk cache.
- Switched from CPU-then-`model.to(cuda)` loading to direct CUDA/Accelerate loading.
- Enabled `low_cpu_mem_usage=True`.
- Added CUDA synchronization and memory checkpoints.
- Forced the safer `eager` attention backend to avoid the Gemma 4 `enable_gqa` SDPA incompatibility.
- Changed the default A100 dtype from fp16 to bf16.
- Added `device_map="auto"` with GPU/CPU memory budgets and offload support.

These changes reduced avoidable loading peaks and made failures easier to inspect, but they did not fully prevent the server-side kill. The remaining failure is likely related to runtime/job cgroup limits or a model-loading peak outside the Python-level control path.

## Validation Scope

Because the full Gemma 4 E2B server runs could not be completed reliably, the submitted results focus on smoke tests and pipeline validation rather than final Gemma 4 E2B performance conclusions.

The completed validation includes:

- Local inference baseline smoke tests with tiny models.
- ONNX export and ONNX Runtime smoke tests using toy no-cache and KV-cache models.
- Apple MLX conversion/generation smoke testing with a supported Gemma 3 model.
- Documentation of the Gemma 4 E2B MLX conversion attempts and failure modes.
- Benchmark, quantization, and optimization scripts prepared for the intended CUDA/A100 environment.

Therefore, the reports describe the implemented methodology, attempted mitigations, smoke-test evidence, and known blockers. The final quantitative Gemma 4 E2B latency, memory, and quality analysis should be rerun in a CUDA environment where the target model can be loaded reliably.
