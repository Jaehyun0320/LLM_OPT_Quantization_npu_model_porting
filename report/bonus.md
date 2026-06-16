# Bonus: Practical Failure Analysis and Reproducibility Smoke Tests

## Motivation
The assignment bonus section allowed any extension that shows how I think about ML systems. Since the full Gemma 4 E2B server experiments were repeatedly blocked by environment-level failures, I treated the failure modes themselves as an additional systems-analysis task rather than ignoring them.

## What I Added
I organized the project around failure isolation and reproducibility checks. Instead of only reporting that a run failed, I separated failures into model-loading, quantization-backend, ONNX-exporter, ONNX-runtime, tokenizer-compatibility, and MLX-conversion categories. This made it clearer which parts of the pipeline were broken and which parts were independently validated.

I added controlled ONNX smoke tests with toy causal-LM models. The no-cache toy model verified that a simple causal language model can be exported and validated with ONNX Runtime. The KV-cache toy model verified explicit `past_key/value` input and `present_key/value` output wiring, including a recurrent ORT loop where cache length grows by one token per step. A separate `07_1` validator then reloaded the exported ONNX files and validated them independently from the export step.

I also documented runtime compatibility boundaries. CUDA/bitsandbytes experiments require NVIDIA GPUs; local CPU/MPS runs are only smoke tests; Gemma 4 ONNX export is blocked by the Transformers causal-mask tracing path; direct MLX conversion is blocked by local model/checkpoint compatibility; and preconverted MLX community checkpoints may be a practical runtime path.

## Why This Is Useful
In real deployment work, failures are often caused by runtime boundaries rather than the model architecture alone. A reproducibility matrix and smoke-test suite help distinguish "the model cannot work" from "this exporter/runtime/version path cannot handle this model yet." This is useful for deciding the next engineering step: change hardware, change runtime, use a preconverted checkpoint, reduce scope to a smoke test, or build a model-specific export path.

## Original Idea / Reflection
If I had been able to run the full server experiments successfully and analyze the resulting metrics, I would have liked to explore either `A creative visualization of quantization error or layer sensitivity` or `A side-by-side comparison of Gemma 4 E2B vs. another open LLM on the same benchmarks`.

I tend to look for underlying reasons when I study, debug, or even try to understand people. When the question "why?" remains unresolved, I often feel that the problem is still unfinished. Through this assignment, I realized that although hands-on implementation is necessary, I am more naturally drawn to the theory-driven reasoning around it: analyzing why something failed, building high-level intuition about possible solutions, and thinking about how results could be improved, even when the full target experiment could not be completed.
