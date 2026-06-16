# ONNX / Edge Format Report

## Summary
- Hugging Face ONNX export failed because the local torch/transformers legacy exporter could not trace the Transformers causal-mask path.
- Toy ONNX tests verified export, ORT validation, and explicit KV-cache I/O on exportable graphs.
- Apple MLX direct conversion was tried with `mlx-lm 0.29.1` and `0.31.3`, but Gemma 4 E2B conversion still did not complete.

## ONNX Export And Validation
Gemma 4 E2B and tiny-gpt2 ONNX export failed while tracing the Transformers causal-mask implementation. The failure occurred in the `masking_utils` / `vmap` / functorch path, not during model loading or because of model size. Removing `attention_mask` did not help, because export then failed in the generic causal-mask path. This suggests an exporter compatibility issue rather than an ONNX Runtime issue.

To separate exporter compatibility from runtime correctness, I added a toy causal LM with explicit masking and ONNX-friendly operations. It exported successfully, and ORT matched PyTorch logits with max error `3.58e-7`, below the `1e-5` tolerance (`results/toy_onnx_smoke.json`).

I also added a toy KV-cache graph with `past_key/value` inputs and `present_key/value` outputs. ORT matched PyTorch logits and cache tensors within `1e-5`, and a four-step loop confirmed cache length grew by one token per step (`results/toy_kv_cache_onnx_smoke.json`). `scripts/07_1_validate_toy_onnx_smoke.py` separately reloads the toy ONNX files and validates no-cache and KV-cache paths (`results/toy_onnx_validation.json`).

## Apple MLX Edge Format
I first attempted to convert `google/gemma-4-E2B` with `mlx-lm 0.29.1`. Download succeeded, but conversion failed because that package lacked `mlx_lm.models.gemma4`. I then created a Python 3.12 MLX environment with `mlx-lm 0.31.3` and `mlx 0.31.2`, which fixed the missing backend.

The `0.31.3` retry still failed while loading weights: MLX-LM reported 60 checkpoint parameters missing from its model class, mostly later-layer `self_attn.k_proj`, `self_attn.v_proj`, and `self_attn.k_norm` weights. This indicates a model/checkpoint key-mapping issue, not memory failure. A Gemma 3 270M MLX smoke test succeeded, so the local MLX toolchain works.

As a workaround, `mlx-community/gemma-4-e2b-it-4bit` may be a better runtime path. It is a preconverted 4-bit MLX model from `google/gemma-4-e2b-it` using `mlx-vlm`, so it can bypass my local `mlx-lm.convert` weight-mapping failure. This is a community-provided conversion, not proof that my conversion script succeeded.

## Limitations
The toy ONNX tests validate the ONNX/ORT pipeline and explicit KV-cache wiring, but not Gemma 4 E2B-specific rotary position handling, grouped-query attention, or Transformers cache utilities. Production export needs a newer exporter/runtime or a model-specific static-cache path.
