# 06-07 ONNX Export and Validation Commands

This file lists the CUDA GPU server commands for:

- `scripts/06_export_onnx.py`: export a Gemma model forward step to ONNX
- `scripts/07_validate_onnx.py`: validate ONNX Runtime outputs against PyTorch reference logits

Run `06_export_onnx.py` first. `07_validate_onnx.py` consumes the `.onnx`,
metadata JSON, and reference logits produced by step 06.

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
access to the gated Gemma model pages. If you use `HF_TOKEN`, it must come from
that access-approved account.

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

## 2. Install Requirements

```bash
python3 -m pip install -r requirements.txt
```

For CUDA ONNX Runtime validation, the server may need `onnxruntime-gpu` rather
than CPU-only `onnxruntime`:

```bash
python3 -m pip install onnxruntime-gpu
```

Check available ONNX Runtime providers:

```bash
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

If `CUDAExecutionProvider` appears, use `--provider cuda` in validation. If not,
use `--provider cpu`.

## 3. No-Cache ONNX Export

This exports a simple forward graph:

```text
input_ids, attention_mask -> logits
```

```bash
python3 scripts/06_export_onnx.py \
  --model-id google/gemma-4-E2B \
  --device cuda \
  --dtype fp16 \
  --export-mode no_cache \
  --opset 17 \
  --prompt "Deep learning is" \
  --output models/gemma4_e2b_no_cache.onnx \
  --metadata-output results/onnx_export_no_cache.json \
  --reference-logits-output results/onnx_reference_logits_no_cache.pt
```

Outputs:

```text
models/gemma4_e2b_no_cache.onnx
results/onnx_export_no_cache.json
results/onnx_reference_logits_no_cache.pt
```

The metadata JSON records input/output names, dynamic axes, export mode, dtype,
device, and the saved PyTorch reference logits path.

## 4. No-Cache ONNX Validation

Use CUDA provider if available:

```bash
python3 scripts/07_validate_onnx.py \
  --metadata results/onnx_export_no_cache.json \
  --provider cuda \
  --tolerance 1e-3 \
  --output results/onnx_validation_no_cache.json
```

If CUDA provider is unavailable:

```bash
python3 scripts/07_validate_onnx.py \
  --metadata results/onnx_export_no_cache.json \
  --provider cpu \
  --tolerance 1e-3 \
  --output results/onnx_validation_no_cache.json
```

Output:

```text
results/onnx_validation_no_cache.json
```

The validation JSON reports ONNX Runtime providers, input/output names,
last-token logits difference against the PyTorch reference, and pass/fail.

## 5. With-Cache ONNX Export

This attempts to export a one-step autoregressive graph with explicit KV-cache
inputs and outputs:

```text
input_ids, attention_mask, position_ids, past_key_i, past_value_i
  -> logits, present_key_i, present_value_i
```

`--past-seq-len` controls the dummy cache length used during export. For
example, `--past-seq-len 8` exports a step where the input cache has sequence
length 8 and the output cache should have sequence length 9.

```bash
python3 scripts/06_export_onnx.py \
  --model-id google/gemma-4-E2B \
  --device cuda \
  --dtype fp16 \
  --export-mode with_cache \
  --past-seq-len 8 \
  --opset 17 \
  --prompt "Deep learning is" \
  --output models/gemma4_e2b_with_cache.onnx \
  --metadata-output results/onnx_export_with_cache.json \
  --reference-logits-output results/onnx_reference_logits_with_cache.pt
```

Expected outputs if export succeeds:

```text
models/gemma4_e2b_with_cache.onnx
results/onnx_export_with_cache.json
results/onnx_reference_logits_with_cache.pt
```

If export fails, `results/onnx_export_with_cache.json` is still written with
the error type, error message, traceback, and notes for the report.

## 6. With-Cache ONNX Validation

Use CUDA provider if available:

```bash
python3 scripts/07_validate_onnx.py \
  --metadata results/onnx_export_with_cache.json \
  --provider cuda \
  --tolerance 1e-3 \
  --loop-steps 4 \
  --output results/onnx_validation_with_cache.json
```

If CUDA provider is unavailable:

```bash
python3 scripts/07_validate_onnx.py \
  --metadata results/onnx_export_with_cache.json \
  --provider cpu \
  --tolerance 1e-3 \
  --loop-steps 4 \
  --output results/onnx_validation_with_cache.json
```

Output:

```text
results/onnx_validation_with_cache.json
```

The validation JSON reports:

- last-token logits difference against PyTorch reference
- whether each `present_key_i` / `present_value_i` grew from `past_seq_len` to
  `past_seq_len + 1`
- optional mechanical greedy loop results from `--loop-steps`

The loop check uses dummy zero initial cache, so it verifies ONNX recurrent I/O
wiring rather than meaningful text quality.

## 7. Recommended Minimum Run

For a practical first pass, run these two commands:

```bash
python3 scripts/06_export_onnx.py \
  --model-id google/gemma-4-E2B \
  --device cuda \
  --dtype fp16 \
  --export-mode no_cache \
  --output models/gemma4_e2b_no_cache.onnx \
  --metadata-output results/onnx_export_no_cache.json \
  --reference-logits-output results/onnx_reference_logits_no_cache.pt
```

```bash
python3 scripts/07_validate_onnx.py \
  --metadata results/onnx_export_no_cache.json \
  --provider cuda \
  --tolerance 1e-3 \
  --output results/onnx_validation_no_cache.json
```

Then try the with-cache export. Full KV-cache export is more fragile because
the model's cache tensors must be exposed as explicit ONNX inputs and outputs.

## Notes

- `06_export_onnx.py` exports a forward step, not the full `generate()` loop.
- `07_validate_onnx.py` runs ONNX Runtime and compares against the PyTorch
  reference logits saved by step 06.
- For generation with an ONNX forward-step model, the runtime must manage the
  decode loop, next-token selection, `present -> past` cache handoff,
  `attention_mask`, and `position_ids`.
- Use the no-cache result as the baseline export artifact. Use the with-cache
  result or failure metadata as evidence for the KV-cache handling attempt.
