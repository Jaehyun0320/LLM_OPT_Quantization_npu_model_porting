#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ID="${MODEL_ID:-google/gemma-4-E2B}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp16}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
NUM_RUNS="${NUM_RUNS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

mkdir -p results models

echo "== 00. Environment check =="
"${PYTHON_BIN}" scripts/00_check_env.py \
  --check-model-access \
  --output results/env.json

echo "== 01. Baseline fp16 =="
"${PYTHON_BIN}" scripts/01_baseline.py \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --num-runs "${NUM_RUNS}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --output results/baseline_gemma4_e2b.json

echo "== 02. INT8 quantization =="
"${PYTHON_BIN}" scripts/02_quantize_int8.py \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --num-runs "${NUM_RUNS}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --output results/quant_int8_gemma4_e2b.json

echo "== 03. INT4 NF4 quantization =="
"${PYTHON_BIN}" scripts/03_quantize_int4.py \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --num-runs "${NUM_RUNS}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --output results/quant_int4_gemma4_e2b.json

echo "== 04. Sensitivity analysis: INT8, all predefined component experiments =="
"${PYTHON_BIN}" scripts/04_sensitivity.py \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --quantization int8 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --num-runs "${NUM_RUNS}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --output results/sensitivity_int8_gemma4_e2b.csv

echo "== 04. Sensitivity analysis: INT4, all predefined component experiments =="
"${PYTHON_BIN}" scripts/04_sensitivity.py \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --quantization int4 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --num-runs "${NUM_RUNS}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --output results/sensitivity_int4_gemma4_e2b.csv

echo "== 05. KV-cache and speculative decoding full 8-case benchmark =="
"${PYTHON_BIN}" scripts/05_optimize_compile.py \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --run-kv-cache \
  --run-speculative \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --num-runs "${NUM_RUNS}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --output-dir results/optimize_full_8_cases

echo "== 06. ONNX export: no-cache =="
"${PYTHON_BIN}" scripts/06_export_onnx.py \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --export-mode no_cache \
  --opset 17 \
  --prompt "Deep learning is" \
  --output models/gemma4_e2b_no_cache.onnx \
  --metadata-output results/onnx_export_no_cache.json \
  --reference-logits-output results/onnx_reference_logits_no_cache.pt

echo "== 07. ONNX validation: no-cache =="
"${PYTHON_BIN}" scripts/07_validate_onnx.py \
  --metadata results/onnx_export_no_cache.json \
  --provider cuda \
  --tolerance 1e-3 \
  --output results/onnx_validation_no_cache.json

echo "== 06. ONNX export: with explicit KV-cache =="
"${PYTHON_BIN}" scripts/06_export_onnx.py \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --export-mode with_cache \
  --past-seq-len 8 \
  --opset 17 \
  --prompt "Deep learning is" \
  --output models/gemma4_e2b_with_cache.onnx \
  --metadata-output results/onnx_export_with_cache.json \
  --reference-logits-output results/onnx_reference_logits_with_cache.pt

if "${PYTHON_BIN}" -c "import json; import sys; sys.exit(0 if json.load(open('results/onnx_export_with_cache.json')).get('success') else 1)"; then
  echo "== 07. ONNX validation: with explicit KV-cache =="
  "${PYTHON_BIN}" scripts/07_validate_onnx.py \
    --metadata results/onnx_export_with_cache.json \
    --provider cuda \
    --tolerance 1e-3 \
    --loop-steps 4 \
    --output results/onnx_validation_with_cache.json
else
  echo "Skipping with-cache ONNX validation because export did not succeed."
  echo "See results/onnx_export_with_cache.json for failure details."
fi

echo "== 08. Edge conversion =="
echo "Skipping Apple MLX conversion on CUDA server."
echo "Use scripts/08_convert_edge.py and scripts/08_1_convert_edge_smoke.py on Apple Silicon."

echo "== 09. Throughput benchmark: default 12 configs with diverse prompts =="
"${PYTHON_BIN}" scripts/09_benchmark.py \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --num-runs "${NUM_RUNS}" \
  --prompt-mode diverse \
  --prompt-variants 4 \
  --output results/benchmark_diverse.csv

echo "All requested A100 CUDA experiment commands completed."
