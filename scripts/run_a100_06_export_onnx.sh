#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ID="${MODEL_ID:-google/gemma-4-E2B}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp16}"
OPSET="${OPSET:-17}"
PROMPT="${PROMPT:-Deep learning is}"
PAST_SEQ_LEN="${PAST_SEQ_LEN:-8}"
RUN_NO_CACHE="${RUN_NO_CACHE:-1}"
RUN_WITH_CACHE="${RUN_WITH_CACHE:-1}"
CONSTANT_FOLDING="${CONSTANT_FOLDING:-0}"
EXTERNAL_DATA="${EXTERNAL_DATA:-1}"
LOG_FILE="${LOG_FILE:-results/a100_06_export_onnx.log}"

mkdir -p results models
exec > >(tee -a "${LOG_FILE}") 2>&1

constant_folding_args=()
if [[ "${CONSTANT_FOLDING}" == "1" ]]; then
  constant_folding_args+=(--constant-folding)
fi

external_data_args=()
if [[ "${EXTERNAL_DATA}" == "0" ]]; then
  external_data_args+=(--disable-external-data)
fi

if [[ "${RUN_NO_CACHE}" == "1" ]]; then
  echo "== 06. ONNX export: no-cache =="
  "${PYTHON_BIN}" scripts/06_export_onnx.py \
    --model-id "${MODEL_ID}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --export-mode no_cache \
    --opset "${OPSET}" \
    --prompt "${PROMPT}" \
    --output models/gemma4_e2b_no_cache.onnx \
    --metadata-output results/onnx_export_no_cache.json \
    --reference-logits-output results/onnx_reference_logits_no_cache.pt \
    "${constant_folding_args[@]}" \
    "${external_data_args[@]}"
fi

if [[ "${RUN_WITH_CACHE}" == "1" ]]; then
  echo "== 06. ONNX export: with explicit KV-cache =="
  "${PYTHON_BIN}" scripts/06_export_onnx.py \
    --model-id "${MODEL_ID}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --export-mode with_cache \
    --past-seq-len "${PAST_SEQ_LEN}" \
    --opset "${OPSET}" \
    --prompt "${PROMPT}" \
    --output models/gemma4_e2b_with_cache.onnx \
    --metadata-output results/onnx_export_with_cache.json \
    --reference-logits-output results/onnx_reference_logits_with_cache.pt \
    "${constant_folding_args[@]}" \
    "${external_data_args[@]}"
fi

echo "A100 06 ONNX export commands completed."
