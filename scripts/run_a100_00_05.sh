#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ID="${MODEL_ID:-google/gemma-4-E2B}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp16}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
NUM_RUNS="${NUM_RUNS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
LOG_FILE="${LOG_FILE:-results/a100_00_05.log}"

mkdir -p results models
exec > >(tee -a "${LOG_FILE}") 2>&1

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

echo "A100 00-05 experiments completed."
