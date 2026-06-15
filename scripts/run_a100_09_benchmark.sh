#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ID="${MODEL_ID:-google/gemma-4-E2B}"
DEVICE="${DEVICE:-cuda}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
NUM_RUNS="${NUM_RUNS:-5}"
PROMPT_VARIANTS="${PROMPT_VARIANTS:-4}"
RUN_SYNTHETIC="${RUN_SYNTHETIC:-1}"
RUN_DIVERSE="${RUN_DIVERSE:-1}"
RUN_BATCH="${RUN_BATCH:-1}"
BATCH_BENCHMARK_CONFIGS="${BATCH_BENCHMARK_CONFIGS:-fp16:1:64:32,fp16:2:64:32,fp16:1:512:128,fp16:2:512:128,int8:1:64:32,int8:2:64:32,int8:1:512:128,int8:2:512:128,int4:1:64:32,int4:2:64:32,int4:1:512:128,int4:2:512:128}"
LOG_FILE="${LOG_FILE:-results/a100_09_benchmark.log}"

mkdir -p results
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ "${RUN_SYNTHETIC}" == "1" ]]; then
  echo "== 09A. Throughput benchmark: controlled synthetic prompts, default 12 configs =="
  "${PYTHON_BIN}" scripts/09_benchmark.py \
    --model-id "${MODEL_ID}" \
    --device "${DEVICE}" \
    --warmup-runs "${WARMUP_RUNS}" \
    --num-runs "${NUM_RUNS}" \
    --prompt-mode synthetic \
    --prompt-variants 1 \
    --output results/benchmark_synthetic.csv
fi

if [[ "${RUN_DIVERSE}" == "1" ]]; then
  echo "== 09B. Throughput benchmark: diverse prompts, default 12 configs =="
  "${PYTHON_BIN}" scripts/09_benchmark.py \
    --model-id "${MODEL_ID}" \
    --device "${DEVICE}" \
    --warmup-runs "${WARMUP_RUNS}" \
    --num-runs "${NUM_RUNS}" \
    --prompt-mode diverse \
    --prompt-variants "${PROMPT_VARIANTS}" \
    --output results/benchmark_diverse.csv
fi

if [[ "${RUN_BATCH}" == "1" ]]; then
  echo "== 09C. Throughput benchmark: batch-size coverage configs =="
  "${PYTHON_BIN}" scripts/09_benchmark.py \
    --model-id "${MODEL_ID}" \
    --device "${DEVICE}" \
    --warmup-runs "${WARMUP_RUNS}" \
    --num-runs "${NUM_RUNS}" \
    --prompt-mode synthetic \
    --prompt-variants 1 \
    --configs "${BATCH_BENCHMARK_CONFIGS}" \
    --output results/benchmark_batch.csv
fi

echo "A100 09 benchmark commands completed."
