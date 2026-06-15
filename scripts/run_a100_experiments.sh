#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ID="${MODEL_ID:-google/gemma-4-E2B}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
NUM_RUNS="${NUM_RUNS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
PROMPT_VARIANTS="${PROMPT_VARIANTS:-4}"
PRIME_CACHE="${PRIME_CACHE:-1}"
LOG_MEMORY="${LOG_MEMORY:-1}"
LOAD_DEVICE_MAP="${LOAD_DEVICE_MAP:-auto}"
MAX_GPU_MEMORY="${MAX_GPU_MEMORY:-70GiB}"
MAX_CPU_MEMORY="${MAX_CPU_MEMORY:-80GiB}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-models/offload/baseline}"
ASSISTANT_MODELS="${ASSISTANT_MODELS:-google/gemma-3-270m-it,google/gemma-3-270m,google/gemma-3-1b-it}"
BATCH_BENCHMARK_CONFIGS="${BATCH_BENCHMARK_CONFIGS:-fp16:1:64:32,fp16:2:64:32,fp16:1:512:128,fp16:2:512:128,int8:1:64:32,int8:2:64:32,int8:1:512:128,int8:2:512:128,int4:1:64:32,int4:2:64:32,int4:1:512:128,int4:2:512:128}"

mkdir -p results models

if [[ "${PRIME_CACHE}" == "1" ]]; then
  echo "== 00. Prime Hugging Face model cache =="
  "${PYTHON_BIN}" scripts/00_prime_model_cache.py \
    --model-ids "${MODEL_ID},${ASSISTANT_MODELS}" \
    --output results/cache_prime_all.json
fi

baseline_memory_args=()
if [[ "${LOG_MEMORY}" == "1" ]]; then
  baseline_memory_args+=(--log-memory)
fi

echo "== 00. Environment check =="
"${PYTHON_BIN}" scripts/00_check_env.py \
  --check-model-access \
  --output results/env.json

echo "== 01. Baseline ${DTYPE} =="
"${PYTHON_BIN}" scripts/01_baseline.py \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --load-device-map "${LOAD_DEVICE_MAP}" \
  --max-gpu-memory "${MAX_GPU_MEMORY}" \
  --max-cpu-memory "${MAX_CPU_MEMORY}" \
  --offload-folder "${OFFLOAD_FOLDER}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --num-runs "${NUM_RUNS}" \
  --warmup-runs "${WARMUP_RUNS}" \
  "${baseline_memory_args[@]}" \
  --output results/baseline_gemma4_e2b.json

echo "== 02. INT8 quantization =="
"${PYTHON_BIN}" scripts/02_quantize_int8.py \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --num-runs "${NUM_RUNS}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --output results/quant_int8_gemma4_e2b.json

echo "== 03. INT4 NF4 quantization =="
"${PYTHON_BIN}" scripts/03_quantize_int4.py \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --num-runs "${NUM_RUNS}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --output results/quant_int4_gemma4_e2b.json

echo "== 04. Sensitivity analysis: INT8, all predefined component experiments =="
"${PYTHON_BIN}" scripts/04_sensitivity.py \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
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
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
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
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --assistant-models "${ASSISTANT_MODELS}" \
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
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
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
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
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

echo "== 09A. Throughput benchmark: controlled synthetic prompts, default 12 configs =="
"${PYTHON_BIN}" scripts/09_benchmark.py \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --num-runs "${NUM_RUNS}" \
  --prompt-mode synthetic \
  --prompt-variants 1 \
  --output results/benchmark_synthetic.csv

echo "== 09B. Throughput benchmark: diverse prompts, default 12 configs =="
"${PYTHON_BIN}" scripts/09_benchmark.py \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --num-runs "${NUM_RUNS}" \
  --prompt-mode diverse \
  --prompt-variants "${PROMPT_VARIANTS}" \
  --output results/benchmark_diverse.csv

echo "== 09C. Throughput benchmark: batch-size coverage configs =="
"${PYTHON_BIN}" scripts/09_benchmark.py \
  --model-id "${MODEL_ID}" \
  --device "${DEVICE}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --num-runs "${NUM_RUNS}" \
  --prompt-mode synthetic \
  --prompt-variants 1 \
  --configs "${BATCH_BENCHMARK_CONFIGS}" \
  --output results/benchmark_batch.csv

echo "All requested A100 CUDA experiment commands completed."
