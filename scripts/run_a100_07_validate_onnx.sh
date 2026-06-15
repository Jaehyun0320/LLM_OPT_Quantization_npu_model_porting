#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROVIDER="${PROVIDER:-cuda}"
TOLERANCE="${TOLERANCE:-1e-3}"
LOOP_STEPS="${LOOP_STEPS:-4}"
RUN_NO_CACHE="${RUN_NO_CACHE:-1}"
RUN_WITH_CACHE="${RUN_WITH_CACHE:-1}"
LOG_FILE="${LOG_FILE:-results/a100_07_validate_onnx.log}"

mkdir -p results
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ "${RUN_NO_CACHE}" == "1" ]]; then
  echo "== 07. ONNX validation: no-cache =="
  "${PYTHON_BIN}" scripts/07_validate_onnx.py \
    --metadata results/onnx_export_no_cache.json \
    --provider "${PROVIDER}" \
    --tolerance "${TOLERANCE}" \
    --output results/onnx_validation_no_cache.json
fi

if [[ "${RUN_WITH_CACHE}" == "1" ]]; then
  if "${PYTHON_BIN}" -c "import json; import sys; sys.exit(0 if json.load(open('results/onnx_export_with_cache.json')).get('success') else 1)"; then
    echo "== 07. ONNX validation: with explicit KV-cache =="
    "${PYTHON_BIN}" scripts/07_validate_onnx.py \
      --metadata results/onnx_export_with_cache.json \
      --provider "${PROVIDER}" \
      --tolerance "${TOLERANCE}" \
      --loop-steps "${LOOP_STEPS}" \
      --output results/onnx_validation_with_cache.json
  else
    echo "Skipping with-cache ONNX validation because export did not succeed."
    echo "See results/onnx_export_with_cache.json for failure details."
  fi
fi

echo "A100 07 ONNX validation commands completed."
