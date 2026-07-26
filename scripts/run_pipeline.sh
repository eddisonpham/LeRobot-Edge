#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-lerobot/smolvla_base}"
OUTPUT_DIR="${2:-benchmark_results}"

echo "=== Full pipeline: ${CHECKPOINT} ==="
echo "--- Benchmark ---"
bash "$(dirname "$0")/benchmark.sh" "${CHECKPOINT}" "${OUTPUT_DIR}" 5 10
echo ""
echo "--- Quantize ---"
bash "$(dirname "$0")/quantize.sh" "${CHECKPOINT}" "./quantized" dynamic_int8
echo ""
echo "=== Pipeline complete ==="
echo "Results: ${OUTPUT_DIR}/"
