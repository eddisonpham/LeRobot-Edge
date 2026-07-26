#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-lerobot/smolvla_base}"
OUTPUT_DIR="${2:-./quantized}"
METHOD="${3:-dynamic_int8}"

echo "=== Quantizing ${CHECKPOINT} ==="
python -m lerobot_edge.compression.quantize \
  --source "${CHECKPOINT}" \
  --output "${OUTPUT_DIR}" \
  --method "${METHOD}" \
  --device cpu
echo "Quantized model saved to ${OUTPUT_DIR}/"
