#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-lerobot/smolvla_base}"
OUTPUT_DIR="${2:-benchmark_results}"
WARMUP="${3:-3}"
NUM_RUNS="${4:-10}"

echo "=== Benchmarking ${CHECKPOINT} ==="
python -m lerobot_edge.evaluation.compare_backends \
  --checkpoint "${CHECKPOINT}" \
  --device cpu \
  --warmup "${WARMUP}" \
  --num-runs "${NUM_RUNS}" \
  --output "${OUTPUT_DIR}/smolvla_real.json"
echo "Results saved to ${OUTPUT_DIR}/"
