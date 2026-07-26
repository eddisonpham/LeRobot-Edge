#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-lerobot/smolvla_base}"
OUTPUT_DIR="${2:-benchmark_results}"
WARMUP="${3:-3}"
NUM_RUNS="${4:-10}"
DEVICE="${5:-cuda}"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "No CUDA GPU detected, falling back to CPU"
  DEVICE="cpu"
fi

echo "=== Benchmarking ${CHECKPOINT} ==="
python -m lerobot_edge.evaluation.compare_backends \
  --checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}" \
  --warmup "${WARMUP}" \
  --num-runs "${NUM_RUNS}" \
  --output "${OUTPUT_DIR}/smolvla_real.json"
echo "Results saved to ${OUTPUT_DIR}/"
