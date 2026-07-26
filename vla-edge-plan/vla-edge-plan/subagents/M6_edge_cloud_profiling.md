# M6 — TensorRT / Edge Profiling (Optional, Cloud/Edge-Only Step)

## Objective
Add a TensorRT export path for a real NVIDIA GPU/Jetson-class target, clearly
marked as an optional extension beyond the laptop-only core pipeline. This is
the milestone most directly relevant to NVIDIA-specific roles, but is
explicitly gated as optional so the project's core value doesn't depend on
having cloud GPU access.

## Inputs
- `onnx_int8` and/or `distilled_onnx_int8` variants from M4/M5.
- A cloud GPU instance (see `04_INFRA_LOCAL_AND_CLOUD.md`) — this step cannot
  run on a laptop without an NVIDIA GPU and matching driver/TensorRT install.

## Steps
1. Guard this entire module behind an optional dependency check
   (`HAS_TENSORRT`) so the rest of the repo remains installable and usable
   without it, per `03_EXTENSION_PLAN.md` §4.
2. Implement `lerobot_edge/export_tensorrt.py`: ONNX → TensorRT engine build,
   following NVIDIA's documented workflow (quantize → ONNX → TensorRT engine
   → runtime). Reuse the ONNX artifact from M4/M5 rather than re-exporting.
3. Benchmark on the cloud GPU instance using the same `benchmark.py` harness
   from M4 — this is why building a reusable harness in M4 mattered.
4. If you have or can access actual Jetson-class hardware, repeating this on
   real edge hardware is the single highest-signal addition to the whole
   project for an NVIDIA-facing resume — but treat it as a stretch goal, not
   a requirement. A cloud GPU number is still a legitimate, honestly-labeled
   result if edge hardware isn't available.

## Acceptance criteria
- The rest of the repo (laptop-only paths) is unaffected by whether this
  module is installed or run.
- If run: a TensorRT engine benchmark result exists, labeled clearly with
  which GPU/hardware it ran on (do not present a cloud GPU number as if it
  were an edge-device number).

## Handoff to M7
Whatever set of variants you actually have results for (this milestone may
be skipped or partially completed without blocking the rest of the project) —
M7 aggregates everything into the final report regardless of how far M6 got.
