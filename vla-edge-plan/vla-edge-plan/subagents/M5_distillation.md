# M5 — Teacher → Student Distillation

## Objective
Train a smaller student policy that imitates a larger teacher (either SmolVLA
itself distilled further, or a larger LeRobot VLA such as π0.5 or GR00T N1.7
as teacher, with SmolVLA-sized or smaller as student), evaluated with the same
benchmark harness from M4.

## Inputs
- `lerobot_edge/benchmark.py` from M4.
- `docs/agent-notes/api-map.md` — reuse the existing `lerobot-train` data
  loading/checkpointing infrastructure rather than writing a new training
  loop from scratch.

## Steps
1. Decide teacher/student pair. Recommended default: teacher = SmolVLA
   fine-tuned (or off-the-shelf) on a chosen sim task; student = a smaller
   architecture variant (fewer transformer layers / smaller hidden dim) —
   check `lerobot_edge/quantize.py`'s notes on architecture compatibility from M3
   before committing to a student shape.
2. Implement `lerobot_edge/distill.py`:
   - Loss: action-chunk regression against the teacher's outputs (and/or KL
     on the flow-matching action distribution if that's how the real policy
     head works — confirm against the actual SmolVLA implementation rather
     than assuming a standard classification-style KL applies).
   - Reuse `LeRobotDataset` loading exactly as `lerobot-train` does.
3. **Try laptop CPU first with a short run** before assuming cloud is
   required — measure actual wall-clock for one epoch on a small dataset
   subset. Only move to the cloud config (`04_INFRA_LOCAL_AND_CLOUD.md`, HF
   Jobs) if the laptop run is impractically slow (define "impractical" as
   >~few hours for a meaningful training run).
4. Run the distilled student through `benchmark.py` and `lerobot-eval`,
   recording results the same way as prior milestones.

## Acceptance criteria
- A distilled student model exists, with a documented parameter count
  reduction vs. the teacher.
- Distilled student's success rate on the sim benchmark is recorded honestly,
  including if it's meaningfully worse than the teacher's — this tradeoff
  *is* the result, not a failure to hide.
- Whichever compute path was used (laptop or cloud) is documented with actual
  wall-clock time, so the README can honestly state training cost.

## Handoff to M6
The distilled checkpoint, ready to optionally combine with quantization/ONNX
export (`distilled_onnx_int8`) and, if pursued, the TensorRT step.
