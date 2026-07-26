# M3 — Quantization

## Objective
Implement post-training quantization for SmolVLA and wrap it as a
`CompressedPolicy` backend, runnable entirely on the laptop.

## Inputs
- Working `identity` backend from M2.
- `docs/agent-notes/baseline-results.json` from M1.

## Steps
1. Implement `lerobot_edge/quantize.py`:
   - Dynamic INT8 quantization (PyTorch's built-in dynamic quantization is the
     simplest starting point and needs no calibration data — start here).
   - Static INT8 quantization with a small calibration set drawn from a
     LeRobot dataset (stretch within this milestone, not required for M3 to
     be "done").
   - Optional 4-bit (e.g. via `bitsandbytes`) if it's compatible with
     SmolVLA's architecture — verify compatibility before committing time to
     it; if it's not straightforward, note why in `docs/agent-notes/api-map.md`
     and move on
     rather than forcing it.
2. Wrap the quantized model as a new `deploy_backend` value
   (`quant_int8`, `quant_int8_static`, `quant_4bit` as applicable).
3. Add unit tests: load SmolVLA, quantize, confirm output action shapes/dtypes
   match the FP32 model's, confirm memory footprint actually drops (measure
   it, don't assume).
4. Run the M1 baseline eval command with each new `deploy_backend` value.
   Record results the same way as M1 (device profile, commit hash, config,
   results) into `docs/agent-notes/results_quant.json`.

## Acceptance criteria
- At least dynamic INT8 quantization works end-to-end through `lerobot-eval`
  on the laptop and produces a real success-rate + latency + memory number.
- Memory footprint reduction is measured and recorded, not assumed.
- Any variant that doesn't work (e.g. 4-bit incompatibility) is documented
  with the actual error/reason, not silently dropped.

## Handoff to M4
`results_quant.json` plus a working quantized backend — M4 exports this (or
the FP32 baseline) to ONNX for a runtime-level speedup on top of quantization.
