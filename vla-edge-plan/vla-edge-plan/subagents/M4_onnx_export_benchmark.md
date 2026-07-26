# M4 — ONNX Export + Benchmark Harness

## Objective
Export the policy (FP32 and/or quantized) to ONNX, run it through ONNX Runtime,
and build the benchmark harness that every later milestone will reuse to
produce comparable numbers.

## Inputs
- Working `quant_int8` backend from M3.
- `docs/agent-notes/results_quant.json`.

## Steps
1. Implement `lerobot_edge/export_onnx.py`:
   - Export the policy's forward pass (vision + language + action head — scope
     this carefully; VLA models often have dynamic/control-flow-heavy parts
     that don't trace cleanly, budget time to work around export issues).
   - Wrap the resulting ONNX Runtime `InferenceSession` (CPU execution
     provider) as a new `deploy_backend` (`onnx_fp32`, `onnx_int8`).
2. Implement `lerobot_edge/benchmark.py`:
   - Measures: mean/p50/p95 latency per inference, peak memory, throughput
     (inferences/sec) — averaged over enough runs to be stable (discard a
     warmup period).
   - Outputs a single structured JSON/CSV row per (backend, device_profile)
     combination, including the reproducibility fields from
     `04_INFRA_LOCAL_AND_CLOUD.md` (commit hash, config, timestamp).
   - Make this a reusable CLI (`lerobot-deploy-benchmark` or similar,
     following the repo's own script-naming convention from M0), not a
     one-off script — every later milestone calls this same tool.
3. Run the full benchmark matrix so far: `none` (FP32 baseline), `identity`,
   `quant_int8`, `onnx_fp32`, `onnx_int8` — all on the laptop CPU profile.
4. Run `lerobot-eval` for each variant on the sim benchmark to get
   success-rate numbers alongside the latency/memory numbers.

## Acceptance criteria
- `benchmark.py` produces consistent, reproducible numbers across repeated
  runs (rerun each backend twice, confirm numbers are close).
- At least one ONNX variant shows a measurable latency and/or memory
  improvement over FP32 on the laptop CPU profile — record the actual number,
  whatever it is, even if smaller than hoped.
- A results table (backend × latency × memory × success rate) exists as a
  checked-in artifact (not just console output).

## Handoff to M5
The benchmark harness (reused, not rebuilt, from here on) plus a full
laptop-only results table covering quantization and ONNX export.
