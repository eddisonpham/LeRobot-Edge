# 05 — Benchmark with Real Models

> Priority: P1 (core value proposition)
> Estimated effort: 8-12 hours

## 5.1 SmolVLA baseline benchmark

**Problem:** No benchmark results with actual SmolVLA checkpoint exist.

**Subtasks:**
- [ ] Download `lerobot/smolvla_base` from HuggingFace Hub
- [ ] Run baseline FP32 benchmark on PushT (10 episodes)
- [ ] Record: success rate, latency, memory, throughput
- [ ] Save results to `docs/agent-notes/baseline-results.json`
- [ ] Verify results are reproducible (run twice, compare)

**Acceptance criteria:**
- Baseline results file exists with real numbers
- Results are reproducible within 5%
- Memory footprint measured accurately

## 5.2 Quantized variant benchmarks

**Problem:** No benchmark results for quantized variants.

**Subtasks:**
- [ ] Load SmolVLA and apply dynamic INT8 quantization
- [ ] Run quantized model on PushT (10 episodes)
- [ ] Record: success rate, latency, memory, throughput
- [ ] Compare against FP32 baseline
- [ ] Calculate memory savings and speedup
- [ ] Save results to `docs/agent-notes/results_quant.json`

**Acceptance criteria:**
- Quantized results file exists
- Memory savings measured (not assumed)
- Speedup calculated

## 5.3 ONNX Runtime benchmarks

**Problem:** No benchmark results for ONNX variants.

**Subtasks:**
- [ ] Export SmolVLA to ONNX (FP32)
- [ ] Run ONNX FP32 on PushT (10 episodes)
- [ ] Record: success rate, latency, memory, throughput
- [ ] Export quantized SmolVLA to ONNX (INT8)
- [ ] Run ONNX INT8 on PushT (10 episodes)
- [ ] Compare all variants
- [ ] Save results to `docs/agent-notes/results_onnx.json`

**Acceptance criteria:**
- ONNX results files exist
- At least one ONNX variant shows measurable improvement
- Results table generated

## 5.4 Generate Pareto frontier plots

**Problem:** No Pareto plots with real numbers exist.

**Subtasks:**
- [ ] Aggregate all results JSON files
- [ ] Generate latency vs success rate Pareto plot
- [ ] Generate memory vs success rate Pareto plot
- [ ] Save plots as PNG files
- [ ] Generate markdown results table
- [ ] Update README with real results

**Acceptance criteria:**
- Pareto plots exist with real numbers
- Results table in README reflects actual measurements
- Plots are clear and informative
