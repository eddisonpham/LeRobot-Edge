# 03 — CLI Implementation

> Priority: P1 (should fix for production quality)
> Estimated effort: 3-5 hours

## 3.1 Complete `lerobot-edge-benchmark` CLI

**Problem:** The benchmark CLI entry point has a TODO comment and doesn't load checkpoints or run benchmarks.

**Subtasks:**
- [ ] Load checkpoint via `lerobot.policies.factory.make_policy`
- [ ] Create backends for each requested variant
- [ ] Build dummy input from policy config
- [ ] Run benchmark on each variant using `benchmark_backend()`
- [ ] Save results to output directory (JSON + CSV)
- [ ] Print summary table to stdout
- [ ] Add progress bar for long benchmarks
- [ ] Handle errors gracefully (missing deps, GPU OOM, etc.)

**Acceptance criteria:**
- `lerobot-edge-benchmark --checkpoint=lerobot/smolvla_base --variants=edge_identity edge_quant_int8` works
- Results saved to output directory
- Summary printed to stdout

## 3.2 Complete `lerobot-edge-quantize` CLI

**Problem:** The quantize CLI entry point logs info but doesn't actually quantize.

**Subtasks:**
- [ ] Load source policy from checkpoint
- [ ] Apply specified quantization method
- [ ] Save quantized checkpoint to output directory
- [ ] Print before/after memory comparison
- [ ] Verify quantized model produces valid output

**Acceptance criteria:**
- `lerobot-edge-quantize --source=lerobot/smolvla_base --output=./quantized --method=dynamic_int8` works
- Output directory contains valid quantized checkpoint
- Memory comparison printed

## 3.3 Complete `lerobot-edge-report` CLI

**Problem:** The report CLI works but needs polish.

**Subtasks:**
- [ ] Add argument for results directory (already exists)
- [ ] Add argument for output directory (already exists)
- [ ] Print summary table to stdout
- [ ] Generate both latency and memory Pareto plots
- [ ] Handle empty results directory gracefully

**Acceptance criteria:**
- `lerobot-edge-report --results-dir=./benchmark_results --output-dir=./docs` works
- Both Pareto plots generated
- Markdown report generated

## 3.4 Complete `lerobot-edge-distill` CLI

**Problem:** Raises NotImplementedError.

**Subtasks:**
- [ ] Implement teacher loading from checkpoint
- [ ] Implement student creation (smaller architecture)
- [ ] Implement dataset loading
- [ ] Run distillation training loop
- [ ] Save distilled checkpoint
- [ ] Print training metrics

**Acceptance criteria:**
- `lerobot-edge-distill --teacher=lerobot/smolvla_base --output=./distilled` works
- Distilled checkpoint saved
- Training metrics printed
