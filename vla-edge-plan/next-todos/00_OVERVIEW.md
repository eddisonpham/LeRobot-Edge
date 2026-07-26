# Next Todos — lerobot_edge Follow-Up Work

> Created: July 26, 2026
> Status: Post-initial-implementation (67 tests passing, 5 ONNX skipped)

## What's Done

- Plugin registration via draccus ChoiceRegistry (6 policy variants)
- CompressedPolicy wrapper satisfying PreTrainedPolicy interface
- Dynamic INT8 quantization with memory measurement
- ONNX export and ONNX Runtime inference backend
- TensorRT export (optional, guarded behind dependency)
- Teacher-student distillation training loop
- Benchmark harness with latency/memory/throughput measurement
- Edge/cloud confidence-based router
- Pareto frontier report generation
- Comprehensive unit test suite (67 passing)
- README, LICENSE, pyproject.toml, device configs

## What's Not Done (Code Reviewer Findings)

### Critical (must fix before claiming project is complete)
1. `static_int8_quantize` is broken — calibration loop always runs 1 step, passes dict instead of tensor
2. No integration smoke test with real SmolVLA (milestone M2 acceptance criterion)
3. `benchmark.py` CLI entry point is a stub with a TODO comment
4. `quantize.py` CLI entry point is a stub

### Important (should fix for production quality)
5. `pyproject.toml` uses non-standard build-backend (`setuptools.backends._legacy:_Backend`)
6. `distill.py` main() raises NotImplementedError but is in `[project.scripts]`
7. ONNX export tests are skipped (onnx not installed) — need integration tests
8. No CI/CD pipeline (GitHub Actions)

### Nice to Have (stretch goals)
9. Migrate from deprecated `torch.ao.quantization` to `torchao`
10. Add 4-bit quantization verification
11. Real SmolVLA PushT/LIBERO benchmark results
12. Pareto plot with real numbers

## Document Map

| File | Purpose |
|---|---|
| `01_CRITICAL_FIXES.md` | Bugs and broken functionality that must be fixed |
| `02_INTEGRATION_TESTS.md` | End-to-end smoke tests with real models |
| `03_CLI_IMPLEMENTATION.md` | Complete CLI entry points |
| `04_BUILD_AND_CI.md` | Build system fixes and CI/CD pipeline |
| `05_BENCHMARK_REAL_MODELS.md` | Run benchmarks with actual SmolVLA checkpoints |
| `06_QUALITY_POLISH.md` | Code quality, documentation, and polish |
