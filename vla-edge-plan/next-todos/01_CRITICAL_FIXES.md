# 01 — Critical Fixes

> Priority: P0 (must fix before claiming project is complete)
> Estimated effort: 2-4 hours

## 1.1 Fix `static_int8_quantize` in `lerobot_edge/quantize.py`

**Problem:** The calibration loop `for i in range(min(num_calibration_steps, 1))` always runs exactly 1 step regardless of the requested number. And `model_prepared(calibration_data)` passes a `dict` where a tensor batch is expected. This function silently falls back to dynamic quantization every time.

**Subtasks:**
- [ ] Extract a tensor batch from `calibration_data` dict (iterate through dict values, find first tensor)
- [ ] Fix the calibration loop to actually iterate `num_calibration_steps` times
- [ ] Add proper input validation and error handling
- [ ] Add unit test for static quantization with mock calibration data
- [ ] Verify the function doesn't crash with a real model

**Acceptance criteria:**
- `static_int8_quantize` runs for the specified number of calibration steps
- Returns a quantized model (not falling back to dynamic)
- Function doesn't silently fall back to dynamic quantization
- Unit test passes with mock calibration data

## 1.2 Fix `benchmark.py` CLI entry point

**Problem:** `main()` function in `benchmark.py` has a `# TODO: Load checkpoint and create backends for each variant` comment and doesn't actually load or benchmark anything.

**Subtasks:**
- [ ] Implement checkpoint loading via `lerobot.policies.factory.make_policy`
- [ ] Create backends for each variant (identity, quant_int8, onnx_fp32, onnx_int8)
- [ ] Run benchmark on each variant
- [ ] Save results to output directory
- [ ] Add error handling for missing dependencies

**Acceptance criteria:**
- `lerobot-edge-benchmark --checkpoint=lerobot/smolvla_base --variants=edge_identity` produces results
- Results saved to specified output directory

## 1.3 Fix `quantize.py` CLI entry point

**Problem:** `main()` function logs info but doesn't actually quantize anything.

**Subtasks:**
- [ ] Implement checkpoint loading
- [ ] Apply specified quantization method
- [ ] Save quantized checkpoint to output directory
- [ ] Add CLI argument for quantization method selection

**Acceptance criteria:**
- `lerobot-edge-quantize --source=lerobot/smolvla_base --output=./quantized --method=dynamic_int8` works
- Output directory contains quantized checkpoint

## 1.4 Fix `distill.py` CLI entry point

**Problem:** `main()` raises `NotImplementedError` but is still listed in `[project.scripts]` in `pyproject.toml`.

**Subtasks:**
- [ ] Either implement a basic CLI (load teacher, create student, run distillation)
- [ ] Or remove the entry from `pyproject.toml` until fully implemented
- [ ] If implementing: add argument parsing, teacher loading, student creation, dataset loading

**Acceptance criteria:**
- Users get a clear error message or the CLI actually works
- No dead entry points in `pyproject.toml`

## 1.5 Fix `report.py` numpy import guard

**Problem:** `report.py` imports `numpy as np` at module level without an explicit guard. While numpy is a transitive dependency, this should be handled gracefully.

**Subtasks:**
- [ ] Add try/except import guard for numpy
- [ ] Provide helpful error message if numpy not installed
- [ ] Guard matplotlib import (already done, verify)

**Acceptance criteria:**
- `import lerobot_edge.report` works even without numpy installed
- Clear error message if numpy missing

## 1.6 Clean up dead CLI entry points in `pyproject.toml`

**Problem:** `lerobot-edge-distill` is listed in `[project.scripts]` but raises NotImplementedError.

**Subtasks:**
- [ ] Remove `lerobot-edge-distill` from `[project.scripts]` until implemented
- [ ] Verify all remaining entry points work
- [ ] Add comments for future entry points

**Acceptance criteria:**
- No dead entry points in `pyproject.toml`
- All listed entry points produce valid output or clear errors
