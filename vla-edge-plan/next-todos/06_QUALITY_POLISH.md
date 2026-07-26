# 06 — Quality Polish

> Priority: P2 (nice to have for portfolio quality)
> Estimated effort: 4-8 hours

## 6.1 Migrate from deprecated `torch.ao.quantization` to `torchao`

**Problem:** PyTorch 2.11 deprecates `torch.ao.quantization` and recommends `torchao`.

**Subtasks:**
- [ ] Research `torchao` API for dynamic quantization
- [ ] Update `quantize.py` to use `torchao` if available
- [ ] Keep fallback to `torch.ao.quantization` for older PyTorch versions
- [ ] Add `torchao` to optional dependencies
- [ ] Test both paths

**Acceptance criteria:**
- `torchao` path works when installed
- Fallback works when `torchao` not installed
- No deprecation warnings

## 6.2 Add `__all__` to remaining modules

**Problem:** Some modules still missing `__all__` exports.

**Subtasks:**
- [ ] Audit all modules for `__all__` completeness
- [ ] Add `__all__` to any missing modules
- [ ] Verify wildcard imports only expose public API

**Acceptance criteria:**
- All modules have `__all__`
- No internal symbols exposed via wildcard imports

## 6.3 Improve error messages and logging

**Problem:** Some error messages are generic or missing.

**Subtasks:**
- [ ] Add helpful error messages for missing dependencies
- [ ] Add logging for long-running operations
- [ ] Add progress bars for benchmarks and training
- [ ] Improve CLI help text

**Acceptance criteria:**
- Missing dependency errors include installation instructions
- Long operations show progress
- CLI help is comprehensive

## 6.4 Documentation improvements

**Problem:** Documentation could be more comprehensive.

**Subtasks:**
- [ ] Add API reference documentation
- [ ] Add architecture diagram to README
- [ ] Add troubleshooting guide
- [ ] Add contribution guidelines
- [ ] Add changelog

**Acceptance criteria:**
- API reference exists
- README has clear architecture section
- Troubleshooting guide covers common issues
