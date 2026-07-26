# 04 — Build and CI

> Priority: P0 (critical for portfolio proof - plugin architecture must be verifiable from clean install)
> Estimated effort: 3-5 hours

## 4.1 Fix `pyproject.toml` build-backend

**Problem:** Uses non-standard `setuptools.backends._legacy:_Backend` which may cause issues with modern pip/build tools.

**Subtasks:**
- [ ] Change build-backend to `setuptools.build_meta`
- [ ] Verify `pip install -e .` works
- [ ] Verify `python -m build` produces valid wheel
- [ ] Test on Python 3.10, 3.11, 3.12

**Acceptance criteria:**
- `pip install -e .` succeeds
- `python -m build` produces valid wheel
- Package installs cleanly on all supported Python versions

## 4.2 Create GitHub Actions CI pipeline

**Problem:** No CI/CD pipeline exists.

**Subtasks:**
- [ ] Create `.github/workflows/test.yml`
- [ ] Run tests on push and PR
- [ ] Run on Python 3.10, 3.11, 3.12
- [ ] Run on Ubuntu and macOS (Windows optional)
- [ ] Install lerobot + lerobot-edge[dev] from scratch
- [ ] Run pytest with coverage
- [ ] Run ruff linting
- [ ] Run mypy type checking
- [ ] Post coverage report as PR comment

**Acceptance criteria:**
- CI runs on every push/PR
- Tests pass on all supported platforms
- Coverage report generated

## 4.3 Create pre-commit hooks

**Problem:** No code quality enforcement.

**Subtasks:**
- [ ] Create `.pre-commit-config.yaml`
- [ ] Add ruff formatter
- [ ] Add ruff linter
- [ ] Add mypy type checking
- [ ] Add pytest runner
- [ ] Test pre-commit hooks

**Acceptance criteria:**
- `pre-commit install` works
- Hooks run on commit
- Code quality enforced automatically
