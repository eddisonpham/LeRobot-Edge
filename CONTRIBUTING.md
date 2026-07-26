# Contributing to LeRobot Edge

Thanks for your interest in contributing. This guide covers local setup, testing, and code style.

## Prerequisites

- Python 3.10+
- Git
- [LeRobot](https://github.com/huggingface/lerobot) installed in your environment
- CUDA toolkit (optional, for GPU quantization and TensorRT)

## Setup

```bash
git clone https://github.com/lerobot-edge/lerobot-edge.git
cd lerobot-edge
make install-dev
```

This installs the package in editable mode with all optional dependencies (ONNX, bitsandbytes, wandb, dev tools).

## Common Commands

| Command | What it does |
|---------|-------------|
| `make install-dev` | Install in editable mode with dev dependencies |
| `make test` | Run fast unit tests (excludes slow and integration) |
| `make test-all` | Run full test suite including integration |
| `make lint` | Check code with ruff |
| `make format` | Auto-format code with ruff |
| `make typecheck` | Run mypy type checks |
| `make ci` | Install + lint + typecheck + test |
| `make benchmark` | Run the benchmark CLI |
| `make quantize` | Run the quantize CLI |

## Running Tests

```bash
# Fast unit tests (no model downloads)
make test

# Full suite including integration tests (requires network)
make test-all

# Specific test file
pytest tests/test_quantize.py -v

# Specific test class
pytest tests/test_quantize.py::TestQuantizedBackend -v
```

Tests use `pytest.mark.slow` and `pytest.mark.integration` to separate fast local runs from heavy end-to-end tests.

## Code Style

Linting and formatting use [ruff](https://docs.astral.sh/ruff/):

```bash
make lint       # check
make format     # auto-fix
```

Type checking uses mypy:

```bash
make typecheck
```

Configuration lives in `pyproject.toml`. The line length is 100 characters.

## Project Structure

```
lerobot_edge/
  core/           # Base classes, configs, utilities
  compression/    # Quantization and distillation
  export/         # ONNX and TensorRT export
  evaluation/     # Benchmarking, metrics, quality gates
  tracking/       # W&B experiment tracking integration
tests/
  test_*.py       # Unit and integration tests
benchmarks/
  bench_*.py      # Standalone performance benchmarks
scripts/
  *.sh            # Utility scripts
```

## Making Changes

1. Create a branch from `main`.
2. Make your changes with tests.
3. Run `make ci` before committing.
4. Open a pull request against `main`.

Keep changes focused. Each PR should address one concern.

## Adding a New Backend

1. Create the backend class in the appropriate module (e.g., `compression/quantize.py`).
2. Register a config in `core/configs.py` using `@PreTrainedConfig.register_subclass("edge_your_variant")`.
3. Add unit tests in `tests/`.
4. If the backend has an optional dependency, guard it with a `HAS_*` flag and skip tests gracefully.

## Reporting Issues

Open an issue on GitHub with:
- What you expected to happen.
- What actually happened.
- Steps to reproduce.
- Python and PyTorch versions.

## License

By contributing, you agree that your contributions will be licensed under Apache-2.0.
