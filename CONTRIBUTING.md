# Contributing

## Setup

```bash
git clone https://github.com/lerobot-edge/lerobot-edge.git
cd lerobot-edge
make install-dev
```

## Commands

| `make install-dev` | Editable install with all deps |
| `make test` | Fast unit tests |
| `make test-all` | Full suite including integration |
| `make lint` | Ruff check |
| `make format` | Ruff auto-fix |
| `make typecheck` | Mypy |

## Code style

- Ruff for lint/format (100 char line length, configured in `pyproject.toml`)
- Mypy for type checking
- Run `make ci` before committing

## Tests

```bash
pytest tests/test_quantize.py -v                     # one file
pytest tests/test_quantize.py::TestQuantizedBackend -v  # one class
```

Tests use `pytest.mark.slow` and `pytest.mark.integration` to separate fast from heavy.

## Adding a backend

1. Add class in appropriate module.
2. Register config in `core/configs.py` with `@PreTrainedConfig.register_subclass("edge_your_variant")`.
3. Add unit tests, guarding optional deps with `HAS_*` flags.
4. Run `make ci`.

## Issues

Include expected vs actual behavior, steps to reproduce, and Python/PyTorch versions.

## License

Apache-2.0.
