.PHONY: install install-dev test test-verbose lint format typecheck clean benchmark quantize evaluate

CHECKPOINT ?= lerobot/smolvla_base
OUTPUT_DIR ?= benchmark_results

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,all]"

test:
	pytest tests/ -v --tb=short -m "not slow and not integration"

test-all:
	pytest tests/ -v --tb=short

test-verbose:
	pytest tests/ -v --tb=long -s

test-evaluation:
	pytest tests/test_evaluation.py tests/test_monitoring.py -v --tb=short

lint:
	ruff check lerobot_edge/

format:
	ruff format lerobot_edge/

typecheck:
	mypy lerobot_edge/ --ignore-missing-imports

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache wandb_logs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

benchmark:
	lerobot-edge-benchmark --checkpoint $(CHECKPOINT) --variants edge_identity edge_quant_int8 --device-profile laptop_cpu --output-dir $(OUTPUT_DIR)

quantize:
	lerobot-edge-quantize --source $(CHECKPOINT) --output ./quantized --method dynamic_int8

evaluate:
	lerobot-edge-report --results-dir $(OUTPUT_DIR) --output-dir docs

ci: install-dev lint typecheck test
