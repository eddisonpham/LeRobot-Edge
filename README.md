# lerobot-edge

> Policy compression and edge deployment plugin for [HuggingFace LeRobot](https://github.com/huggingface/lerobot)

**lerobot_edge** is a standalone, pip-installable extension package that registers with LeRobot's public policy plugin system to add quantization, ONNX export, teacher-student distillation, and automated benchmarking — so that a VLA trained in LeRobot can be evaluated not just on "does it work" but on "does it work fast enough, small enough, and cheap enough to actually ship."

## Why This Matters

Every 2026 industry writeup on VLA deployment says the same thing: getting a multi-hundred-million-to-billion-parameter policy to run at useful control rates (5–30 Hz) on a power-and-latency-constrained board is the actual bottleneck between a lab demo and a shipped robot.

**lerobot_edge** bridges this gap: a standalone package that takes any LeRobot policy checkpoint and produces a compressed, benchmarked, deployment-ready variant — installed as a plugin, without touching LeRobot's own source.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      lerobot_edge Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  checkpoint (FP32, e.g. lerobot/smolvla_base)                  │
│        │                                                        │
│        ├─► quantize.py ──► INT8 / 4-bit weights ─┐             │
│        │                                          │             │
│        ├─► distill.py  ──► smaller student ───────┤             │
│        │   (teacher: FP32 or a larger VLA)         │             │
│        │                                          │             │
│        └──────────────────────────────────────────┘             │
│                     │                                           │
│                     ├─► export_onnx.py ──► ONNX Runtime session │
│                     │        │                                  │
│                     │        └─► export_tensorrt.py (GPU only)  │
│                     │                                           │
│                     ▼                                           │
│         benchmark.py (latency, memory, throughput)              │
│                     │                                           │
│                     ▼                                           │
│      lerobot-eval on PushT / LIBERO, once per variant          │
│                     │                                           │
│                     ▼                                           │
│           report.py -> Pareto plot + results table              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Basic installation
pip install lerobot-edge

# With ONNX support
pip install "lerobot-edge[onnx]"

# With quantization (bitsandbytes)
pip install "lerobot-edge[quantize]"

# With TensorRT (GPU only)
pip install "lerobot-edge[tensorrt]"

# All optional dependencies
pip install "lerobot-edge[all]"
```

## Quickstart

### 1. Baseline (unmodified LeRobot)

```bash
# Install LeRobot
pip install "lerobot>=0.6.0"

# Run baseline evaluation
lerobot-eval \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --env.type=pusht \
  --eval.n_episodes=10
```

### 2. Identity Plugin (proves plugin hook works)

```bash
# Install lerobot-edge
pip install -e .

# Run with edge_identity plugin
lerobot-eval \
  --policy.type=edge_identity \
  --policy.pretrained_path=lerobot/smolvla_base \
  --env.type=pusht \
  --eval.n_episodes=10
```

### 3. Quantized Variant

```bash
# Run with INT8 quantization
lerobot-eval \
  --policy.type=edge_quant_int8 \
  --policy.pretrained_path=lerobot/smolvla_base \
  --env.type=pusht \
  --eval.n_episodes=10
```

### 4. ONNX Runtime Variant

```bash
# Run with ONNX Runtime
lerobot-eval \
  --policy.type=edge_onnx_fp32 \
  --policy.pretrained_path=lerobot/smolvla_base \
  --env.type=pusht \
  --eval.n_episodes=10
```

## Available Policy Variants

| Variant | Description | Requirements |
|---------|-------------|--------------|
| `edge_identity` | Passthrough wrapper (no compression) | Core |
| `edge_quant_int8` | Dynamic INT8 quantization | Core |
| `edge_onnx_fp32` | ONNX Runtime inference (FP32) | `lerobot-edge[onnx]` |
| `edge_onnx_int8` | ONNX Runtime inference (INT8) | `lerobot-edge[onnx]` |
| `edge_distilled` | Teacher-student distilled | Core |
| `edge_distilled_onnx_int8` | Distilled + ONNX + INT8 | `lerobot-edge[onnx]` |

## Benchmarking

```bash
# Run full benchmark suite
lerobot-edge-benchmark \
  --checkpoint=lerobot/smolvla_base \
  --variants=edge_identity edge_quant_int8 edge_onnx_fp32 \
  --device-profile=laptop_cpu \
  --output-dir=benchmark_results

# Generate Pareto report
lerobot-edge-report \
  --results-dir=benchmark_results \
  --output-dir=docs
```

## Configuration

### Device Profiles

- `configs/device/laptop_cpu.yaml` — CPU-only development
- `configs/device/cloud_gpu.yaml` — GPU-accelerated training/benchmarking

### Custom Configuration

```python
from lerobot_edge.configs import EdgeQuantInt8Config

config = EdgeQuantInt8Config(
    quantize_dynamic=True,
    quantize_bits=8,
    benchmark_warmup=20,
    benchmark_num_runs=500,
)
```

## Project Structure

```
lerobot-edge/
├── pyproject.toml              # Package metadata and dependencies
├── README.md                   # This file
├── LICENSE                     # Apache-2.0
├── lerobot_edge/
│   ├── __init__.py             # Plugin registration
│   ├── configs.py              # Draccus config dataclasses
│   ├── base.py                 # CompressedPolicy wrapper + backends
│   ├── quantize.py             # Post-training quantization
│   ├── export_onnx.py          # ONNX export + Runtime backend
│   ├── export_tensorrt.py      # TensorRT export (optional)
│   ├── distill.py              # Teacher-student distillation
│   ├── benchmark.py            # Latency/memory/throughput harness
│   ├── router.py               # Edge/cloud confidence router
│   └── report.py               # Pareto frontier plotting
├── tests/
│   ├── test_configs.py         # Config registration tests
│   ├── test_base.py            # CompressedPolicy tests
│   ├── test_quantize.py        # Quantization tests
│   ├── test_export_onnx.py     # ONNX export tests
│   ├── test_distill.py         # Distillation tests
│   ├── test_benchmark.py       # Benchmark harness tests
│   └── test_router.py          # Router tests
├── configs/
│   └── device/
│       ├── laptop_cpu.yaml     # CPU device profile
│       └── cloud_gpu.yaml      # GPU device profile
└── docs/
    └── agent-notes/
        └── api-map.md          # LeRobot API reference
```

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=lerobot_edge

# Type checking
mypy lerobot_edge/

# Linting
ruff check lerobot_edge/
```

## How the Plugin Works

lerobot_edge participates in LeRobot's plugin system via `draccus.ChoiceRegistry`:

1. **Registration**: Config classes use `@PreTrainedConfig.register_subclass("edge_*")`
2. **Discovery**: LeRobot's factory fallback finds registered configs via naming conventions
3. **Import**: Policy classes are dynamically imported (`configuration_X` → `modeling_X`)

This means **zero changes to LeRobot's source** — just `pip install lerobot-edge` and the variants are available via `--policy.type=edge_*`.

## Attribution

lerobot_edge is a policy-compression and edge-deployment plugin for [🤗 LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0). It installs alongside a stock `pip install lerobot` and adds no changes to LeRobot itself.

## License

Apache-2.0 — see [LICENSE](LICENSE) for details.

## What I'd Do Next

1. **Real SmolVLA evaluation**: Run full PushT/LIBERO benchmarks with actual SmolVLA checkpoints
2. **4-bit quantization**: Verify bitsandbytes compatibility with SmolVLA's architecture
3. **TensorRT on Jetson**: Test on actual edge hardware for real-world deployment numbers
4. **Action chunking optimization**: Optimize the ONNX export for temporal action predictions
5. **Streaming inference**: Add async inference pipeline for real-time robot control
