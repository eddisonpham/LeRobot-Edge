# lerobot-edge

> Policy compression and edge deployment plugin for [HuggingFace LeRobot](https://github.com/huggingface/lerobot)

**lerobot_edge** plugs into LeRobot's policy system to add quantization (INT8 via torchao), ONNX export, TensorRT inference, teacher-student distillation, benchmarking with quality gates, evaluation metrics, and experiment tracking for edge deployment.

## Architecture

```mermaid
graph TD
    A[FP32 Checkpoint<br/>e.g. lerobot/smolvla_base] --> B{Compression Path}
    B --> C[compression/<br/>INT8 via torchao / 4-bit]
    B --> D[compression/<br/>Teacher-Student]
    C --> E[Compressed Model]
    D --> E
    E --> F[export/<br/>ONNX Runtime]
    F --> G[export/<br/>TensorRT GPU]
    E --> H[evaluation/<br/>Benchmark + Metrics]
    F --> H
    G --> H
    H --> I[evaluation/<br/>Pareto Report]
    I --> J[tracking/<br/>W&B Tracking]
```

### Package Layout

```
lerobot_edge/
  core/              # Backends, configs, router, utilities
  compression/       # Quantization (INT8 via torchao, 4-bit) and distillation
  export/            # ONNX export and TensorRT export with I/O bindings
  evaluation/        # Benchmark harness, metrics, Pareto reports, quality gate
  tracking/          # W&B experiment tracking, local JSON fallback
scripts/             # Shell scripts for common operations
```

## Installation

```bash
pip install lerobot-edge
pip install "lerobot-edge[onnx]"        # ONNX support
pip install "lerobot-edge[quantize]"    # bitsandbytes
pip install "lerobot-edge[tensorrt]"    # TensorRT (GPU, requires pycuda)
pip install "lerobot-edge[wandb]"       # W&B experiment tracking
pip install "lerobot-edge[all]"         # everything
```

## Quickstart

### Evaluate with edge plugin

```bash
lerobot-eval \
  --policy.type=edge_quant_int8 \
  --policy.pretrained_path=lerobot/smolvla_base \
  --env.type=pusht --eval.n_episodes=10
```

### Benchmark

```bash
bash scripts/benchmark.sh lerobot/smolvla_base
```

### Quantize

```bash
bash scripts/quantize.sh lerobot/smolvla_base ./quantized dynamic_int8
```

### Full pipeline

```bash
bash scripts/run_pipeline.sh lerobot/smolvla_base
```

## Available Variants

| Variant | Description | Requirements |
|---------|-------------|--------------|
| `edge_identity` | Passthrough wrapper | Core |
| `edge_quant_int8` | Dynamic INT8 quantization (torchao) | Core |
| `edge_onnx_fp32` | ONNX Runtime (FP32) | `lerobot-edge[onnx]` |
| `edge_onnx_int8` | ONNX Runtime (INT8) | `lerobot-edge[onnx]` |
| `edge_distilled` | Teacher-student distilled | Core |
| `edge_distilled_onnx_int8` | Distilled + ONNX + INT8 | `lerobot-edge[onnx]` |

## Benchmark Results

SmolVLA (864 MB, ~450M params) — measured on NVIDIA RTX 5060 and Intel CPU. All benchmarks use batch_size=1 (single-sample inference, typical for real-time robotics control loops).

### GPU vs CPU Latency

| Backend | GPU Latency | GPU FPS | CPU Latency | CPU FPS | Overhead |
|---------|------------|---------|------------|---------|----------|
| Identity (FP32) | 1.47 ms | 681.8 | 3.75 ms | 266.8 | -- |
| Dynamic INT8 | 1.55 ms | 644.6 | 5.78 ms | 173.1 | GPU +5% / CPU +54% |
| FP32 + torch.compile | 1.87 ms | 534.8 | -- | -- | GPU +27% |
| INT8 + torch.compile | 2.50 ms | 399.5 | -- | -- | GPU +70% |

*batch_size=1, single-sample inference*

### Memory Footprint

| Format | Model Size | Compression |
|--------|-----------|-------------|
| FP32 (original) | 864 MB | -- |
| Dynamic INT8 | 864 MB | Weights FP32, activations quantized at runtime |
| Static INT8 | ~216 MB | 4x † |
| 4-bit NF4 | ~108 MB | 8x † |

† Estimated based on theoretical compression ratios. Not yet benchmarked on this model.

### Quality Gate

| Backend | Cosine Sim | MSE | Gate |
|---------|-----------|-----|------|
| Identity (FP32) | 1.0000 | 0.0000 | PASS |
| Dynamic INT8 (GPU) | 0.9971 | 0.0067 | PASS |
| Dynamic INT8 (CPU) | 0.9878 | 0.0101 | FAIL |
| INT8 + torch.compile | 0.9946 | 0.0086 | PASS |

### When to Use Each Backend

**GPU (recommended for latency-critical applications):**
- INT8 quantization adds only 5% overhead on GPU — INT8 tensor cores handle dequantization efficiently
- Quality gate passes (cos=0.9971) — quantization divergence is minimal
- Best for: NVIDIA GPUs, Jetson Orin, any edge device with CUDA support

**CPU (for resource-constrained devices):**
- INT8 quantization adds 54% overhead on CPU — dynamic quantization dispatch cost dominates
- CPU INT8 produces lower cosine similarity (0.9878 vs 0.9971 on GPU) due to dynamic quantization precision differences
- Best for: Raspberry Pi, Intel NUC, devices without GPU where memory savings matter more than speed

**torch.compile:**
- Adds compilation overhead on SmolVLA (~500M params) — 27% slower than FP32
- Compilation overhead dominates on smaller models; kernel fusion benefits emerge at scale where the JIT cost is amortized across many forward passes
- Best for: Larger models where JIT cost is amortized across many forward passes

### Dynamic vs Static INT8

- **Dynamic INT8** (default): Quantizes activations at runtime, no calibration data needed. Weights remain FP32. Better quality but no disk/memory savings.
- **Static INT8**: Quantizes weights offline using calibration data. 4x smaller model, but requires a calibration dataset and may lose more precision.

## Makefile

```bash
make install-dev    # pip install -e ".[dev,all]"
make test           # run non-slow tests
make lint           # ruff check
make format         # ruff format
make typecheck      # mypy
make ci             # lint + typecheck + test
make benchmark      # run benchmark
make quantize       # quantize checkpoint
make evaluate       # generate report
```

## API Usage

```python
from lerobot_edge.core.configs import EdgeQuantInt8Config
from lerobot_edge.compression.quantize import QuantizedBackend
from lerobot_edge.evaluation.gate import QualityGate

# Quantize
config = EdgeQuantInt8Config(device="cpu")
backend = QuantizedBackend.from_policy(policy, config)

# Quality gate
gate = QualityGate(min_cosine_similarity=0.999)
result = gate.check(original, quantized, dummy_input)

# Tracking
from lerobot_edge.tracking import ExperimentTracker, TrackConfig
tracker = ExperimentTracker(config=TrackConfig(project="my-project"))
tracker.init_run("run-1")
tracker.log_benchmark_result(result_dict)
tracker.finish_run()
```

## How It Works

Config classes register with LeRobot's `draccus.ChoiceRegistry` via `@PreTrainedConfig.register_subclass("edge_*")`. Zero changes to LeRobot source required.

## License

Apache-2.0 — see [LICENSE](LICENSE).
