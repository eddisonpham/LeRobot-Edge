# LeRobot Edge

> Quantization, export, and benchmarking pipeline for deploying [HuggingFace LeRobot](https://github.com/huggingface/lerobot) policies on edge devices.

LeRobot Edge plugs into LeRobot's policy system with zero source modifications. It adds INT8/4-bit quantization, ONNX/TensorRT export, quality validation, and benchmarking — everything you need to compress and deploy VLA models on GPU and CPU hardware.



## Installation

```bash
pip install lerobot-edge
pip install "lerobot-edge[onnx]"       # ONNX export
pip install "lerobot-edge[quantize]"   # bitsandbytes INT8/4-bit
pip install "lerobot-edge[tensorrt]"   # TensorRT (GPU)
pip install "lerobot-edge[wandb]"      # experiment tracking
pip install "lerobot-edge[all]"        # everything
```

## Quickstart

```bash
# Evaluate a quantized policy on PushT
lerobot-eval \
  --policy.type=edge_quant_int8 \
  --policy.pretrained_path=lerobot/smolvla_base \
  --env.type=pusht --eval.n_episodes=10

# Benchmark all backends
python -m lerobot_edge.evaluation.compare_backends \
  --checkpoint lerobot/smolvla_base \
  --backends identity int8 bnb_int8 nf4 bnb_fp4

# Quantize and save
python -m lerobot_edge.compression.quantize \
  --source lerobot/smolvla_base \
  --output ./quantized \
  --method dynamic_int8
```

## Available Variants

| Variant | Description | Install |
|---------|-------------|---------|
| `edge_identity` | FP32 passthrough | core |
| `edge_quant_int8` | Dynamic INT8 (torchao) | core |
| `edge_quant_bnb_int8` | INT8 (bitsandbytes) | `[quantize]` |
| `edge_quant_bnb_nf4` | NF4 4-bit (bitsandbytes) | `[quantize]` |
| `edge_quant_bnb_fp4` | FP4 4-bit (bitsandbytes) | `[quantize]` |
| `edge_onnx_fp32` | ONNX Runtime FP32 | `[onnx]` |
| `edge_onnx_int8` | ONNX Runtime INT8 | `[onnx]` |
| `edge_distilled` | Teacher-student distilled | core |

## Benchmark Results

SmolVLA (450M params, 1142 MB FP32) on NVIDIA RTX 5060 (8.5 GB VRAM).

### Latency

| Backend | bs=1 | bs=4 |
|---------|------|------|
| FP32 (baseline) | 18.07 ms | 19.24 ms |
| FP16 autocast | 18.19 ms | 18.40 ms |
| FP16 vs FP32 | 0.99x | 1.05x |
| NF4 4-bit | -- | -- |

NF4 quantizes to 287 MB (3.97x reduction) but SmolVLA's HuggingFace attention is incompatible with bitsandbytes Linear4bit dequantized outputs.

FP16 autocast doesn't accelerate SmolVLA at bs=1 because the model is compute-bound at 450M params. At bs=4, GPU saturation begins and FP16 shows a modest 1.05x speedup. Larger models (1B+) benefit more from reduced-precision compute.

### Memory

| Format | Size | Reduction |
|--------|------|-----------|
| FP32 | 1142 MB | 1x |
| NF4 4-bit | 287 MB | 3.97x |
| FP16 (half) | 571 MB | 2x |

## Quantization Methods

| Method | Library | Bit Width | Best For |
|--------|---------|-----------|----------|
| Dynamic INT8 | torchao | 8-bit | GPU with CUDA |
| Linear8bitLt | bitsandbytes | 8-bit | GPU with CUDA |
| NF4 | bitsandbytes | 4-bit | Memory-constrained GPU |
| FP4 | bitsandbytes | 4-bit | Memory-constrained GPU |
| Static INT8 | torchao | 8-bit | Disk space (needs calibration) |

## How It Works

Config classes register with LeRobot's `draccus.ChoiceRegistry` via `@PreTrainedConfig.register_subclass("edge_*")`. This means zero changes to LeRobot source — install `lerobot-edge` and change the policy type in your CLI or config.

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
```

## Development

```bash
make install-dev   # pip install -e ".[dev,all]"
make test          # non-slow tests
make lint          # ruff check
make format        # ruff format
make ci            # lint + test
```

## License

Apache-2.0
