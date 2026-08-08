# LeRobot Edge

Quantization, export, and benchmarking for deploying [LeRobot](https://github.com/huggingface/lerobot) policies on edge GPUs. Zero source modifications — install and change your policy type.

## Install

```bash
pip install lerobot-edge                    # core (torchao INT8/INT4)
pip install "lerobot-edge[quantize]"        # bitsandbytes NF4/FP4
pip install "lerobot-edge[onnx]"            # ONNX export
pip install "lerobot-edge[wandb]"           # experiment tracking
pip install "lerobot-edge[all]"             # everything
```

## Quickstart

```bash
# Evaluate quantized policy
lerobot-eval --policy.type=edge_quant_int4 --policy.pretrained_path=lerobot/smolvla_base

# Quantize a checkpoint
lerobot-edge-quantize --source lerobot/smolvla_base --output ./quantized --method int4

# Benchmark all backends
python -m benchmarks.bench_smolvla --batch-sizes 1,4

# Systematic A/B experiment grid
lerobot-edge-experiment --checkpoint lerobot/smolvla_base --methods fp32 int4

# Regression dashboard
lerobot-edge-regression --dirs benchmark_results
```

## Variants

| Variant | Method | Library |
|---------|--------|---------|
| `edge_identity` | FP32 passthrough | core |
| `edge_quant_int8` | Dynamic INT8 | torchao |
| `edge_quant_int4` | INT4 weight-only (recommended) | torchao |
| `edge_quant_bnb_int8` | Linear8bitLt INT8 | bitsandbytes |
| `edge_quant_bnb_nf4` | NF4 4-bit | bitsandbytes |
| `edge_quant_bnb_fp4` | FP4 4-bit | bitsandbytes |
| `edge_onnx_fp32` | ONNX Runtime FP32 | onnx |
| `edge_onnx_int8` | ONNX Runtime INT8 | onnx |

## Optimizations

Three optimizations are applied automatically at load time:

1. **SDPA/FlashAttention** — replaces eager matmul attention (2.06× speedup on SmolVLA)
2. **INT8 KV-cache** — compresses cached key/value states (3.98× reduction, lossless)
3. **torchao INT4** — weight-only quantization (4× memory reduction)

```python
from lerobot_edge.optimization import optimize_policy_for_inference

policy = optimize_policy_for_inference(policy,
    enable_attention=True,          # SDPA/FlashAttention
    enable_kv_cache_quant=True,     # INT8 KV-cache
    enable_compile=False,           # torch.compile (Linux + Triton)
)
```

## Benchmarks

SmolVLA (450M params) on RTX 5060 Laptop (7 GB), batch=1:

| Backend | Latency | Memory |
|---------|---------|--------|
| FP32 + SDPA | 8.77 ms | 1142 MB |
| FP16 + SDPA | 8.49 ms | 571 MB |
| NF4 + SDPA | 8.92 ms | 287 MB |

SDPA delivers 2.06× over eager attention (18.07→8.77 ms). NF4 gives 3.97× memory reduction.

## Development

```bash
make install-dev   # pip install -e ".[dev,all]"
make test          # fast unit tests
make lint          # ruff check
make format        # ruff format
make typecheck     # mypy
```

## License

Apache-2.0
