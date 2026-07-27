"""Benchmark quantization speedup on large transformer models (1B+ params).

Tests FP32, FP16, bitsandbytes INT8, and NF4 on memory-bandwidth-bound
models where quantization actually helps.

Usage:
    python -m benchmarks.bench_large --device cpu --config 1b
    python -m benchmarks.bench_large --device cuda --config 2b
    python -m benchmarks.bench_large --device cpu --config custom --layers 32 --dim 2048
"""

from __future__ import annotations

import copy
import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "500m": {"layers": 12, "dim": 2048},
    "1b": {"layers": 32, "dim": 2048},
    "2b": {"layers": 40, "dim": 2560},
    "3b": {"layers": 32, "dim": 3072},
    "7b": {"layers": 32, "dim": 4096},
}


class FFNBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, dim * 4)
        self.ff2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff2(torch.nn.functional.gelu(self.ff1(self.norm(x))))


class TransformerModel(nn.Module):
    def __init__(self, layers: int = 24, dim: int = 2048) -> None:
        super().__init__()
        self.embed = nn.Linear(7, dim)
        self.layers = nn.ModuleList([FFNBlock(dim) for _ in range(layers)])
        self.head = nn.Linear(dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        for val in batch.values():
            if isinstance(val, torch.Tensor) and val.dim() == 2:
                return self.forward(val)
        return self.forward(next(iter(batch.values())))

    def reset(self) -> None:
        return


@dataclass
class BenchResult:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    fps: float
    memory_mb: float


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_memory_mb(model: nn.Module) -> float:
    total = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffers = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (total + buffers) / (1024 * 1024)


def bench_latency(fn, dummy_input: dict, warmup: int = 20, num_runs: int = 200) -> BenchResult:
    for _ in range(warmup):
        fn(dummy_input)
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            fn(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
    arr = np.array(latencies)
    return BenchResult(
        mean_ms=float(np.mean(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        fps=1000.0 / float(np.mean(arr)) if np.mean(arr) > 0 else 0,
        memory_mb=0.0,
    )


def quantize_bnb_int8(model: nn.Module) -> nn.Module:
    from bitsandbytes.nn import Int8Params, Linear8bitLt

    quantized = copy.deepcopy(model)
    linear_layers = [
        (name, module)
        for name, module in quantized.named_modules()
        if isinstance(module, nn.Linear)
    ]
    modules_dict = dict(quantized.named_modules())

    for name, module in linear_layers:
        parent_name, _, child_name = name.rpartition(".")
        parent = modules_dict[parent_name] if parent_name else quantized
        new_layer = Linear8bitLt(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            has_fp16_weights=False,
            threshold=6.0,
        )
        new_layer.weight = Int8Params(module.weight.data.half(), requires_grad=False)
        if module.bias is not None:
            new_layer.bias = nn.Parameter(module.bias.half(), requires_grad=False)
        setattr(parent, child_name, new_layer)

    return quantized


def quantize_nf4(model: nn.Module) -> nn.Module:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Params4bit

    quantized = copy.deepcopy(model)
    linear_layers = [
        (name, module)
        for name, module in quantized.named_modules()
        if isinstance(module, nn.Linear)
    ]
    modules_dict = dict(quantized.named_modules())

    for name, module in linear_layers:
        parent_name, _, child_name = name.rpartition(".")
        parent = modules_dict[parent_name] if parent_name else quantized
        new_layer = Linear4bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            compute_dtype=module.weight.dtype,
            quant_type="nf4",
        )
        w4, state = bnb.functional.quantize_4bit(module.weight.data, quant_type="nf4")
        new_layer.weight = Params4bit(w4, requires_grad=False, quant_type="nf4", quant_state=state)
        if module.bias is not None:
            new_layer.bias = nn.Parameter(module.bias.data.clone(), requires_grad=False)
        setattr(parent, child_name, new_layer)

    return quantized


def run_benchmark(
    device: str = "cpu",
    layers: int = 24,
    dim: int = 2048,
    config_name: str = "custom",
) -> dict:
    dev = torch.device(device)
    logger.info("Building model: %s (layers=%d, dim=%d) on %s", config_name, layers, dim, device)

    model = TransformerModel(layers=layers, dim=dim).to(dev).eval()
    num_params = count_params(model)
    fp32_mem = measure_memory_mb(model)
    logger.info(
        "Model: %d params (%.1fM), FP32 memory: %.1f MB",
        num_params,
        num_params / 1e6,
        fp32_mem,
    )

    dummy = {"observation.state": torch.randn(1, 7, device=dev)}
    warmup = 30 if device == "cuda" else 10
    num_runs = 300 if device == "cuda" else 200

    results: dict = {
        "config": config_name,
        "device": device,
        "model_params": num_params,
        "fp32_memory_mb": fp32_mem,
        "layers": layers,
        "dim": dim,
    }

    def fp32_fn(batch: dict) -> torch.Tensor:
        with torch.no_grad():
            return model(batch["observation.state"])  # noqa: F821

    logger.info("Benchmarking FP32...")
    fp32 = bench_latency(fp32_fn, dummy, warmup, num_runs)
    fp32.memory_mb = fp32_mem
    results["fp32"] = {
        "mean_ms": fp32.mean_ms,
        "p50_ms": fp32.p50_ms,
        "p95_ms": fp32.p95_ms,
        "fps": fp32.fps,
    }
    logger.info("FP32: %.2f ms (%.0f fps)", fp32.mean_ms, fp32.fps)

    logger.info("Benchmarking FP16...")
    fp16_model = copy.deepcopy(model).half().to(dev).eval()
    fp16_mem = measure_memory_mb(fp16_model)
    results["fp16_memory_mb"] = fp16_mem

    def fp16_fn(batch: dict) -> torch.Tensor:
        with torch.no_grad():
            x = batch["observation.state"]
            if x.dtype != torch.float16:
                x = x.half()
            return fp16_model(x)  # noqa: F821

    fp16 = bench_latency(fp16_fn, dummy, warmup, num_runs)
    fp16.memory_mb = fp16_mem
    results["fp16"] = {
        "mean_ms": fp16.mean_ms,
        "p50_ms": fp16.p50_ms,
        "p95_ms": fp16.p95_ms,
        "fps": fp16.fps,
    }
    logger.info("FP16: %.2f ms (%.0f fps), memory: %.1f MB", fp16.mean_ms, fp16.fps, fp16_mem)

    logger.info("Quantizing to bnb INT8...")
    int8_model = quantize_bnb_int8(model).to(dev).eval()
    int8_mem = measure_memory_mb(int8_model)
    results["int8_memory_mb"] = int8_mem

    def int8_fn(batch: dict) -> torch.Tensor:
        with torch.no_grad():
            return int8_model(batch["observation.state"])  # noqa: F821

    int8 = bench_latency(int8_fn, dummy, warmup, num_runs)
    int8.memory_mb = int8_mem
    results["int8"] = {
        "mean_ms": int8.mean_ms,
        "p50_ms": int8.p50_ms,
        "p95_ms": int8.p95_ms,
        "fps": int8.fps,
    }
    logger.info("INT8: %.2f ms (%.0f fps), memory: %.1f MB", int8.mean_ms, int8.fps, int8_mem)

    try:
        logger.info("Quantizing to NF4...")
        nf4_model = quantize_nf4(model).to(dev).eval()
        nf4_mem = measure_memory_mb(nf4_model)
        results["nf4_memory_mb"] = nf4_mem

        def nf4_fn(batch: dict) -> torch.Tensor:
            with torch.no_grad():
                return nf4_model(batch["observation.state"])

        nf4 = bench_latency(nf4_fn, dummy, warmup, num_runs)
        nf4.memory_mb = nf4_mem
        results["nf4"] = {
            "mean_ms": nf4.mean_ms,
            "p50_ms": nf4.p50_ms,
            "p95_ms": nf4.p95_ms,
            "fps": nf4.fps,
        }
        logger.info("NF4: %.2f ms (%.0f fps), memory: %.1f MB", nf4.mean_ms, nf4.fps, nf4_mem)
    except Exception as e:
        logger.warning("NF4 quantization failed: %s", e)

    del model, fp16_model, int8_model  # noqa: F821
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


def print_results(results: dict) -> None:
    print("\n" + "=" * 80)
    print(f"MODEL BENCHMARK — {results['device'].upper()}")
    print(
        f"Config: {results['config']} | Params: {results['model_params'] / 1e6:.0f}M | "
        f"Layers: {results['layers']} | Dim: {results['dim']}"
    )
    print("=" * 80)
    print(f"{'Backend':<25} {'Latency (ms)':<15} {'FPS':<10} {'Memory (MB)':<12} {'Speedup':<10}")
    print("-" * 80)

    fp32_ms = results["fp32"]["mean_ms"]
    fp32_mem = results["fp32_memory_mb"]

    print(
        f"{'FP32 (baseline)':<25} {fp32_ms:>8.2f}      "
        f"{results['fp32']['fps']:>6.0f}    {fp32_mem:>8.1f}     {'1.00x':>8}"
    )

    for key, label in [
        ("fp16", "FP16"),
        ("int8", "INT8 (bitsandbytes)"),
        ("nf4", "NF4 (bitsandbytes)"),
    ]:
        if key in results:
            ms = results[key]["mean_ms"]
            fps = results[key]["fps"]
            mem = results[f"{key}_memory_mb"]
            speedup = fp32_ms / ms if ms > 0 else 0
            print(f"{label:<25} {ms:>8.2f}      {fps:>6.0f}    {mem:>8.1f}     {speedup:>6.2f}x")

    print("=" * 80)

    print("\nANALYSIS:")
    if "fp16" in results:
        fp16_speedup = fp32_ms / results["fp16"]["mean_ms"]
        fp16_mem_ratio = results["fp16_memory_mb"] / fp32_mem
        print(f"  FP16:  {fp16_speedup:.2f}x latency, {fp16_mem_ratio:.2f}x memory")
    if "int8" in results:
        int8_speedup = fp32_ms / results["int8"]["mean_ms"]
        int8_mem_ratio = results["int8_memory_mb"] / fp32_mem
        print(f"  INT8:  {int8_speedup:.2f}x latency, {int8_mem_ratio:.2f}x memory")
    if "nf4" in results:
        nf4_speedup = fp32_ms / results["nf4"]["mean_ms"]
        nf4_mem_ratio = results["nf4_memory_mb"] / fp32_mem
        print(f"  NF4:   {nf4_speedup:.2f}x latency, {nf4_mem_ratio:.2f}x memory")

    print("\nNOTES:")
    print("  - FP16 halves memory bandwidth pressure, giving real speedup.")
    print("  - INT8/NF4 reduce memory further but add kernel overhead.")
    print("  - Speedup appears on larger models where memory bandwidth is the bottleneck.")
    print("  - For batch=1 on CPU, kernel launch overhead dominates on small models.\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark quantization on large models")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--config",
        default="1b",
        choices=list(MODEL_CONFIGS.keys()) + ["custom"],
    )
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--output", default="benchmark_results/large_benchmark.json")
    args = parser.parse_args()

    if args.config == "custom":
        if args.layers is None or args.dim is None:
            parser.error("--config custom requires --layers and --dim")
        layers, dim = args.layers, args.dim
    else:
        cfg = MODEL_CONFIGS[args.config]
        layers, dim = cfg["layers"], cfg["dim"]

    results = run_benchmark(args.device, layers, dim, args.config)
    print_results(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
