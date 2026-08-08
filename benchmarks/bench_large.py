"""Benchmark quantization on large synthetic models across batch sizes."""

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
    throughput: float  # samples/sec = batch_size / latency_sec
    memory_mb: float


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_memory_mb(model: nn.Module) -> float:
    total = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffers = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (total + buffers) / (1024 * 1024)


def bench_latency(
    fn, dummy_input: dict, warmup: int = 20, num_runs: int = 200, batch_size: int = 1
) -> BenchResult:
    for _ in range(warmup):
        fn(dummy_input)
    latencies = []
    is_cuda = torch.cuda.is_available()
    with torch.no_grad():
        for _ in range(num_runs):
            if is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            fn(dummy_input)
            if is_cuda:
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
    arr = np.array(latencies)
    mean_ms = float(np.mean(arr))
    throughput = (batch_size / (mean_ms / 1000.0)) if mean_ms > 0 else 0
    return BenchResult(
        mean_ms=mean_ms,
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        fps=1000.0 / mean_ms if mean_ms > 0 else 0,
        throughput=throughput,
        memory_mb=0.0,
    )


def quantize_bnb_int8(model: nn.Module, min_out_features: int = 16) -> nn.Module:
    from bitsandbytes.nn import Int8Params, Linear8bitLt

    quantized = copy.deepcopy(model)
    linear_layers = [
        (name, module)
        for name, module in quantized.named_modules()
        if isinstance(module, nn.Linear)
    ]
    modules_dict = dict(quantized.named_modules())
    skipped = 0

    for name, module in linear_layers:
        if module.out_features < min_out_features:
            skipped += 1
            continue
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

    if skipped:
        logger.info("INT8: skipped %d layers with out_features < %d", skipped, min_out_features)
    return quantized


def quantize_nf4(model: nn.Module, device: torch.device | None = None) -> nn.Module:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Params4bit

    quantized = copy.deepcopy(model)
    if device is not None:
        quantized = quantized.to(device)
    linear_layers = [
        (name, module)
        for name, module in quantized.named_modules()
        if isinstance(module, nn.Linear)
    ]
    modules_dict = dict(quantized.named_modules())
    compute_dtype = torch.float16 if (device and device.type == "cuda") else torch.float32

    for name, module in linear_layers:
        parent_name, _, child_name = name.rpartition(".")
        parent = modules_dict[parent_name] if parent_name else quantized
        w4, state = bnb.functional.quantize_4bit(module.weight.data.float(), quant_type="nf4")
        new_layer = Linear4bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            compute_dtype=compute_dtype,
            quant_type="nf4",
        )
        new_layer.weight = Params4bit(w4, requires_grad=False, quant_type="nf4", quant_state=state)
        if module.bias is not None:
            new_layer.bias = nn.Parameter(module.bias.data.clone(), requires_grad=False)
        setattr(parent, child_name, new_layer)

    return quantized


def maybe_compile(model: nn.Module, use_compile: bool) -> nn.Module:
    if use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Applied torch.compile with reduce-overhead mode")
        except Exception as e:
            logger.warning("torch.compile failed: %s", e)
    return model


def run_benchmark(
    device: str = "cpu",
    layers: int = 24,
    dim: int = 2048,
    config_name: str = "custom",
    batch_sizes: list[int] | None = None,
    use_compile: bool = False,
) -> dict:
    dev = torch.device(device)
    if batch_sizes is None:
        batch_sizes = [1, 4, 16, 64] if device == "cuda" else [1, 4, 16]

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

    warmup = 30 if device == "cuda" else 10
    num_runs = 300 if device == "cuda" else 200

    results: dict = {
        "config": config_name,
        "device": device,
        "model_params": num_params,
        "fp32_memory_mb": fp32_mem,
        "layers": layers,
        "dim": dim,
        "batch_sizes": batch_sizes,
        "compiled": use_compile,
    }

    # --- FP32 baseline ---
    fp32_compiled = maybe_compile(model, use_compile)
    fp32_results = {}
    for bs in batch_sizes:
        dummy = {"observation.state": torch.randn(bs, 7, device=dev)}

        def fp32_fn(batch: dict, _m: nn.Module = fp32_compiled) -> torch.Tensor:
            with torch.no_grad():
                return _m(batch["observation.state"])

        r = bench_latency(fp32_fn, dummy, warmup, num_runs, bs)
        r.memory_mb = fp32_mem
        fp32_results[bs] = r
        logger.info("FP32  bs=%-3d: %.2f ms  (%6.0f samples/s)", bs, r.mean_ms, r.throughput)

    results["fp32"] = {
        str(bs): {"mean_ms": r.mean_ms, "p50_ms": r.p50_ms, "throughput": r.throughput}
        for bs, r in fp32_results.items()
    }

    # --- FP16 ---
    fp16_model = maybe_compile(copy.deepcopy(model).half().to(dev).eval(), use_compile)
    fp16_mem = measure_memory_mb(fp16_model)
    results["fp16_memory_mb"] = fp16_mem

    fp16_results = {}
    for bs in batch_sizes:
        dummy = {"observation.state": torch.randn(bs, 7, device=dev)}

        def fp16_fn(batch: dict, _m: nn.Module = fp16_model) -> torch.Tensor:
            with torch.no_grad():
                x = batch["observation.state"]
                if x.dtype != torch.float16:
                    x = x.half()
                return _m(x)

        r = bench_latency(fp16_fn, dummy, warmup, num_runs, bs)
        r.memory_mb = fp16_mem
        fp16_results[bs] = r
        logger.info("FP16  bs=%-3d: %.2f ms  (%6.0f samples/s)", bs, r.mean_ms, r.throughput)

    results["fp16"] = {
        str(bs): {"mean_ms": r.mean_ms, "p50_ms": r.p50_ms, "throughput": r.throughput}
        for bs, r in fp16_results.items()
    }

    # --- INT8 ---
    int8_results = {}
    int8_model = None
    try:
        int8_model = quantize_bnb_int8(model).to(dev).eval()
        int8_model = maybe_compile(int8_model, use_compile)
        int8_mem = measure_memory_mb(int8_model)
        results["int8_memory_mb"] = int8_mem

        for bs in batch_sizes:
            dummy = {"observation.state": torch.randn(bs, 7, device=dev)}

            def int8_fn(batch: dict, _m: nn.Module = int8_model) -> torch.Tensor:
                with torch.no_grad():
                    return _m(batch["observation.state"])

            try:
                r = bench_latency(int8_fn, dummy, warmup, num_runs, bs)
                r.memory_mb = int8_mem
                int8_results[bs] = r
                logger.info(
                    "INT8  bs=%-3d: %.2f ms  (%6.0f samples/s)", bs, r.mean_ms, r.throughput
                )
            except Exception as e:
                logger.warning("INT8 failed at bs=%d: %s", bs, e)

        if int8_results:
            results["int8"] = {
                str(bs): {"mean_ms": r.mean_ms, "p50_ms": r.p50_ms, "throughput": r.throughput}
                for bs, r in int8_results.items()
            }
    except Exception as e:
        logger.warning("INT8 quantization failed: %s", e)

    # --- NF4 ---
    nf4_results = {}
    nf4_model = None
    try:
        nf4_model = quantize_nf4(model, device=dev)
        nf4_model = maybe_compile(nf4_model.eval(), use_compile)
        nf4_mem = measure_memory_mb(nf4_model)
        results["nf4_memory_mb"] = nf4_mem

        for bs in batch_sizes:
            dummy = {"observation.state": torch.randn(bs, 7, device=dev)}

            def nf4_fn(batch: dict, _m: nn.Module = nf4_model) -> torch.Tensor:
                with torch.no_grad():
                    return _m(batch["observation.state"])

            try:
                r = bench_latency(nf4_fn, dummy, warmup, num_runs, bs)
                r.memory_mb = nf4_mem
                nf4_results[bs] = r
                logger.info(
                    "NF4   bs=%-3d: %.2f ms  (%6.0f samples/s)", bs, r.mean_ms, r.throughput
                )
            except Exception as e:
                logger.warning("NF4 failed at bs=%d: %s", bs, e)

        if nf4_results:
            results["nf4"] = {
                str(bs): {"mean_ms": r.mean_ms, "p50_ms": r.p50_ms, "throughput": r.throughput}
                for bs, r in nf4_results.items()
            }
    except Exception as e:
        logger.warning("NF4 failed: %s", e)

    del model, fp16_model, fp32_compiled
    if int8_model is not None:
        del int8_model
    if nf4_model is not None:
        del nf4_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


def print_results(results: dict) -> None:
    device = results["device"].upper()
    config = results["config"]
    params_m = results["model_params"] / 1e6
    compiled = " (torch.compile)" if results.get("compiled") else ""
    batch_sizes = results["batch_sizes"]

    print(f"\n{'=' * 90}")
    print(f"BENCHMARK — {device}{compiled}")
    print(
        f"Config: {config} | Params: {params_m:.0f}M | Layers: {results['layers']} | Dim: {results['dim']}"
    )
    print(f"{'=' * 90}")

    header = f"{'Backend':<22}"
    for bs in batch_sizes:
        header += f"  {'bs=' + str(bs):>18}"
    print(header)
    print("-" * 90)

    fp32_mem = results["fp32_memory_mb"]

    # FP32 row
    row = f"{'FP32':<22}"
    for bs in batch_sizes:
        r = results["fp32"][str(bs)]
        row += f"  {r['mean_ms']:>7.1f}ms/{r['throughput']:>6.0f}sp"
    print(row)

    # Quantized rows
    for key, label, _mem_key in [
        ("fp16", "FP16", "fp16_memory_mb"),
        ("int8", "INT8", "int8_memory_mb"),
        ("nf4", "NF4", "nf4_memory_mb"),
    ]:
        if key not in results:
            continue
        row = f"{label:<22}"
        for bs in batch_sizes:
            r = results[key][str(bs)]
            fp32_t = results["fp32"][str(bs)]["throughput"]
            speedup = r["throughput"] / fp32_t if fp32_t > 0 else 0
            row += f"  {r['mean_ms']:>7.1f}ms/{speedup:>5.2f}x"
        print(row)

    # Memory row
    mem_row = f"{'Memory':<22}"
    for _ in batch_sizes:
        mem_row += f"  {'':>18}"
    print()
    print(f"  FP32 memory: {fp32_mem:.1f} MB")
    if "fp16_memory_mb" in results:
        print(
            f"  FP16 memory: {results['fp16_memory_mb']:.1f} MB ({results['fp16_memory_mb'] / fp32_mem:.2f}x)"
        )
    if "int8_memory_mb" in results:
        print(
            f"  INT8 memory: {results['int8_memory_mb']:.1f} MB ({results['int8_memory_mb'] / fp32_mem:.2f}x)"
        )
    if "nf4_memory_mb" in results:
        print(
            f"  NF4  memory: {results['nf4_memory_mb']:.1f} MB ({results['nf4_memory_mb'] / fp32_mem:.2f}x)"
        )

    print(f"\n{'=' * 90}")
    print("INTERPRETATION:")
    print("  Throughput (sp) = batch_size / latency_sec. Higher = better.")
    print("  At small batch, kernel launch overhead dominates — FP16 often wins.")
    print("  At large batch, reduced memory bandwidth from INT8/NF4 shows real speedup.")
    print("  torch.compile fuses dequant kernels, giving INT8/NF4 a real chance.")
    print()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark quantization across batch sizes")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--config", default="1b", choices=list(MODEL_CONFIGS.keys()) + ["custom"])
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--batch-sizes", type=str, default=None, help="Comma-separated batch sizes")
    parser.add_argument("--output", default="benchmark_results/large_benchmark.json")
    args = parser.parse_args()

    if args.config == "custom":
        if args.layers is None or args.dim is None:
            parser.error("--config custom requires --layers and --dim")
        layers, dim = args.layers, args.dim
    else:
        cfg = MODEL_CONFIGS[args.config]
        layers, dim = cfg["layers"], cfg["dim"]

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")] if args.batch_sizes else None

    results = run_benchmark(args.device, layers, dim, args.config, batch_sizes, args.compile)
    print_results(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
