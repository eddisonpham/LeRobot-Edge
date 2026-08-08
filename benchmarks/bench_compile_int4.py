"""Verify torchao int4 + torch.compile kernel fusion delivers real speedups.

Systematic 2×3 matrix:
  - Quantization: FP32 (baseline), INT4 (torchao weight-only)
  - Compile mode: None, reduce-overhead, max-autotune

Tests both on synthetic transformer models (configurable sizes) and
optionally on real SmolVLA checkpoints.

Usage:
    python -m benchmarks.bench_compile_int4
    python -m benchmarks.bench_compile_int4 --device cuda --config 1b
    python -m benchmarks.bench_compile_int4 --real-smolvla
    python -m benchmarks.bench_compile_int4 --batch-sizes 1,4,16
"""

from __future__ import annotations

import copy
import gc
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configs (synthetic)
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict] = {
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

    def select_action(self, batch: dict) -> torch.Tensor:
        for val in batch.values():
            if isinstance(val, torch.Tensor) and val.dim() == 2:
                return self.forward(val)
        return self.forward(next(iter(batch.values())))

    def reset(self) -> None:
        return


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


@dataclass
class CompileInt4Result:
    config_name: str
    quantization: str  # fp32 or int4
    compile_mode: str | None  # None, reduce-overhead, max-autotune
    batch_size: int
    num_params: int
    model_memory_mb: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    throughput_fps: float
    peak_memory_mb: float
    compile_time_s: float
    warmup_runs: int
    bench_runs: int
    device: str
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict:
        return {
            "config": self.config_name,
            "quantization": self.quantization,
            "compile_mode": self.compile_mode,
            "batch_size": self.batch_size,
            "num_params": self.num_params,
            "model_memory_mb": round(self.model_memory_mb, 2),
            "latency_mean_ms": round(self.latency_mean_ms, 3),
            "latency_p50_ms": round(self.latency_p50_ms, 3),
            "latency_p95_ms": round(self.latency_p95_ms, 3),
            "throughput_fps": round(self.throughput_fps, 1),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "compile_time_s": round(self.compile_time_s, 2),
            "warmup_runs": self.warmup_runs,
            "bench_runs": self.bench_runs,
            "device": self.device,
            "timestamp": self.timestamp,
        }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_memory_mb(model: nn.Module) -> float:
    total = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffers = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (total + buffers) / (1024 * 1024)


def bench_latency(
    fn,
    dummy_input: dict,
    warmup: int = 30,
    num_runs: int = 200,
    batch_size: int = 1,
    is_cuda: bool = False,
) -> tuple[float, float, float, float, float]:
    """Benchmark a callable. Returns (mean_ms, p50_ms, p95_ms, fps, peak_mem_mb)."""
    for _ in range(warmup):
        fn(dummy_input)

    latencies: list[float] = []
    peak_mem = 0.0

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for _ in range(num_runs):
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn(dummy_input)
        if is_cuda:
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
        if is_cuda:
            mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            if mem > peak_mem:
                peak_mem = mem

    arr = np.array(latencies)
    mean_ms = float(np.mean(arr))
    fps = (batch_size / (mean_ms / 1000.0)) if mean_ms > 0 else 0.0
    return (
        mean_ms,
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        fps,
        peak_mem,
    )


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_compile_int4_benchmark(
    device: str = "cpu",
    config_name: str = "500m",
    layers: int | None = None,
    dim: int | None = None,
    batch_sizes: list[int] | None = None,
    use_smolvla: bool = False,
    force_compile: bool = False,
) -> list[CompileInt4Result]:
    """Run the 2×3 compile+int4 matrix benchmark.

    Returns:
        List of CompileInt4Result for every combination.
    """
    dev = torch.device(device)
    is_cuda = dev.type == "cuda"

    if batch_sizes is None:
        batch_sizes = [1, 4, 16] if is_cuda else [1, 4]

    # --- Build model ---
    if use_smolvla:
        try:
            from benchmarks.bench_smolvla import load_smolvla

            model_fp32, _ = load_smolvla()
            model_fp32 = model_fp32.to(dev).eval()
            config_name = "smolvla"
        except Exception as e:
            logger.error("Failed to load SmolVLA: %s. Falling back to synthetic.", e)
            use_smolvla = False

    if not use_smolvla:
        if layers is None or dim is None:
            cfg = MODEL_CONFIGS.get(config_name, MODEL_CONFIGS["500m"])
            layers = cfg["layers"]
            dim = cfg["dim"]
        model_fp32 = TransformerModel(layers=layers, dim=dim).to(dev).eval()

    num_params = count_params(model_fp32)
    fp32_mem = measure_memory_mb(model_fp32)
    logger.info(
        "Model: %s, %.1fM params, FP32 memory: %.1f MB",
        config_name,
        num_params / 1e6,
        fp32_mem,
    )

    # --- Build INT4 model ---
    from lerobot_edge.compression.quantize import HAS_TORCHAO, quantize_int4_weight_only

    model_int4 = None
    int4_mem = 0.0
    if HAS_TORCHAO:
        model_int4 = copy.deepcopy(model_fp32).to(dev).eval()
        t0 = time.perf_counter()
        model_int4 = quantize_int4_weight_only(model_int4, group_size=32)
        quantize_time = time.perf_counter() - t0
        int4_mem = measure_memory_mb(model_int4)
        logger.info(
            "INT4 memory: %.1f MB (%.2fx reduction), quantize time: %.2fs",
            int4_mem,
            fp32_mem / int4_mem if int4_mem > 0 else 0,
            quantize_time,
        )
    else:
        logger.warning(
            "torchao not installed — skipping INT4 benchmarks. "
            "Install with: pip install torchao>=0.17.0"
        )

    # --- Compile mode matrix ---
    compile_modes: list[str | None] = [None]
    if is_cuda or force_compile:
        if hasattr(torch, "compile"):
            compile_modes.extend(["reduce-overhead", "max-autotune"])

    # Use higher warmup/runs for GPU
    warmup = 30 if is_cuda else 10
    num_runs = 300 if is_cuda else 100

    results: list[CompileInt4Result] = []

    models_to_bench: list[tuple[str, nn.Module | None, float]] = [
        ("fp32", model_fp32, fp32_mem),
    ]
    if model_int4 is not None:
        models_to_bench.append(("int4", model_int4, int4_mem))

    for quant_name, model_base, base_mem in models_to_bench:
        for cmode in compile_modes:
            # Build model variant
            model = copy.deepcopy(model_base).to(dev).eval()
            compile_s = 0.0

            if cmode:
                t0 = time.perf_counter()
                try:
                    model = torch.compile(model, mode=cmode)
                    # Warm up to trigger compilation
                    dummy = {"observation.state": torch.randn(1, 7, device=dev)}
                    with torch.no_grad():
                        model(dummy["observation.state"])
                    if is_cuda:
                        torch.cuda.synchronize()
                    compile_s = time.perf_counter() - t0
                    logger.info(
                        "Compiled %s (mode=%s) in %.2fs",
                        quant_name,
                        cmode,
                        compile_s,
                    )
                except Exception as e:
                    logger.warning(
                        "torch.compile failed for %s (mode=%s): %s. Skipping.",
                        quant_name,
                        cmode,
                        e,
                    )
                    continue

            for bs in batch_sizes:
                dummy = {"observation.state": torch.randn(bs, 7, device=dev)}

                def make_fn(m: nn.Module) -> Any:
                    def fn(batch: dict) -> torch.Tensor:
                        with torch.no_grad():
                            return m(batch["observation.state"])

                    return fn

                fn = make_fn(model)

                # Extra warmup for compiled models
                extra_warmup = warmup + (10 if cmode else 0)
                mean_ms, p50, p95, fps, peak_mem = bench_latency(
                    fn, dummy, extra_warmup, num_runs, bs, is_cuda
                )

                logger.info(
                    "%s + compile=%s @ bs=%d: %.2f ms (%.0f fps, %.1f MB peak)",
                    quant_name,
                    cmode or "none",
                    bs,
                    mean_ms,
                    fps,
                    peak_mem,
                )

                results.append(
                    CompileInt4Result(
                        config_name=config_name,
                        quantization=quant_name,
                        compile_mode=cmode,
                        batch_size=bs,
                        num_params=num_params,
                        model_memory_mb=base_mem,
                        latency_mean_ms=mean_ms,
                        latency_p50_ms=p50,
                        latency_p95_ms=p95,
                        throughput_fps=fps,
                        peak_memory_mb=peak_mem,
                        compile_time_s=compile_s,
                        warmup_runs=warmup,
                        bench_runs=num_runs,
                        device=device,
                    )
                )

            del model
            gc.collect()
            if is_cuda:
                torch.cuda.empty_cache()

    del model_fp32
    if model_int4 is not None:
        del model_int4
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_results_table(results: list[CompileInt4Result]) -> None:
    """Print a formatted comparison table grouped by batch size."""
    if not results:
        print("No results.")
        return

    config = results[0].config_name
    device = results[0].device.upper()

    print(f"\n{'=' * 95}")
    print(f"  torch.compile + INT4 KERNEL FUSION BENCHMARK")
    print(f"  Model: {config}  |  Device: {device}")
    print(f"{'=' * 95}")

    for bs in sorted(set(r.batch_size for r in results)):
        subset = [r for r in results if r.batch_size == bs]
        subset.sort(key=lambda r: r.latency_mean_ms)

        print(f"\n  Batch Size = {bs}")
        print(f"  {'Quant':<6} {'Compile':<18} {'Mean(ms)':<10} {'P50(ms)':<10} {'P95(ms)':<10} {'FPS':<8} {'Mem(MB)':<10} {'Peak(MB)':<10} {'Compile(s)':<10}")
        print(f"  {'-' * 90}")

        baseline_ms = None
        for r in subset:
            if r.quantization == "fp32" and r.compile_mode is None:
                baseline_ms = r.latency_mean_ms
                break

        for r in subset:
            cmode = r.compile_mode or "none"
            marker = ""
            if baseline_ms and r.latency_mean_ms > 0:
                speedup = baseline_ms / r.latency_mean_ms
                if speedup > 1.5:
                    marker = " 🔥"
                elif speedup > 1.2:
                    marker = " ✅"
                elif speedup < 0.95:
                    marker = " ⚠️"

            print(
                f"  {r.quantization:<6} {cmode:<18} "
                f"{r.latency_mean_ms:>8.2f}  {r.latency_p50_ms:>8.2f}  "
                f"{r.latency_p95_ms:>8.2f}  {r.throughput_fps:>6.0f}  "
                f"{r.model_memory_mb:>8.1f}  {r.peak_memory_mb:>8.1f}  "
                f"{r.compile_time_s:>8.2f}{marker}"
            )

    print(f"\n{'=' * 95}")

    # Summary: best config per batch size
    print("\n  BEST PER BATCH SIZE:")
    for bs in sorted(set(r.batch_size for r in results)):
        subset = [r for r in results if r.batch_size == bs]
        best = min(subset, key=lambda r: r.latency_mean_ms)
        fp32_base = [r for r in subset if r.quantization == "fp32" and r.compile_mode is None]
        baseline = fp32_base[0] if fp32_base else best
        speedup = baseline.latency_mean_ms / best.latency_mean_ms if best.latency_mean_ms > 0 else 0
        print(
            f"    bs={bs}: {best.quantization}+compile={best.compile_mode or 'none'} "
            f"-> {best.latency_mean_ms:.2f} ms "
            f"({speedup:.2f}x vs FP32)"
        )

    # Memory summary
    fp32_results = [r for r in results if r.quantization == "fp32"]
    if fp32_results:
        fp32_mem = fp32_results[0].model_memory_mb
        int4_results = [r for r in results if r.quantization == "int4"]
        if int4_results:
            int4_mem = int4_results[0].model_memory_mb
            print(f"\n  Memory: FP32={fp32_mem:.0f}MB  INT4={int4_mem:.0f}MB  "
                  f"({fp32_mem/int4_mem:.2f}x reduction)")

    print(f"{'=' * 95}\n")


def compute_speedups(results: list[CompileInt4Result]) -> dict:
    """Compute speedup matrix: BS × quant × compile."""
    speedups: dict = {}
    for bs in sorted(set(r.batch_size for r in results)):
        subset = [r for r in results if r.batch_size == bs]
        fp32_base = [r for r in subset if r.quantization == "fp32" and r.compile_mode is None]
        if not fp32_base:
            continue
        baseline_ms = fp32_base[0].latency_mean_ms
        speedups[f"bs={bs}"] = {}
        for r in subset:
            key = f"{r.quantization}"
            if r.compile_mode:
                key += f"+compile={r.compile_mode}"
            speedups[f"bs={bs}"][key] = {
                "latency_ms": r.latency_mean_ms,
                "speedup": round(baseline_ms / r.latency_mean_ms, 3) if r.latency_mean_ms > 0 else 0,
                "memory_mb": r.model_memory_mb,
            }
    return speedups


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark torch.compile + INT4 kernel fusion"
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--config",
        default="500m",
        choices=list(MODEL_CONFIGS.keys()) + ["custom"],
    )
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated batch sizes",
    )
    parser.add_argument(
        "--real-smolvla",
        action="store_true",
        help="Benchmark on real SmolVLA checkpoint (requires GPU + network)",
    )
    parser.add_argument(
        "--force-compile",
        action="store_true",
        help="Enable torch.compile even on CPU",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results/compile_int4_benchmark.json",
    )
    args = parser.parse_args()

    if args.config == "custom" and (args.layers is None or args.dim is None):
        parser.error("--config custom requires --layers and --dim")

    batch_sizes = (
        [int(x) for x in args.batch_sizes.split(",")]
        if args.batch_sizes
        else None
    )

    results = run_compile_int4_benchmark(
        device=args.device,
        config_name=args.config,
        layers=args.layers,
        dim=args.dim,
        batch_sizes=batch_sizes,
        use_smolvla=args.real_smolvla,
        force_compile=args.force_compile,
    )

    print_results_table(results)

    # Compute and print speedups
    speedups = compute_speedups(results)
    print("SPEEDUP MATRIX (vs FP32 no-compile):")
    print(json.dumps(speedups, indent=2))

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
