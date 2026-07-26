"""Benchmark harness for lerobot_edge policy variants."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot_edge.core.base import DeploymentBackend
from lerobot_edge.core.configs import EdgeBaseConfig
from lerobot_edge.core.utils import (
    build_dummy_input,
    get_git_commit_hash,
    measure_model_memory,
    measure_peak_memory_mb,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BenchmarkResult",
    "benchmark_backend",
    "benchmark_policy_variants",
    "compare_results",
    "load_results",
]


@dataclass
class BenchmarkResult:
    """Single benchmark result for one backend + device profile combination."""

    backend_name: str
    device_profile: str
    git_commit: str
    timestamp: str
    config: dict[str, Any] = field(default_factory=dict)
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_std_ms: float = 0.0
    throughput_fps: float = 0.0
    peak_memory_mb: float = 0.0
    param_memory_mb: float = 0.0
    num_parameters: int = 0
    model_size_mb: float = 0.0
    success_rate: float | None = None
    eval_episodes: int = 0
    warmup_runs: int = 0
    benchmark_runs: int = 0
    notes: str = ""


def benchmark_backend(
    backend: DeploymentBackend,
    dummy_input: dict[str, torch.Tensor],
    *,
    warmup_runs: int = 10,
    num_runs: int = 100,
    backend_name: str = "unknown",
    device_profile: str = "cpu",
    config: EdgeBaseConfig | None = None,
) -> BenchmarkResult:
    """Benchmark a deployment backend."""
    logger.info(
        "Benchmarking %s on %s: warmup=%d, runs=%d",
        backend_name,
        device_profile,
        warmup_runs,
        num_runs,
    )

    device = backend.device
    dummy_input = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dummy_input.items()
    }

    logger.debug("Running %d warmup iterations...", warmup_runs)
    for _ in range(warmup_runs):
        _ = backend.predict(dummy_input)

    latencies_ms: list[float] = []
    peak_mem = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    logger.debug("Running %d benchmark iterations...", num_runs)
    for _ in range(num_runs):
        start_time = time.perf_counter()

        with torch.no_grad():
            _ = backend.predict(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        latencies_ms.append((end_time - start_time) * 1000)

        current_mem = measure_peak_memory_mb()
        if current_mem > peak_mem:
            peak_mem = current_mem

    latencies = np.array(latencies_ms)
    latency_mean = float(np.mean(latencies))
    latency_p50 = float(np.percentile(latencies, 50))
    latency_p95 = float(np.percentile(latencies, 95))
    latency_p99 = float(np.percentile(latencies, 99))
    latency_std = float(np.std(latencies))

    throughput = 1000.0 / latency_mean if latency_mean > 0 else 0.0

    if hasattr(backend, "_policy") and hasattr(backend._policy, "parameters"):
        model_mem = measure_model_memory(backend._policy)
    else:
        model_mem = {"param_mb": 0.0, "buffer_mb": 0.0, "total_mb": 0.0, "num_parameters": 0}

    result = BenchmarkResult(
        backend_name=backend_name,
        device_profile=device_profile,
        git_commit=get_git_commit_hash(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        config=asdict(config) if config else {},
        latency_mean_ms=latency_mean,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        latency_p99_ms=latency_p99,
        latency_std_ms=latency_std,
        throughput_fps=throughput,
        peak_memory_mb=peak_mem,
        param_memory_mb=model_mem["param_mb"],
        num_parameters=model_mem["num_parameters"],
        model_size_mb=model_mem["total_mb"],
        warmup_runs=warmup_runs,
        benchmark_runs=num_runs,
    )

    logger.info(
        "Benchmark complete: latency=%.2f+/- %.2f ms (p50=%.2f, p95=%.2f), throughput=%.1f fps",
        latency_mean,
        latency_std,
        latency_p50,
        latency_p95,
        throughput,
    )

    return result


def benchmark_policy_variants(
    variants: dict[str, DeploymentBackend],
    dummy_input: dict[str, torch.Tensor],
    *,
    device_profile: str = "cpu",
    output_dir: str | Path = "benchmark_results",
    **kwargs: Any,
) -> list[BenchmarkResult]:
    """Benchmark multiple policy variants and save results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkResult] = []

    for name, backend in variants.items():
        logger.info("Benchmarking variant: %s", name)
        result = benchmark_backend(
            backend,
            dummy_input,
            backend_name=name,
            device_profile=device_profile,
            **kwargs,
        )
        results.append(result)

    _save_results_json(results, output_dir / "benchmark_results.json")
    _save_results_csv(results, output_dir / "benchmark_results.csv")

    return results


def _save_results_json(results: list[BenchmarkResult], path: Path) -> None:
    """Save benchmark results to JSON."""
    data = [asdict(r) for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Results saved to %s", path)


def _save_results_csv(results: list[BenchmarkResult], path: Path) -> None:
    """Save benchmark results to CSV."""
    if not results:
        return

    fieldnames = list(asdict(results[0]).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    logger.info("Results saved to %s", path)


def load_results(path: str | Path) -> list[BenchmarkResult]:
    """Load benchmark results from JSON."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    return [BenchmarkResult(**item) for item in data]


def compare_results(
    results: list[BenchmarkResult],
    baseline_name: str = "edge_identity",
) -> dict[str, Any]:
    """Compare benchmark results against a baseline."""
    baseline = None
    for r in results:
        if r.backend_name == baseline_name:
            baseline = r
            break

    if baseline is None:
        logger.warning("Baseline '%s' not found in results", baseline_name)
        return {}

    comparisons = {}
    for r in results:
        if r.backend_name == baseline_name:
            continue

        comparisons[r.backend_name] = {
            "latency_ratio": r.latency_mean_ms / baseline.latency_mean_ms
            if baseline.latency_mean_ms > 0
            else float("inf"),
            "speedup": baseline.latency_mean_ms / r.latency_mean_ms if r.latency_mean_ms > 0 else 0,
            "memory_ratio": r.peak_memory_mb / baseline.peak_memory_mb
            if baseline.peak_memory_mb > 0
            else float("inf"),
            "memory_savings": (1 - r.peak_memory_mb / baseline.peak_memory_mb) * 100
            if baseline.peak_memory_mb > 0
            else 0,
            "throughput_ratio": r.throughput_fps / baseline.throughput_fps
            if baseline.throughput_fps > 0
            else 0,
        }

    return comparisons


def main() -> None:
    """CLI entry point for ``lerobot-edge-benchmark``."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark lerobot_edge policy variants")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Policy checkpoint path or HuggingFace Hub ID"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["edge_identity"],
        help="Variants to benchmark (default: edge_identity)",
    )
    parser.add_argument(
        "--device-profile",
        type=str,
        default="laptop_cpu",
        choices=["laptop_cpu", "cloud_gpu", "edge"],
        help="Device profile (default: laptop_cpu)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmark_results", help="Output directory for results"
    )
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("Benchmark CLI starting...")
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Variants: %s", args.variants)
    logger.info("Device profile: %s", args.device_profile)

    from lerobot_edge.compression.quantize import QuantizedBackend
    from lerobot_edge.core.base import IdentityBackend, NativePyTorchBackend
    from lerobot_edge.core.configs import (
        EdgeIdentityConfig,
        EdgeOnnxFp32Config,
        EdgeOnnxInt8Config,
        EdgeQuantInt8Config,
    )
    from lerobot_edge.core.utils import load_policy_from_checkpoint

    device = torch.device(args.device_profile)

    logger.info("Loading policy from %s...", args.checkpoint)
    try:
        policy = load_policy_from_checkpoint(args.checkpoint, "smolvla", str(device))
    except Exception as e:
        logger.error("Failed to load policy: %s", e)
        return

    dummy_input = build_dummy_input(policy, device)

    variants: dict[str, NativePyTorchBackend] = {}
    variant_config_map = {
        "edge_identity": EdgeIdentityConfig,
        "edge_quant_int8": EdgeQuantInt8Config,
        "edge_onnx_fp32": EdgeOnnxFp32Config,
        "edge_onnx_int8": EdgeOnnxInt8Config,
    }

    for variant_name in args.variants:
        logger.info("Creating backend for variant: %s", variant_name)
        try:
            if variant_name == "edge_identity":
                variants[variant_name] = IdentityBackend(policy)
            elif variant_name in ("edge_quant_int8",):
                edge_config = variant_config_map[variant_name](device=str(device))
                variants[variant_name] = QuantizedBackend.from_policy(policy, edge_config)
            else:
                logger.warning("Unknown variant '%s', using identity backend", variant_name)
                variants[variant_name] = IdentityBackend(policy)
        except Exception as e:
            logger.error("Failed to create backend for %s: %s", variant_name, e)

    if not variants:
        logger.error("No variants to benchmark")
        return

    results = benchmark_policy_variants(
        variants,
        dummy_input,
        device_profile=args.device_profile,
        output_dir=args.output_dir,
        warmup_runs=args.warmup,
        num_runs=args.num_runs,
    )

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Backend':<25} {'Latency (ms)':<20} {'Throughput (fps)':<20} {'Memory (MB)':<15}")
    print("-" * 80)
    for r in results:
        print(
            f"{r.backend_name:<25} {r.latency_mean_ms:>8.2f} +/- {r.latency_std_ms:<6.2f} {r.throughput_fps:>12.1f} {r.peak_memory_mb:>10.1f}"
        )
    print("=" * 80)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
