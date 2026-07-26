"""Benchmark harness for lerobot_edge policy variants.

Measures mean/p50/p95 latency per inference, peak memory, throughput
(inferences/sec) — averaged over enough runs to be stable.

Outputs a single structured JSON/CSV row per (backend, device_profile) combination,
including reproducibility fields (commit hash, config, timestamp).
"""

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

from lerobot_edge.base import DeploymentBackend
from lerobot_edge.configs import EdgeBaseConfig
from lerobot_edge.utils import get_git_commit_hash, measure_model_memory, measure_peak_memory_mb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Benchmark result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Single benchmark result for one backend + device profile combination."""

    # Identity
    backend_name: str
    device_profile: str
    git_commit: str
    timestamp: str

    # Configuration
    config: dict[str, Any] = field(default_factory=dict)

    # Latency (ms)
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_std_ms: float = 0.0

    # Throughput
    throughput_fps: float = 0.0  # inferences per second

    # Memory
    peak_memory_mb: float = 0.0
    param_memory_mb: float = 0.0

    # Model info
    num_parameters: int = 0
    model_size_mb: float = 0.0

    # Success rate (from eval)
    success_rate: float | None = None
    eval_episodes: int = 0

    # Metadata
    warmup_runs: int = 0
    benchmark_runs: int = 0
    notes: str = ""





# ---------------------------------------------------------------------------
# Core benchmark functions
# ---------------------------------------------------------------------------


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
    """Benchmark a deployment backend.

    Args:
        backend: The deployment backend to benchmark.
        dummy_input: Sample input batch for inference.
        warmup_runs: Number of warmup runs (not counted in results).
        num_runs: Number of benchmark runs.
        backend_name: Name identifier for this backend.
        device_profile: Device profile name (e.g. "laptop_cpu", "cloud_gpu").
        config: Optional edge configuration.

    Returns:
        BenchmarkResult with all measurements.
    """
    logger.info(
        "Benchmarking %s on %s: warmup=%d, runs=%d",
        backend_name,
        device_profile,
        warmup_runs,
        num_runs,
    )

    # Move input to correct device
    device = backend.device
    dummy_input = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in dummy_input.items()
    }

    # Warmup
    logger.debug("Running %d warmup iterations...", warmup_runs)
    for _ in range(warmup_runs):
        _ = backend.predict(dummy_input)

    # Benchmark
    latencies_ms: list[float] = []
    peak_mem = 0.0

    # Reset peak memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    logger.debug("Running %d benchmark iterations...", num_runs)
    for i in range(num_runs):
        start_time = time.perf_counter()

        with torch.no_grad():
            _ = backend.predict(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        latencies_ms.append((end_time - start_time) * 1000)

        # Track peak memory
        current_mem = measure_peak_memory_mb()
        if current_mem > peak_mem:
            peak_mem = current_mem

    # Compute statistics
    latencies = np.array(latencies_ms)
    latency_mean = float(np.mean(latencies))
    latency_p50 = float(np.percentile(latencies, 50))
    latency_p95 = float(np.percentile(latencies, 95))
    latency_p99 = float(np.percentile(latencies, 99))
    latency_std = float(np.std(latencies))

    throughput = 1000.0 / latency_mean if latency_mean > 0 else 0.0

    # Model memory
    if hasattr(backend, '_policy') and hasattr(backend._policy, 'parameters'):
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
        "Benchmark complete: latency=%.2f±%.2f ms (p50=%.2f, p95=%.2f), throughput=%.1f fps",
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
    """Benchmark multiple policy variants and save results.

    Args:
        variants: Dict mapping variant names to backends.
        dummy_input: Sample input batch.
        device_profile: Device profile name.
        output_dir: Directory to save results.
        **kwargs: Additional arguments passed to benchmark_backend.

    Returns:
        List of BenchmarkResult objects.
    """
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

    # Save results
    _save_results_json(results, output_dir / "benchmark_results.json")
    _save_results_csv(results, output_dir / "benchmark_results.csv")

    return results


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------


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

    results = []
    for item in data:
        results.append(BenchmarkResult(**item))
    return results


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


def compare_results(
    results: list[BenchmarkResult],
    baseline_name: str = "edge_identity",
) -> dict[str, Any]:
    """Compare benchmark results against a baseline.

    Returns:
        Dict with comparison metrics for each variant.
    """
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
            "latency_ratio": r.latency_mean_ms / baseline.latency_mean_ms if baseline.latency_mean_ms > 0 else float('inf'),
            "speedup": baseline.latency_mean_ms / r.latency_mean_ms if r.latency_mean_ms > 0 else 0,
            "memory_ratio": r.peak_memory_mb / baseline.peak_memory_mb if baseline.peak_memory_mb > 0 else float('inf'),
            "memory_savings": (1 - r.peak_memory_mb / baseline.peak_memory_mb) * 100 if baseline.peak_memory_mb > 0 else 0,
            "throughput_ratio": r.throughput_fps / baseline.throughput_fps if baseline.throughput_fps > 0 else 0,
        }

    return comparisons


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``lerobot-edge-benchmark``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark lerobot_edge policy variants"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Policy checkpoint path or HuggingFace Hub ID",
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
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Number of warmup runs"
    )
    parser.add_argument(
        "--num-runs", type=int, default=100, help="Number of benchmark runs"
    )

    args = parser.parse_args()

    logger.info("Benchmark CLI starting...")
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Variants: %s", args.variants)
    logger.info("Device profile: %s", args.device_profile)

    # TODO: Load checkpoint and create backends for each variant
    # For now, log the configuration
    logger.info(
        "Configuration: warmup=%d, num_runs=%d, output_dir=%s",
        args.warmup,
        args.num_runs,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
