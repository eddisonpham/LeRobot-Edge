"""Systematic A/B experiment runner for edge deployment optimization."""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from lerobot_edge.core.base import (
    CompiledBackend,
    CUDAGraphBackend,
    DeploymentBackend,
    IdentityBackend,
)
from lerobot_edge.core.utils import (
    build_dummy_input,
    load_policy_from_checkpoint,
    measure_model_memory,
    measure_peak_memory_mb,
)

logger = logging.getLogger(__name__)

__all__ = ["ExperimentRunner", "ExperimentResult", "run_experiment_grid"]


@dataclass
class ExperimentResult:
    """Single experiment result row."""

    method: str
    compile_mode: str | None
    batch_size: int
    device: str
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_std_ms: float
    throughput_fps: float
    model_memory_mb: float
    peak_memory_mb: float
    num_parameters: int
    warmup_runs: int
    benchmark_runs: int
    timestamp: str = field(default_factory=time.strftime)
    extra: dict[str, Any] = field(default_factory=dict)
    attention_opt: bool = False
    kv_cache_opt: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "compile_mode": self.compile_mode,
            "attention_opt": self.attention_opt,
            "kv_cache_opt": self.kv_cache_opt,
            "batch_size": self.batch_size,
            "device": self.device,
            "latency_mean_ms": round(self.latency_mean_ms, 3),
            "latency_p50_ms": round(self.latency_p50_ms, 3),
            "latency_p95_ms": round(self.latency_p95_ms, 3),
            "latency_std_ms": round(self.latency_std_ms, 3),
            "throughput_fps": round(self.throughput_fps, 1),
            "model_memory_mb": round(self.model_memory_mb, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "num_parameters": self.num_parameters,
            "warmup_runs": self.warmup_runs,
            "benchmark_runs": self.benchmark_runs,
            "timestamp": self.timestamp,
            **self.extra,
        }


class ExperimentRunner:
    """Runs a grid of (quantization × compile mode × batch size) and saves results."""

    def __init__(
        self,
        checkpoint: str,
        *,
        policy_type: str = "smolvla",
        device: str | None = None,
        output_dir: str | Path = "experiment_results",
        warmup_runs: int = 30,
        benchmark_runs: int = 200,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_str = device
        self.device = torch.device(device)
        self.checkpoint = checkpoint
        self.policy_type = policy_type
        self.output_dir = Path(output_dir)
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self._policy: nn.Module | None = None
        self._dummy_input: dict[str, torch.Tensor] | None = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -- Policy loading --

    @property
    def policy(self) -> nn.Module:
        if self._policy is None:
            logger.info("Loading policy: %s (type=%s)", self.checkpoint, self.policy_type)
            self._policy = load_policy_from_checkpoint(
                self.checkpoint, self.policy_type, str(self.device)
            )
            self._policy.eval()
            # Apply attention optimization (SDPA/FlashAttention)
            from lerobot_edge.optimization import optimize_policy_for_inference

            self._policy = optimize_policy_for_inference(
                self._policy, enable_attention=True, enable_kv_cache_quant=True
            )
        return self._policy

    @property
    def dummy_input(self) -> dict[str, torch.Tensor]:
        if self._dummy_input is None:
            self._dummy_input = build_dummy_input(self.policy, self.device)
        return self._dummy_input

    # -- Backend factory --

    def _make_backend(
        self,
        method: str,
        compile_mode: str | None = None,
        cuda_graph: bool = False,
    ) -> DeploymentBackend:
        """Build a backend for the given method and compile mode."""
        policy = self.policy

        # --- Quantization ---
        if method == "fp32":
            quantized = policy
            quant_type = "fp32"
        elif method == "int8":
            from lerobot_edge.compression.quantize import dynamic_int8_quantize

            quantized = dynamic_int8_quantize(copy.deepcopy(policy))
            quant_type = "dynamic_int8"
        elif method == "int4":
            from lerobot_edge.compression.quantize import quantize_int4_weight_only

            quantized = quantize_int4_weight_only(
                copy.deepcopy(policy), group_size=32
            )
            quant_type = "int4_weight_only"
        elif method == "nf4":
            from lerobot_edge.compression.quantize import quantize_4bit

            quantized = quantize_4bit(copy.deepcopy(policy), quant_type="nf4")
            quant_type = "nf4"
        elif method == "bnb_int8":
            from lerobot_edge.compression.quantize import quantize_bnb_int8

            quantized = quantize_bnb_int8(copy.deepcopy(policy))
            quant_type = "bnb_int8"
        else:
            raise ValueError(f"Unknown method: {method}")

        base = IdentityBackend(quantized, self.device)

        # --- Compilation ---
        if compile_mode:
            base = CompiledBackend(base, mode=compile_mode)
            if hasattr(base, "warmup"):
                base.warmup(self.dummy_input, runs=self.warmup_runs)

        # --- CUDA graphs ---
        if cuda_graph and self.device.type == "cuda":
            cg = CUDAGraphBackend(base)
            cg.capture(self.dummy_input)
            return cg

        # Store quantization info for result metadata
        base._quantization_type = quant_type  # type: ignore[attr-defined]
        return base

    # -- Single benchmark --

    def _bench(
        self,
        backend: DeploymentBackend,
        batch_size: int,
    ) -> dict[str, Any]:
        """Benchmark a single backend for a given batch size."""
        # Build sized batch
        if batch_size == 1:
            batch = {k: v.clone() for k, v in self.dummy_input.items()}
        else:
            batch = {}
            for k, v in self.dummy_input.items():
                if isinstance(v, torch.Tensor):
                    shape = list(v.shape)
                    shape[0] = batch_size
                    batch[k] = torch.randn(
                        shape, device=self.device, dtype=v.dtype
                    )
                else:
                    batch[k] = v

        # Warmup
        for _ in range(self.warmup_runs):
            backend.predict(batch)

        # Measure
        latencies: list[float] = []
        peak_mem = 0.0
        is_cuda = self.device.type == "cuda"

        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        for _ in range(self.benchmark_runs):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            backend.predict(batch)
            if is_cuda:
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
            mem = measure_peak_memory_mb()
            if mem > peak_mem:
                peak_mem = mem

        arr = np.array(latencies)
        mean_ms = float(np.mean(arr))
        throughput = (batch_size / (mean_ms / 1000.0)) if mean_ms > 0 else 0.0

        # Model memory
        if hasattr(backend, "_policy"):
            mem_info = measure_model_memory(backend._policy)  # type: ignore[arg-type]
            model_mem = mem_info["total_mb"]
            n_params = int(mem_info["num_parameters"])
        else:
            model_mem = 0.0
            n_params = 0

        return {
            "latency_mean_ms": mean_ms,
            "latency_p50_ms": float(np.percentile(arr, 50)),
            "latency_p95_ms": float(np.percentile(arr, 95)),
            "latency_std_ms": float(np.std(arr)),
            "throughput_fps": throughput,
            "model_memory_mb": model_mem,
            "peak_memory_mb": peak_mem,
            "num_parameters": n_params,
        }

    # -- Grid runner --

    def run(
        self,
        methods: list[str] | None = None,
        compile_modes: list[str | None] | None = None,
        batch_sizes: list[int] | None = None,
        cuda_graph: bool = False,
        attention_opt: bool = True,
        kv_cache_opt: bool = True,
    ) -> list[ExperimentResult]:
        """Run the full experiment grid. Returns list of ExperimentResult."""
        if methods is None:
            methods = ["fp32", "int8", "int4"]
            if self.device.type == "cuda":
                methods.append("nf4")

        if compile_modes is None:
            compile_modes = [None]
            if self.device.type == "cuda":
                compile_modes.append("reduce-overhead")

        if batch_sizes is None:
            batch_sizes = [1, 4] if self.device.type == "cuda" else [1]

        results: list[ExperimentResult] = []
        total = len(methods) * len(compile_modes) * len(batch_sizes)
        count = 0

        logger.info("Experiment grid: %d combinations", total)
        logger.info("Methods: %s", methods)
        logger.info("Compile modes: %s", compile_modes)
        logger.info("Batch sizes: %s", batch_sizes)

        for method in methods:
            for cmode in compile_modes:
                # Build backend once per (method, cmode) pair
                try:
                    backend = self._make_backend(method, cmode, cuda_graph)
                except Exception as e:
                    logger.error(
                        "Failed to create backend (method=%s, compile=%s): %s",
                        method,
                        cmode,
                        e,
                    )
                    continue

                for bs in batch_sizes:
                    count += 1
                    logger.info(
                        "[%d/%d] %s + compile=%s @ bs=%d",
                        count,
                        total,
                        method,
                        cmode or "none",
                        bs,
                    )

                    try:
                        metrics = self._bench(backend, bs)
                    except Exception as e:
                        logger.error("Bench failed: %s", e)
                        continue

                    quant_type = getattr(
                        backend, "_quantization_type", method
                    )
                    results.append(
                        ExperimentResult(
                            method=quant_type,
                            compile_mode=cmode,
                            batch_size=bs,
                            device=self.device_str,
                            warmup_runs=self.warmup_runs,
                            benchmark_runs=self.benchmark_runs,
                            attention_opt=attention_opt,
                            kv_cache_opt=kv_cache_opt,
                            extra={
                                "checkpoint": self.checkpoint,
                                "policy_type": self.policy_type,
                            },
                            **metrics,
                        )
                    )

                # Free backend memory
                del backend
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        # Save
        self._save(results)
        return results

    # -- Save/Load --

    def _save(self, results: list[ExperimentResult]) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"experiment_{timestamp}.json"
        data = [r.to_dict() for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Results saved to %s", path)
        return path

    @staticmethod
    def load(path: str | Path) -> list[ExperimentResult]:
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        results = []
        for item in data:
            extra = {
                k: v
                for k, v in item.items()
                if k not in ExperimentResult.__dataclass_fields__
                and k not in ("method", "compile_mode", "batch_size", "device")
            }
            results.append(
                ExperimentResult(
                    method=item["method"],
                    compile_mode=item.get("compile_mode"),
                    batch_size=item["batch_size"],
                    device=item["device"],
                    latency_mean_ms=item["latency_mean_ms"],
                    latency_p50_ms=item["latency_p50_ms"],
                    latency_p95_ms=item["latency_p95_ms"],
                    latency_std_ms=item["latency_std_ms"],
                    throughput_fps=item["throughput_fps"],
                    model_memory_mb=item["model_memory_mb"],
                    peak_memory_mb=item["peak_memory_mb"],
                    num_parameters=item["num_parameters"],
                    warmup_runs=item.get("warmup_runs", 0),
                    benchmark_runs=item.get("benchmark_runs", 0),
                    timestamp=item.get("timestamp", ""),
                    extra=extra,
                )
            )
        return results

    # -- Reporting --

    @staticmethod
    def print_summary(results: list[ExperimentResult]) -> None:
        """Print comparison table grouped by batch size."""
        if not results:
            print("No results.")
            return

        # Group by batch size
        for bs in sorted(set(r.batch_size for r in results)):
            subset = [r for r in results if r.batch_size == bs]
            subset.sort(key=lambda r: r.latency_mean_ms)

            print(f"\n{'=' * 100}")
            print(f"  Batch Size = {bs}")
            print(f"{'=' * 100}")
            header = (
                f"{'Method':<25} {'Compile':<18} "
                f"{'Mean (ms)':<12} {'P50 (ms)':<12} {'P95 (ms)':<12} "
                f"{'FPS':<10} {'Mem (MB)':<12} {'Peak (MB)':<12}"
            )
            print(header)
            print("-" * 100)

            for r in subset:
                cmode = r.compile_mode or "none"
                print(
                    f"{r.method:<25} {cmode:<18} "
                    f"{r.latency_mean_ms:>8.2f}    "
                    f"{r.latency_p50_ms:>8.2f}    "
                    f"{r.latency_p95_ms:>8.2f}    "
                    f"{r.throughput_fps:>8.1f}  "
                    f"{r.model_memory_mb:>8.1f}    "
                    f"{r.peak_memory_mb:>8.1f}"
                )
            print("=" * 100)

    @staticmethod
    def compute_speedups(
        results: list[ExperimentResult],
        baseline_method: str = "fp32",
        baseline_compile: str | None = None,
    ) -> dict[str, Any]:
        """Compute speedups vs baseline method."""
        speedups: dict[str, Any] = {}
        for batch_size in sorted(set(r.batch_size for r in results)):
            baseline = None
            for r in results:
                if (
                    r.batch_size == batch_size
                    and r.method == baseline_method
                    and r.compile_mode == baseline_compile
                ):
                    baseline = r
                    break

            if baseline is None:
                continue

            speedups[f"bs={batch_size}"] = {}
            for r in results:
                if r.batch_size != batch_size:
                    continue
                key = f"{r.method}"
                if r.compile_mode:
                    key += f"+compile={r.compile_mode}"
                speedups[f"bs={batch_size}"][key] = {
                    "latency_ms": r.latency_mean_ms,
                    "speedup": (
                        baseline.latency_mean_ms / r.latency_mean_ms
                        if r.latency_mean_ms > 0
                        else 0
                    ),
                    "memory_ratio": (
                        r.model_memory_mb / baseline.model_memory_mb
                        if baseline.model_memory_mb > 0
                        else 0
                    ),
                }

        return speedups


# -- Convenience --


def run_experiment_grid(
    checkpoint: str,
    output_dir: str | Path = "experiment_results",
    **kwargs: Any,
) -> list[ExperimentResult]:
    """Create runner and run default grid."""
    runner = ExperimentRunner(
        checkpoint=checkpoint,
        output_dir=output_dir,
        **kwargs,
    )
    return runner.run()


# -- CLI --


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run systematic A/B experiment grid"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="HuggingFace Hub ID or local checkpoint path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Quantization methods (default: fp32 int8 int4 [nf4 if GPU])",
    )
    parser.add_argument(
        "--compile-modes",
        nargs="+",
        default=None,
        help="Compile modes: none reduce-overhead max-autotune",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Batch sizes (default: 1,4 for GPU, 1 for CPU)",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Test CUDA graph capture",
    )
    parser.add_argument(
        "--warmup", type=int, default=30, help="Warmup runs"
    )
    parser.add_argument(
        "--num-runs", type=int, default=200, help="Benchmark runs"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )

    # Parse compile modes
    compile_modes: list[str | None] | None = None
    if args.compile_modes:
        compile_modes = [
            None if m.lower() == "none" else m for m in args.compile_modes
        ]

    runner = ExperimentRunner(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        warmup_runs=args.warmup,
        benchmark_runs=args.num_runs,
    )
    results = runner.run(
        methods=args.methods,
        compile_modes=compile_modes,
        batch_sizes=args.batch_sizes,
        cuda_graph=args.cuda_graph,
    )

    runner.print_summary(results)

    # Print speedups
    speedups = ExperimentRunner.compute_speedups(results)
    if speedups:
        print("\nSpeedups vs FP32 baseline:")
        for bs_key, variants in speedups.items():
            print(f"  {bs_key}:")
            for name, data in sorted(
                variants.items(), key=lambda x: x[1]["speedup"], reverse=True
            ):
                marker = " 🔥" if data["speedup"] > 1.5 else ""
                print(
                    f"    {name:<35} "
                    f"{data['latency_ms']:>7.2f} ms  "
                    f"{data['speedup']:>5.2f}x{marker}"
                )


if __name__ == "__main__":
    main()
