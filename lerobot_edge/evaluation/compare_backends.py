"""Benchmark comparison across quantization backends."""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from lerobot_edge.compression.quantize import (
    HAS_BNB,
    QuantizedBackend,
    quantize_4bit,
    quantize_bnb_fp4,
    quantize_bnb_int8,
)
from lerobot_edge.core.base import CompiledBackend, IdentityBackend, NativePyTorchBackend
from lerobot_edge.core.configs import EdgeQuantInt8Config
from lerobot_edge.core.utils import (
    build_dummy_input,
    load_policy_from_checkpoint,
    measure_model_memory,
)
from lerobot_edge.evaluation.gate import QualityGate

logger = logging.getLogger(__name__)

__all__ = ["compare_all_backends", "print_comparison"]


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(7, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))

    def select_action(self, batch):
        for val in batch.values():
            if isinstance(val, torch.Tensor) and val.dim() == 2:
                return self.forward(val)
        return self.forward(list(batch.values())[0])

    def reset(self):
        return


def _simple_model():
    return _SimpleModel(), {"observation.state": torch.randn(1, 7)}


DEFAULT_BACKENDS = ["identity", "int8"]
ALL_BACKENDS = [
    "identity", "int8", "bnb_int8", "nf4", "bnb_fp4",
    "onnx_fp32", "onnx_int8",
    "identity_compile", "int8_compile",
]


def compare_all_backends(
    model: nn.Module,
    dummy_input: dict[str, torch.Tensor],
    *,
    warmup: int = 10,
    num_runs: int = 100,
    device: str = "cpu",
    backends: list[str] | None = None,
    compile_mode: str | None = None,
) -> dict[str, Any]:
    """Benchmark selected backends against the original model.

    Args:
        backends: Which backends to benchmark. Default ["identity", "int8"].
            Options: "identity", "int8", "onnx_fp32", "onnx_int8",
            "identity_compile", "int8_compile".
        compile_mode: torch.compile mode (e.g. "max-autotune"). None disables.
    """
    if backends is None:
        backends = list(DEFAULT_BACKENDS)

    dev = torch.device(device)
    results: dict[str, Any] = {}

    if "identity" in backends:
        identity = IdentityBackend(model, dev)
        results["identity"] = _bench_backend(identity, dummy_input, warmup, num_runs)
        results["identity"]["memory"] = measure_model_memory(model)

    if "int8" in backends:
        quant_config = EdgeQuantInt8Config(device=device)
        quantized = QuantizedBackend.from_policy(model, quant_config)
        results["dynamic_int8"] = _bench_backend(quantized, dummy_input, warmup, num_runs)
        if hasattr(quantized, "_policy"):
            results["dynamic_int8"]["memory"] = measure_model_memory(quantized._policy)

    if "identity_compile" in backends:
        mode = compile_mode or "max-autotune"
        identity = IdentityBackend(model, dev)
        compiled = CompiledBackend(identity, mode=mode)
        results["identity_compiled"] = _bench_backend(
            compiled, dummy_input, warmup, num_runs, is_compiled=True
        )
        results["identity_compiled"]["memory"] = measure_model_memory(model)

    if "int8_compile" in backends:
        mode = compile_mode or "max-autotune"
        quant_config = EdgeQuantInt8Config(device=device)
        quantized = QuantizedBackend.from_policy(model, quant_config)
        compiled = CompiledBackend(quantized, mode=mode)
        results["int8_compiled"] = _bench_backend(
            compiled, dummy_input, warmup, num_runs, is_compiled=True
        )
        if hasattr(quantized, "_policy"):
            results["int8_compiled"]["memory"] = measure_model_memory(quantized._policy)

    if compile_mode and "identity" in backends and "identity_compile" not in backends:
        identity = IdentityBackend(model, dev)
        compiled = CompiledBackend(identity, mode=compile_mode)
        results["identity_compiled"] = _bench_backend(
            compiled, dummy_input, warmup, num_runs, is_compiled=True
        )
        results["identity_compiled"]["memory"] = measure_model_memory(model)

    if compile_mode and "int8" in backends and "int8_compile" not in backends:
        quant_config = EdgeQuantInt8Config(device=device)
        quantized = QuantizedBackend.from_policy(model, quant_config)
        compiled = CompiledBackend(quantized, mode=compile_mode)
        results["int8_compiled"] = _bench_backend(
            compiled, dummy_input, warmup, num_runs, is_compiled=True
        )
        if hasattr(quantized, "_policy"):
            results["int8_compiled"]["memory"] = measure_model_memory(quantized._policy)

    if "bnb_int8" in backends and HAS_BNB:
        bnb_int8_model = quantize_bnb_int8(copy.deepcopy(model))
        bnb_int8_backend = NativePyTorchBackend(bnb_int8_model, dev)
        results["bnb_int8"] = _bench_backend(bnb_int8_backend, dummy_input, warmup, num_runs)
        results["bnb_int8"]["memory"] = measure_model_memory(bnb_int8_model)

    if "nf4" in backends and HAS_BNB:
        nf4_model = quantize_4bit(copy.deepcopy(model))
        nf4_backend = NativePyTorchBackend(nf4_model, dev)
        results["nf4"] = _bench_backend(nf4_backend, dummy_input, warmup, num_runs)
        results["nf4"]["memory"] = measure_model_memory(nf4_model)

    if "bnb_fp4" in backends and HAS_BNB:
        fp4_model = quantize_bnb_fp4(copy.deepcopy(model))
        fp4_backend = NativePyTorchBackend(fp4_model, dev)
        results["bnb_fp4"] = _bench_backend(fp4_backend, dummy_input, warmup, num_runs)
        results["bnb_fp4"]["memory"] = measure_model_memory(fp4_model)

    if "onnx_fp32" in backends:
        from lerobot_edge.core.configs import EdgeOnnxFp32Config

        _bench_onnx(
            model, dummy_input, dev, warmup, num_runs, results,
            "onnx_fp32", EdgeOnnxFp32Config, "model.onnx",
        )

    if "onnx_int8" in backends:
        from lerobot_edge.core.configs import EdgeOnnxInt8Config

        _bench_onnx(
            model, dummy_input, dev, warmup, num_runs, results,
            "onnx_int8", EdgeOnnxInt8Config, "model_int8.onnx",
        )

    return results


def _bench_onnx(model, dummy_input, dev, warmup, num_runs, results, key, config_cls, filename):
    try:
        from lerobot_edge.export.onnx import (
            HAS_ONNX,
            HAS_ORT,
            OnnxRuntimeBackend,
            export_policy_to_onnx,
        )

        if not (HAS_ONNX and HAS_ORT):
            logger.warning("ONNX benchmark skipped (%s): onnx/onnxruntime not installed", key)
            return
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / filename
            export_policy_to_onnx(model, config_cls(device=str(dev)), onnx_path)
            ort_backend = OnnxRuntimeBackend(onnx_path, device=dev)
            results[key] = _bench_backend(ort_backend, dummy_input, warmup, num_runs)
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        logger.warning("ONNX benchmark failed (%s): %s", key, e)


def _bench_backend(backend, dummy_input, warmup, num_runs, is_compiled=False):
    device = backend.device
    inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dummy_input.items()}

    effective_warmup = warmup + (10 if is_compiled else 0)
    for _ in range(effective_warmup):
        backend.predict(inp)

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            backend.predict(inp)
            latencies.append((time.perf_counter() - start) * 1000)

    arr = np.array(latencies)
    return {
        "latency_mean_ms": float(np.mean(arr)),
        "latency_p50_ms": float(np.percentile(arr, 50)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_std_ms": float(np.std(arr)),
        "throughput_fps": 1000.0 / float(np.mean(arr)) if np.mean(arr) > 0 else 0,
        "num_runs": num_runs,
    }


def print_comparison(results: dict[str, Any]) -> None:
    """Print formatted benchmark comparison table."""
    print("\n" + "=" * 80)
    print("BACKEND COMPARISON")
    print("=" * 80)
    print(f"{'Backend':<20} {'Mean (ms)':<12} {'P50 (ms)':<12} {'P95 (ms)':<12} {'FPS':<10}")
    print("-" * 80)

    backend_results = {
        k: v for k, v in results.items() if isinstance(v, dict) and "latency_mean_ms" in v
    }
    for name, r in backend_results.items():
        mem = r.get("memory", {})
        mem_str = f" ({mem.get('total_mb', 0):.2f} MB)" if mem else ""
        print(
            f"{name + mem_str:<20} {r['latency_mean_ms']:>8.2f}    {r['latency_p50_ms']:>8.2f}    {r['latency_p95_ms']:>8.2f}    {r['throughput_fps']:>8.1f}"
        )
    print("=" * 80)

    if any(k in backend_results for k in ["dynamic_int8", "onnx_int8"]):
        print("NOTE: INT8 quantization adds framework dispatch overhead on small/CPU models.")
        print("      Benefits appear on larger models (>100M params) or with torch.compile.")


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Compare backend performance")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="HuggingFace Hub model ID or local path"
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default="smolvla",
        help="Policy type for loading (default: smolvla)",
    )
    parser.add_argument("--output", type=str, default="benchmark_results/comparison.json")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        choices=ALL_BACKENDS,
        help=f"Backends to benchmark (default: identity int8). Options: {', '.join(ALL_BACKENDS)}",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for GPU kernel fusion",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (default: max-autotune)",
    )
    args = parser.parse_args()

    if args.checkpoint:
        logger.info("Loading real checkpoint: %s (type=%s)", args.checkpoint, args.policy_type)
        try:
            model = load_policy_from_checkpoint(args.checkpoint, args.policy_type, args.device)
            dummy_input = build_dummy_input(model, torch.device(args.device))
        except Exception as e:
            logger.warning(
                "Could not load checkpoint (missing dependencies). Falling back to simple model."
            )
            logger.debug("Load error: %s", e)
            model, dummy_input = _simple_model()
    else:
        model, dummy_input = _simple_model()

    compile_mode = args.compile_mode if args.compile else None
    results = compare_all_backends(
        model,
        dummy_input,
        warmup=args.warmup,
        num_runs=args.num_runs,
        device=args.device,
        backends=args.backends,
        compile_mode=compile_mode,
    )
    results["checkpoint"] = args.checkpoint or "simple_model"
    results["policy_type"] = args.policy_type if args.checkpoint else None
    print_comparison(results)

    quant_config = EdgeQuantInt8Config(device=args.device)
    quantized_backend = QuantizedBackend.from_policy(model, quant_config)
    gate = QualityGate(min_cosine_similarity=0.98, num_samples=10)
    gate_result = gate.check(model, quantized_backend._policy, dummy_input)
    results["quality_gate"] = {
        "passed": gate_result.passed,
        "cosine_similarity": gate_result.cosine_similarity,
        "mse": gate_result.mse,
        "threshold_cosine": gate_result.threshold_cosine,
        "message": gate_result.message,
    }
    print(f"\nQuality Gate: {gate_result.message}")
    if not gate_result.passed:
        logger.error("Quality gate FAILED — quantization divergence exceeds threshold")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
