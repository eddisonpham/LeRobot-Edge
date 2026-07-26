"""Evaluation metrics for quantization quality and pipeline assessment."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "OutputDivergence",
    "QuantizationQualityReport",
    "measure_output_divergence",
    "compare_backends",
    "bootstrap_confidence_interval",
]


@dataclass
class OutputDivergence:
    """Metrics comparing original vs quantized model outputs."""

    mse: float = 0.0
    mae: float = 0.0
    cosine_similarity: float = 0.0
    max_abs_error: float = 0.0
    relative_error_pct: float = 0.0
    num_samples: int = 0

    @property
    def degradation_pct(self) -> float:
        """Quality degradation as percentage (lower is better)."""
        return (1.0 - self.cosine_similarity) * 100


@dataclass
class QuantizationQualityReport:
    """Full quality assessment of a quantization pass."""

    variant_name: str
    original_params: int = 0
    quantized_params: int = 0
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    divergence: OutputDivergence = field(default_factory=OutputDivergence)
    latency_original_ms: float = 0.0
    latency_quantized_ms: float = 0.0
    memory_original_mb: float = 0.0
    memory_quantized_mb: float = 0.0

    @property
    def compression_ratio(self) -> float:
        if self.original_size_mb == 0:
            return 0.0
        return self.original_size_mb / self.quantized_size_mb

    @property
    def memory_savings_pct(self) -> float:
        if self.memory_original_mb == 0:
            return 0.0
        return (1.0 - self.memory_quantized_mb / self.memory_original_mb) * 100

    @property
    def speedup_ratio(self) -> float:
        if self.latency_quantized_ms == 0:
            return 0.0
        return self.latency_original_ms / self.latency_quantized_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant_name,
            "compression_ratio": round(self.compression_ratio, 2),
            "memory_savings_pct": round(self.memory_savings_pct, 1),
            "quality_degradation_pct": round(self.divergence.degradation_pct, 2),
            "cosine_similarity": round(self.divergence.cosine_similarity, 6),
            "mse": round(self.divergence.mse, 8),
            "mae": round(self.divergence.mae, 8),
            "speedup_ratio": round(self.speedup_ratio, 2),
            "latency_original_ms": round(self.latency_original_ms, 2),
            "latency_quantized_ms": round(self.latency_quantized_ms, 2),
            "original_params": self.original_params,
            "quantized_params": self.quantized_params,
        }


def measure_output_divergence(
    original_outputs: list[torch.Tensor],
    quantized_outputs: list[torch.Tensor],
) -> OutputDivergence:
    """Compute divergence metrics between original and quantized model outputs.

    Args:
        original_outputs: List of output tensors from the original model.
        quantized_outputs: List of output tensors from the quantized model.

    Returns:
        OutputDivergence with MSE, MAE, cosine similarity, and max error.
    """
    if len(original_outputs) != len(quantized_outputs):
        raise ValueError(
            f"Output lists must have same length: {len(original_outputs)} vs {len(quantized_outputs)}"
        )

    if not original_outputs:
        return OutputDivergence()

    all_mse = []
    all_mae = []
    all_cosine = []
    all_max_err = []
    all_rel_err = []
    total_samples = 0

    for orig, quant in zip(original_outputs, quantized_outputs):
        orig_f = orig.float().detach()
        quant_f = quant.float().detach()

        if orig_f.shape != quant_f.shape:
            logger.warning("Shape mismatch: %s vs %s, skipping pair", orig_f.shape, quant_f.shape)
            continue

        mse = torch.mean((orig_f - quant_f) ** 2).item()
        mae = torch.mean(torch.abs(orig_f - quant_f)).item()

        orig_flat = orig_f.flatten()
        quant_flat = quant_f.flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            orig_flat.unsqueeze(0), quant_flat.unsqueeze(0)
        ).item()

        max_err = torch.max(torch.abs(orig_f - quant_f)).item()

        orig_mean = orig_f.mean().abs().item()
        rel_err = (max_err / orig_mean * 100) if orig_mean > 1e-8 else 0.0

        n = orig_f.numel()
        all_mse.append(mse * n)
        all_mae.append(mae * n)
        all_cosine.append(cos_sim * n)
        all_max_err.append(max_err)
        all_rel_err.append(rel_err)
        total_samples += n

    if total_samples == 0:
        return OutputDivergence()

    return OutputDivergence(
        mse=sum(all_mse) / total_samples,
        mae=sum(all_mae) / total_samples,
        cosine_similarity=sum(all_cosine) / total_samples,
        max_abs_error=max(all_max_err),
        relative_error_pct=np.mean(all_rel_err),
        num_samples=total_samples,
    )


def compare_backends(
    original: nn.Module,
    quantized: nn.Module,
    dummy_input: dict[str, torch.Tensor],
    num_samples: int = 10,
) -> QuantizationQualityReport:
    """Run original and quantized models on dummy inputs and compute full quality report.

    Args:
        original: The original FP32 model.
        quantized: The quantized model.
        dummy_input: Input batch for inference.
        num_samples: Number of forward passes to compare.

    Returns:
        QuantizationQualityReport with all quality and performance metrics.
    """
    from lerobot_edge.utils import measure_model_memory

    original.eval()
    quantized.eval()

    orig_mem = measure_model_memory(original)
    quant_mem = measure_model_memory(quantized)

    orig_outputs = []
    quant_outputs = []
    latencies_orig = []
    latencies_quant = []

    for _ in range(num_samples):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = original.select_action(dummy_input)
        latencies_orig.append((time.perf_counter() - t0) * 1000)
        orig_outputs.append(out if isinstance(out, torch.Tensor) else out[0])

        t0 = time.perf_counter()
        with torch.no_grad():
            out = quantized.select_action(dummy_input)
        latencies_quant.append((time.perf_counter() - t0) * 1000)
        quant_outputs.append(out if isinstance(out, torch.Tensor) else out[0])

    divergence = measure_output_divergence(orig_outputs, quant_outputs)

    return QuantizationQualityReport(
        variant_name=type(quantized).__name__,
        original_params=orig_mem["num_parameters"],
        quantized_params=quant_mem["num_parameters"],
        original_size_mb=orig_mem["total_mb"],
        quantized_size_mb=quant_mem["total_mb"],
        divergence=divergence,
        latency_original_ms=float(np.mean(latencies_orig)),
        latency_quantized_ms=float(np.mean(latencies_quant)),
        memory_original_mb=orig_mem["total_mb"],
        memory_quantized_mb=quant_mem["total_mb"],
    )


def bootstrap_confidence_interval(
    data: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        data: List of metric values.
        confidence: Confidence level (default 0.95).
        n_bootstrap: Number of bootstrap resamples.

    Returns:
        Tuple of (mean, lower_bound, upper_bound).
    """
    if not data:
        return (0.0, 0.0, 0.0)

    arr = np.array(data)
    mean_val = float(np.mean(arr))

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        bootstrap_means.append(float(np.mean(sample)))

    alpha = (1 - confidence) / 2
    lower = float(np.percentile(bootstrap_means, alpha * 100))
    upper = float(np.percentile(bootstrap_means, (1 - alpha) * 100))

    return (mean_val, lower, upper)
