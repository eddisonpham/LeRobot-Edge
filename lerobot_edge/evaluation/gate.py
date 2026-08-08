"""Quality gate for quantization divergence thresholds."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from lerobot_edge.evaluation.metrics import measure_output_divergence

logger = logging.getLogger(__name__)

__all__ = ["QualityGate", "QualityGateResult"]


@dataclass
class QualityGateResult:
    passed: bool
    cosine_similarity: float
    mse: float
    threshold_cosine: float
    threshold_mse: float
    message: str


class QualityGate:
    """Rejects quantized models whose outputs diverge past thresholds."""

    def __init__(
        self,
        min_cosine_similarity: float = 0.98,
        max_mse: float | None = None,
        num_samples: int = 20,
    ) -> None:
        self.min_cosine = min_cosine_similarity
        self.max_mse = max_mse
        self.num_samples = num_samples

    def check(
        self,
        original: nn.Module,
        quantized: nn.Module,
        dummy_input: dict[str, torch.Tensor],
    ) -> QualityGateResult:
        """Run both models and check divergence against thresholds."""
        original.eval()
        quantized.eval()

        orig_outputs = []
        quant_outputs = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                o = original.select_action(dummy_input)  # type: ignore[operator]
                q = quantized.select_action(dummy_input)  # type: ignore[operator]
                orig_outputs.append(o if isinstance(o, torch.Tensor) else o[0])
                quant_outputs.append(q if isinstance(q, torch.Tensor) else q[0])

        div = measure_output_divergence(orig_outputs, quant_outputs)

        cos_ok = div.cosine_similarity >= self.min_cosine
        mse_ok = self.max_mse is None or div.mse <= self.max_mse
        passed = cos_ok and mse_ok

        parts = []
        if not cos_ok:
            parts.append(f"cosine similarity {div.cosine_similarity:.6f} < {self.min_cosine}")
        if not mse_ok:
            parts.append(f"MSE {div.mse:.8f} > {self.max_mse}")

        if passed:
            msg = f"PASSED (cos={div.cosine_similarity:.6f}, mse={div.mse:.8f})"
        else:
            msg = f"FAILED: {'; '.join(parts)}"

        logger.info("Quality gate %s", "PASSED" if passed else "FAILED")

        return QualityGateResult(
            passed=passed,
            cosine_similarity=div.cosine_similarity,
            mse=div.mse,
            threshold_cosine=self.min_cosine,
            threshold_mse=self.max_mse or float("inf"),
            message=msg,
        )
