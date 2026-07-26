"""Edge/cloud confidence-based router."""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn

from lerobot_edge.core.base import DeploymentBackend

logger = logging.getLogger(__name__)

__all__ = [
    "ConfidenceRouter",
]


class ConfidenceRouter:
    """Routes inference between edge and cloud backends based on confidence."""

    def __init__(
        self,
        edge_backend: DeploymentBackend,
        cloud_backend: DeploymentBackend | None = None,
        confidence_threshold: float = 0.5,
        max_escalation_rate: float = 0.1,
    ) -> None:
        self._edge = edge_backend
        self._cloud = cloud_backend
        self._threshold = confidence_threshold
        self._max_escalation_rate = max_escalation_rate
        self._total_inferences = 0
        self._escalations = 0

    def predict(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run inference with confidence-based routing."""
        self._total_inferences += 1

        edge_actions = self._edge.predict(batch)
        confidence = self._compute_confidence(edge_actions)

        escalated = False
        if self._cloud is not None and confidence < self._threshold:
            escalation_rate = self._escalations / max(self._total_inferences, 1)
            if escalation_rate < self._max_escalation_rate:
                cloud_actions = self._cloud.predict(batch)
                edge_actions = cloud_actions
                escalated = True
                self._escalations += 1
                logger.debug("Escalated to cloud (confidence=%.3f < threshold=%.3f)", confidence, self._threshold)

        info = {
            "confidence": confidence,
            "escalated": escalated,
            "escalation_rate": self._escalations / max(self._total_inferences, 1),
            "total_inferences": self._total_inferences,
            "total_escalations": self._escalations,
        }

        return edge_actions, info

    def _compute_confidence(self, actions: torch.Tensor) -> float:
        """Compute confidence from action variance (lower variance = higher confidence)."""
        if actions.numel() == 0:
            return 1.0

        mean = actions.mean()
        std = actions.std()
        if mean.abs() < 1e-8:
            return 1.0

        cv = (std / mean.abs()).item()
        x = max(-500.0, min(500.0, -cv * 5 + 2.5))
        confidence = 1.0 / (1.0 + math.exp(-x))
        return confidence

    @property
    def stats(self) -> dict[str, Any]:
        """Return routing statistics."""
        return {
            "total_inferences": self._total_inferences,
            "escalations": self._escalations,
            "escalation_rate": self._escalations / max(self._total_inferences, 1),
            "confidence_threshold": self._threshold,
        }

    def reset(self) -> None:
        """Reset routing statistics."""
        self._total_inferences = 0
        self._escalations = 0
        self._edge.reset()
        if self._cloud is not None:
            self._cloud.reset()
