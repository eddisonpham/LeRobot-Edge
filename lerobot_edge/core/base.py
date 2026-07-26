"""CompressedPolicy wrapper and deployment backend interface."""

from __future__ import annotations

import abc
import logging
from typing import Any

import torch
import torch.nn as nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot_edge.core.configs import EdgeBaseConfig

logger = logging.getLogger(__name__)

__all__ = [
    "DeploymentBackend",
    "NativePyTorchBackend",
    "IdentityBackend",
    "CompressedPolicy",
]


class DeploymentBackend(abc.ABC):
    """Abstract interface for a deployment backend."""

    @abc.abstractmethod
    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        ...

    @property
    @abc.abstractmethod
    def parameters(self) -> list[nn.Parameter]:
        ...


class NativePyTorchBackend(DeploymentBackend):
    """Wraps an existing PyTorch policy module as a deployment backend."""

    def __init__(self, policy: nn.Module, device: torch.device | None = None) -> None:
        self._policy = policy
        self._device = device or next(policy.parameters()).device
        self._policy.eval()

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            return self._policy.select_action(batch)

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def parameters(self) -> list[nn.Parameter]:
        return list(self._policy.parameters())


class IdentityBackend(NativePyTorchBackend):
    """Passthrough backend — no transformation applied."""


class CompressedPolicy(PreTrainedPolicy):
    """Wraps a DeploymentBackend behind LeRobot's policy interface."""

    config_class = EdgeBaseConfig  # type: ignore[assignment]
    name = "edge_compressed"  # type: ignore[assignment]

    def __init__(
        self,
        config: EdgeBaseConfig,
        backend: DeploymentBackend | None = None,
        *,
        pretrained_name_or_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)

        if backend is not None:
            self._backend = backend
        elif pretrained_name_or_path is not None:
            self._backend = self._build_backend_from_checkpoint(
                pretrained_name_or_path, config
            )
        else:
            self._backend = _PlaceholderBackend(config.device or "cpu")

        self._action_cache: torch.Tensor | None = None
        self._cache_idx: int = 0

    def _build_backend_from_checkpoint(
        self, path: str, config: EdgeBaseConfig
    ) -> DeploymentBackend:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _build_backend_from_checkpoint"
        )

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self._action_cache is not None and self._cache_idx < self._action_cache.shape[0]:
            action = self._action_cache[self._cache_idx]
            self._cache_idx += 1
            return action

        self._action_cache = self.predict_action_chunk(batch)
        self._cache_idx = 1
        return self._action_cache[0]

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return self._backend.predict(batch)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        actions = self._backend.predict(batch)
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        return loss, {"actions": actions}

    def reset(self) -> None:
        self._action_cache = None
        self._cache_idx = 0
        self._backend.reset()

    def get_optim_params(self) -> dict[str, Any]:
        return {"params": self._backend.parameters}

    @property
    def device(self) -> torch.device:
        return self._backend.device


class _PlaceholderBackend(DeploymentBackend):
    """Minimal backend that returns zeros — used for registration tests."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self._device = torch.device(device)

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        for key, val in batch.items():
            if "action" in key.lower():
                return torch.zeros(val.shape, device=self._device)
        return torch.zeros(1, device=self._device)

    def reset(self) -> None:
        pass

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def parameters(self) -> list[nn.Parameter]:
        return []
