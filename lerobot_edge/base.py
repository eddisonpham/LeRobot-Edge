"""CompressedPolicy wrapper and deployment backend interface.

``CompressedPolicy`` wraps any deployment backend (quantized weights, ONNX
Runtime session, TensorRT engine, distilled student model) behind the exact
call signature that LeRobot's eval/record scripts expect from a native policy.

Every edge variant ultimately produces a ``CompressedPolicy`` instance that
is registered and handed to the eval loop.
"""

from __future__ import annotations

import abc
import time
import logging
from typing import Any

import torch
import torch.nn as nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot_edge.configs import EdgeBaseConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DeploymentBackend interface
# ---------------------------------------------------------------------------

class DeploymentBackend(abc.ABC):
    """Abstract interface for a deployment backend.

    Each backend wraps a model (quantized, ONNX, TensorRT, distilled) and
    exposes a ``predict`` method that mirrors ``PreTrainedPolicy.select_action``.
    """

    @abc.abstractmethod
    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run inference on a batch and return action tensor."""
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset any internal state (e.g. action caches)."""
        ...

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        """Return the device the backend runs on."""
        ...

    @property
    @abc.abstractmethod
    def parameters(self) -> list[nn.Parameter]:
        """Return model parameters (empty for non-trainable backends)."""
        ...


# ---------------------------------------------------------------------------
# NativePyTorchBackend – wraps an existing nn.Module policy
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# IdentityBackend – passthrough, no transformation
# ---------------------------------------------------------------------------

class IdentityBackend(NativePyTorchBackend):
    """Identity backend – passes through to the original policy unchanged.

    Used by the ``edge_identity`` variant to prove the plugin hook works.
    """


# ---------------------------------------------------------------------------
# CompressedPolicy – the registered LeRobot policy
# ---------------------------------------------------------------------------

class CompressedPolicy(PreTrainedPolicy):
    """Wraps a ``DeploymentBackend`` behind LeRobot's policy interface.

    This class satisfies the abstract contract of ``PreTrainedPolicy`` by
    delegating ``select_action``, ``forward``, ``reset``, and
    ``predict_action_chunk`` to the underlying backend.

    Subclasses must define ``config_class`` and ``name`` class attributes
    (enforced by ``PreTrainedPolicy.__init_subclass__``).
    """

    # These must be set by concrete subclasses
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
        # If a backend is not provided yet, build one lazily from the config.
        # This path is hit when LeRobot's factory calls
        # ``policy_cls(config=cfg)`` without a pre-built backend.
        super().__init__(config)

        if backend is not None:
            self._backend = backend
        elif pretrained_name_or_path is not None:
            self._backend = self._build_backend_from_checkpoint(
                pretrained_name_or_path, config
            )
        else:
            # Default: create an empty backend placeholder (for registration testing)
            self._backend = _PlaceholderBackend(config.device or "cpu")

        self._action_cache: torch.Tensor | None = None
        self._cache_idx: int = 0

    # -- factory that concrete subclasses override to build their backend -----

    def _build_backend_from_checkpoint(
        self, path: str, config: EdgeBaseConfig
    ) -> DeploymentBackend:
        """Build a backend from a saved checkpoint.  Override in subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _build_backend_from_checkpoint"
        )

    # -- PreTrainedPolicy interface -------------------------------------------

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select a single action for the current timestep.

        If the backend produces an action chunk, cache subsequent actions.
        """
        if self._action_cache is not None and self._cache_idx < self._action_cache.shape[0]:
            action = self._action_cache[self._cache_idx]
            self._cache_idx += 1
            return action

        # No cache or cache exhausted – run inference
        self._action_cache = self.predict_action_chunk(batch)
        self._cache_idx = 1
        return self._action_cache[0]

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        """Return an action chunk (sequence of future actions)."""
        return self._backend.predict(batch)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        """Forward pass for training (returns loss, info).

        For compressed/deployed policies this is typically not used during
        training but must be implemented to satisfy the abstract interface.
        """
        actions = self._backend.predict(batch)
        # For a deployed policy, we can't compute a meaningful loss without
        # ground-truth targets.  Return a differentiable zero loss tied to
        # the model's parameters so that backward() works correctly.
        loss = sum(p.sum() * 0.0 for p in self.parameters())
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        return loss, {"actions": actions}

    def reset(self) -> None:
        """Reset internal state (clear action cache)."""
        self._action_cache = None
        self._cache_idx = 0
        self._backend.reset()

    def get_optim_params(self) -> dict[str, Any]:
        """Return parameters for the optimizer."""
        return {"params": self._backend.parameters}

    @property
    def device(self) -> torch.device:
        return self._backend.device


# ---------------------------------------------------------------------------
# Placeholder backend (for registration testing only)
# ---------------------------------------------------------------------------

class _PlaceholderBackend(DeploymentBackend):
    """A minimal backend that returns zeros – used for plugin registration tests."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self._device = torch.device(device)

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Find any action-shaped output in the batch to determine shape
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
