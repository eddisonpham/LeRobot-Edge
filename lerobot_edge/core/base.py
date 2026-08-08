"""CompressedPolicy wrapper and deployment backends."""

from __future__ import annotations

import abc
import copy
import logging
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from lerobot.policies.pretrained import PreTrainedPolicy

from lerobot_edge.core.configs import (
    EdgeBaseConfig,
    EdgeOnnxFp32Config,
    EdgeOnnxInt8Config,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DeploymentBackend",
    "NativePyTorchBackend",
    "IdentityBackend",
    "CompiledBackend",
    "CUDAGraphBackend",
    "CompressedPolicy",
]


class DeploymentBackend(abc.ABC):
    """Abstract interface for a deployment backend."""

    @abc.abstractmethod
    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor: ...

    @abc.abstractmethod
    def reset(self) -> None: ...

    @property
    @abc.abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abc.abstractmethod
    def parameters(self) -> list[nn.Parameter]: ...


class NativePyTorchBackend(DeploymentBackend):
    """Wraps a PyTorch policy as a backend."""

    def __init__(self, policy: nn.Module, device: torch.device | None = None) -> None:
        self._policy = policy
        self._device = device or next(policy.parameters()).device
        self._policy.eval()

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            return self._policy.select_action(batch)  # type: ignore[no-any-return,operator]

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()  # type: ignore[operator]

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def parameters(self) -> list[nn.Parameter]:
        return list(self._policy.parameters())


class IdentityBackend(NativePyTorchBackend):
    """Passthrough backend — no transformation applied."""


class _PredictWrapper(nn.Module):
    """nn.Module wrapper that delegates predict to a DeploymentBackend."""

    def __init__(self, backend: DeploymentBackend) -> None:
        super().__init__()
        self._backend = backend

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._backend.predict(batch)


class CompiledBackend(DeploymentBackend):
    """Wraps a backend with torch.compile for kernel fusion."""

    def __init__(
        self,
        backend: DeploymentBackend,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = False,
    ) -> None:
        if not hasattr(torch, "compile"):
            raise ImportError("torch.compile requires PyTorch >= 2.0")
        self._backend = backend
        self._mode = mode
        self._fullgraph = fullgraph
        self._dynamic = dynamic
        wrapper = _PredictWrapper(backend)
        self._compiled = torch.compile(
            wrapper,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        self._warmed_up = False
        logger.info(
            "CompiledBackend created (mode=%s, fullgraph=%s, dynamic=%s)",
            mode,
            fullgraph,
            dynamic,
        )

    def warmup(self, batch: dict[str, torch.Tensor], runs: int = 10) -> None:
        """Warm up the compiled function with representative inputs.

        This triggers the initial compilation and, for ``reduce-overhead``
        and ``max-autotune`` modes, captures CUDA graphs. Must be called
        before benchmarking.
        """
        if self._warmed_up:
            return
        for _ in range(runs):
            self._compiled(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._warmed_up = True
        logger.info("CompiledBackend warmed up (%d runs)", runs)

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._compiled(batch)  # type: ignore[no-any-return]

    def reset(self) -> None:
        self._backend.reset()
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        self._warmed_up = False

    @property
    def device(self) -> torch.device:
        return self._backend.device

    @property
    def parameters(self) -> list[nn.Parameter]:
        return self._backend.parameters


class CUDAGraphBackend(DeploymentBackend):
    """Ultra-low-latency backend via manual CUDA graph capture."""

    def __init__(
        self,
        backend: DeploymentBackend,
        pool_size: int = 1,
    ) -> None:
        if backend.device.type != "cuda":
            raise ValueError(f"CUDAGraphBackend requires a CUDA backend, got {backend.device.type}")
        self._backend = backend
        self._pool_size = pool_size
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_input: dict[str, torch.Tensor] | None = None
        self._static_output: torch.Tensor | None = None
        self._captured = False

    def capture(self, batch: dict[str, torch.Tensor]) -> None:
        """Capture CUDA graph. Input shapes frozen at capture time."""
        if self._captured:
            return

        # Move to CUDA and create persistent copies for graph capture
        static_input = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                static_input[key] = val.clone().to(self._backend.device)
            else:
                static_input[key] = val

        # Warm up CUDA allocator
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._backend.predict(static_input)
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_output = self._backend.predict(static_input)

        self._static_input = static_input
        self._captured = True
        logger.info(
            "CUDAGraphBackend captured graph (device=%s, pool_size=%d)",
            self._backend.device,
            self._pool_size,
        )

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self._captured or self._graph is None:
            # Auto-capture on first call
            self.capture(batch)

        assert self._static_input is not None
        assert self._static_output is not None
        assert self._graph is not None

        # Copy new data into static input tensors
        for key, static_val in self._static_input.items():
            if isinstance(static_val, torch.Tensor) and key in batch:
                src = batch[key]
                if isinstance(src, torch.Tensor):
                    static_val.copy_(src)

        self._graph.replay()
        return self._static_output.clone()

    def reset(self) -> None:
        self._backend.reset()
        self._graph = None
        self._static_input = None
        self._static_output = None
        self._captured = False

    @property
    def device(self) -> torch.device:
        return self._backend.device

    @property
    def parameters(self) -> list[nn.Parameter]:
        return self._backend.parameters


class CompressedPolicy(PreTrainedPolicy):
    """Wraps a DeploymentBackend behind the LeRobot policy interface."""

    config_class = EdgeBaseConfig
    name = "edge_compressed"

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
            self._backend = self._build_backend_from_checkpoint(pretrained_name_or_path, config)
        else:
            self._backend = _PlaceholderBackend(config.device or "cpu")

        self._action_cache: torch.Tensor | None = None
        self._cache_idx: int = 0

    def _build_backend_from_checkpoint(
        self, path: str, config: EdgeBaseConfig
    ) -> DeploymentBackend:
        device = torch.device(config.device or "cpu")
        source = path or config.source_pretrained_path
        if source is None:
            logger.warning("No source checkpoint provided. Using placeholder backend.")
            return _PlaceholderBackend(str(device))

        logger.info("Loading policy from %s", source)
        try:
            from lerobot_edge.core.utils import load_policy_from_checkpoint

            policy = load_policy_from_checkpoint(
                source, config.source_policy_type or "smolvla", str(device)
            )

            # Apply attention optimization (SDPA/FlashAttention)
            if getattr(config, "optimize_attention", True):
                from lerobot_edge.optimization import optimize_policy_for_inference

                policy = optimize_policy_for_inference(policy, enable_attention=True)
        except Exception as e:
            logger.error("Failed to load policy from %s: %s", source, e)
            return _PlaceholderBackend(str(device))

        try:
            if isinstance(config, EdgeOnnxInt8Config):
                from lerobot_edge.export.onnx import OnnxRuntimeBackend, export_policy_to_onnx

                onnx_path = export_policy_to_onnx(
                    policy,
                    config,
                    Path(tempfile.mkdtemp(prefix="lerobot_edge_onnx_")) / "model.onnx",
                )
                logger.info("Creating OnnxRuntimeBackend from %s", onnx_path)
                return OnnxRuntimeBackend(str(onnx_path), device=device)

            if isinstance(config, EdgeOnnxFp32Config):
                from lerobot_edge.export.onnx import OnnxRuntimeBackend, export_policy_to_onnx

                export_config = copy.deepcopy(config)
                export_config.quantize_dynamic = False
                onnx_path = export_policy_to_onnx(
                    policy,
                    export_config,
                    Path(tempfile.mkdtemp(prefix="lerobot_edge_onnx_")) / "model.onnx",
                )
                logger.info("Creating OnnxRuntimeBackend (FP32) from %s", onnx_path)
                return OnnxRuntimeBackend(str(onnx_path), device=device)

            if config.quantize_dynamic:
                from lerobot_edge.compression.quantize import QuantizedBackend

                logger.info("Creating QuantizedBackend (dynamic INT8)")
                return QuantizedBackend.from_policy(policy, config)

            logger.info("Creating IdentityBackend (no compression)")
            return IdentityBackend(policy, device)
        except Exception as e:
            logger.warning("Backend creation failed: %s. Falling back to IdentityBackend.", e)
            return IdentityBackend(policy, device)

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
    """Zeros-returning backend for registration tests."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self._device = torch.device(device)

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        for key, val in batch.items():
            if "action" in key.lower():
                return torch.zeros(val.shape, device=self._device)
        return torch.zeros(1, device=self._device)

    def reset(self) -> None:
        return

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def parameters(self) -> list[nn.Parameter]:
        return []
