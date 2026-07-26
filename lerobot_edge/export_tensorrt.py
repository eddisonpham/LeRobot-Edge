"""TensorRT export for LeRobot policies (optional, cloud/edge GPU only).

This module is gated behind an optional dependency check so the rest of
the repo remains installable and usable without it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from lerobot_edge.base import DeploymentBackend
from lerobot_edge.configs import EdgeBaseConfig

logger = logging.getLogger(__name__)

__all__ = [
    "HAS_TENSORRT",
    "export_onnx_to_tensorrt",
    "TensorRTBackend",
    "get_tensorrt_info",
]

# ---------------------------------------------------------------------------
# Optional dependency: TensorRT
# ---------------------------------------------------------------------------

try:
    import tensorrt as trt

    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False


# ---------------------------------------------------------------------------
# TensorRT export
# ---------------------------------------------------------------------------


def export_onnx_to_tensorrt(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    fp16: bool = True,
    int8: bool = False,
    max_batch_size: int = 8,
    workspace_size: int = 1 << 30,  # 1 GB
) -> Path:
    """Convert an ONNX model to a TensorRT engine.

    Args:
        onnx_path: Path to the input ONNX model.
        output_path: Path to save the TensorRT engine.
        fp16: Enable FP16 precision mode.
        int8: Enable INT8 precision mode (requires calibration).
        max_batch_size: Maximum batch size for dynamic batching.
        workspace_size: Maximum GPU memory for workspace (bytes).

    Returns:
        Path to the saved TensorRT engine.

    Raises:
        ImportError: If TensorRT is not installed.
        RuntimeError: If export fails.
    """
    if not HAS_TENSORRT:
        raise ImportError(
            "TensorRT is required for this operation. "
            "Install with: pip install lerobot-edge[tensorrt]"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building TensorRT engine from %s...", onnx_path)

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)

    # Parse ONNX model
    with open(str(onnx_path), "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("ONNX parse error: %s", parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 mode enabled")

    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        logger.info("INT8 mode enabled (calibration required)")

    # Build engine
    logger.info("Building TensorRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed")

    # Save engine
    with open(str(output_path), "wb") as f:
        f.write(serialized_engine)

    logger.info("TensorRT engine saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# TensorRT Runtime Backend
# ---------------------------------------------------------------------------


class TensorRTBackend(DeploymentBackend):
    """Deployment backend using TensorRT for inference.

    Wraps a TensorRT engine for high-performance GPU inference.
    """

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device: torch.device | None = None,
    ) -> None:
        if not HAS_TENSORRT:
            raise ImportError(
                "TensorRT is required. Install with: pip install lerobot-edge[tensorrt]"
            )

        self._engine_path = Path(engine_path)
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load and deserialize engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(str(self._engine_path), "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()

        logger.info("TensorRT engine loaded from %s", self._engine_path)

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run inference via TensorRT."""
        raise NotImplementedError(
            "TensorRTBackend.predict requires a fully initialized TensorRT engine "
            "with proper I/O bindings. This is a placeholder for future implementation."
        )

    def reset(self) -> None:
        """No state to reset for TensorRT."""
        pass

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def parameters(self) -> list[nn.Parameter]:
        """TensorRT engines don't have trainable parameters."""
        return []


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_tensorrt_info() -> dict[str, Any]:
    """Get information about the TensorRT installation."""
    if not HAS_TENSORRT:
        return {"available": False, "error": "TensorRT not installed"}

    return {
        "available": True,
        "version": trt.__version__,
        "cuda_version": trt.cuda_engine.get_cuda_device_capability() if hasattr(trt, 'cuda_engine') else None,
    }
