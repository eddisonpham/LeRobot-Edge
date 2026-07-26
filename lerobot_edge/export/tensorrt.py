"""TensorRT export for LeRobot policies (optional, GPU only)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from lerobot_edge.core.base import DeploymentBackend
from lerobot_edge.core.configs import EdgeBaseConfig

logger = logging.getLogger(__name__)

__all__ = [
    "HAS_TENSORRT",
    "export_onnx_to_tensorrt",
    "TensorRTBackend",
    "get_tensorrt_info",
]

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


def export_onnx_to_tensorrt(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    fp16: bool = True,
    int8: bool = False,
    max_batch_size: int = 8,
    workspace_size: int = 1 << 30,
) -> Path:
    """Convert an ONNX model to a TensorRT engine."""
    if not HAS_TENSORRT:
        raise ImportError("TensorRT is required. Install with: pip install lerobot-edge[tensorrt]")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building TensorRT engine from %s...", onnx_path)

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    with open(str(onnx_path), "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("ONNX parse error: %s", parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 mode enabled")

    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        logger.info("INT8 mode enabled")

    logger.info("Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed")

    with open(str(output_path), "wb") as f:
        f.write(serialized_engine)

    logger.info("TensorRT engine saved to %s", output_path)
    return output_path


class TensorRTBackend(DeploymentBackend):
    """Deployment backend using TensorRT for inference."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device: torch.device | None = None,
    ) -> None:
        if not HAS_TENSORRT:
            raise ImportError("TensorRT is required. Install with: pip install lerobot-edge[tensorrt]")

        self._engine_path = Path(engine_path)
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
            self._cuda = cuda
        except ImportError:
            raise ImportError("pycuda is required for TensorRT inference. Install with: pip install pycuda")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(str(self._engine_path), "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()
        self._stream = self._cuda.Stream()

        logger.info("TensorRT engine loaded from %s", self._engine_path)

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cuda = self._cuda
        device_allocs: list[int] = []

        try:
            for i in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(i)
                mode = self._engine.get_tensor_mode(name)
                dtype_trt = self._engine.get_tensor_dtype(name)
                dtype_np = trt.nptype(dtype_trt)

                if mode == trt.TensorIOMode.INPUT:
                    if name not in batch:
                        raise ValueError(
                            f"Missing required input '{name}' in batch. "
                            f"Available keys: {list(batch.keys())}"
                        )
                    tensor = batch[name]
                    host_data = tensor.detach().cpu().numpy().astype(dtype_np)
                    shape = tuple(host_data.shape)
                    self._context.set_input_shape(name, shape)
                    device_mem = cuda.mem_alloc(host_data.nbytes)
                    device_allocs.append(int(device_mem))
                    self._context.set_tensor_address(name, int(device_mem))
                    cuda.memcpy_htod_async(device_mem, host_data, self._stream)
                else:
                    shape = tuple(self._context.get_tensor_shape(name))
                    host_data = np.empty(shape, dtype=dtype_np)
                    device_mem = cuda.mem_alloc(host_data.nbytes)
                    device_allocs.append(int(device_mem))
                    self._context.set_tensor_address(name, int(device_mem))

            self._context.execute_async_v3(stream_handle=self._stream.handle)

            results: dict[str, np.ndarray] = {}
            for i in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(i)
                mode = self._engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.OUTPUT:
                    dtype_np = trt.nptype(self._engine.get_tensor_dtype(name))
                    shape = tuple(self._context.get_tensor_shape(name))
                    host_data = np.empty(shape, dtype=dtype_np)
                    cuda.memcpy_dtoh_async(host_data, self._context.get_tensor_address(name), self._stream)
                    results[name] = host_data

            self._stream.synchronize()

            output_name = next(iter(results))
            return torch.from_numpy(results[output_name]).to(self._device)

        finally:
            for ptr in device_allocs:
                cuda.mem_free(ptr)

    def reset(self) -> None:
        return

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def parameters(self) -> list[nn.Parameter]:
        return []


def get_tensorrt_info() -> dict[str, Any]:
    """Get information about the TensorRT installation."""
    if not HAS_TENSORRT:
        return {"available": False, "error": "TensorRT not installed"}

    return {
        "available": True,
        "version": trt.__version__,
    }
