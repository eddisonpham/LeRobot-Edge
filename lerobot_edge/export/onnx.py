"""ONNX export and ONNX Runtime inference for LeRobot policies."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from lerobot_edge.core.base import DeploymentBackend
from lerobot_edge.core.configs import EdgeBaseConfig

logger = logging.getLogger(__name__)

__all__ = [
    "export_policy_to_onnx",
    "OnnxRuntimeBackend",
    "validate_onnx_model",
]

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    import onnx

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


def export_policy_to_onnx(
    policy: nn.Module,
    config: EdgeBaseConfig,
    output_path: str | Path,
    *,
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
) -> Path:
    """Export a LeRobot policy to ONNX format."""
    if not HAS_ONNX:
        raise ImportError("onnx package is required. Install with: pip install lerobot-edge[onnx]")

    if not HAS_ORT:
        raise ImportError("onnxruntime is required. Install with: pip install lerobot-edge[onnx]")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting policy to ONNX (opset=%d)...", opset_version)

    policy.eval()
    device = torch.device(config.device or "cpu")

    dummy_inputs = _build_dummy_inputs(policy, config, device)

    if dynamic_axes is None:
        dynamic_axes = _infer_dynamic_axes(policy, config)

    if input_names is None:
        input_names = list(dummy_inputs.keys())
    if output_names is None:
        output_names = ["actions"]

    try:
        torch.onnx.export(
            policy,
            tuple(dummy_inputs.values()),
            str(output_path),
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        logger.info("ONNX model exported successfully to %s", output_path)
        return output_path

    except Exception as e:
        logger.error("ONNX export failed: %s", e)
        raise


def _build_dummy_inputs(
    policy: nn.Module,
    config: EdgeBaseConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build dummy inputs for ONNX export."""
    dummy_inputs = {}

    if config.input_features:
        for name, feature in config.input_features.items():
            shape = list(feature.shape) if hasattr(feature, "shape") else [1, 3, 224, 224]
            if len(shape) == 0:
                shape = [1]
            elif shape[0] != 1:
                shape.insert(0, 1)
            dummy_inputs[name] = torch.randn(shape, device=device)
    else:
        dummy_inputs["observation.images"] = torch.randn(1, 3, 224, 224, device=device)
        dummy_inputs["observation.state"] = torch.randn(1, 7, device=device)

    return dummy_inputs


def _infer_dynamic_axes(
    policy: nn.Module,
    config: EdgeBaseConfig,
) -> dict[str, dict[int, str]]:
    """Infer dynamic axes from the policy's input/output features."""
    dynamic_axes: dict[str, dict[int, str]] = {}

    if config.input_features:
        for name in config.input_features:
            dynamic_axes[name] = {0: "batch"}
    else:
        dynamic_axes["observation.images"] = {0: "batch"}
        dynamic_axes["observation.state"] = {0: "batch"}

    dynamic_axes["actions"] = {0: "batch"}

    return dynamic_axes


class OnnxRuntimeBackend(DeploymentBackend):
    """Deployment backend using ONNX Runtime for inference."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        provider: str = "CPUExecutionProvider",
        device: torch.device | None = None,
    ) -> None:
        if not HAS_ORT:
            raise ImportError(
                "onnxruntime is required. Install with: pip install lerobot-edge[onnx]"
            )

        self._model_path = Path(model_path)
        self._provider = provider
        self._device = device or torch.device("cpu")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(self._model_path),
            sess_options=sess_options,
            providers=[provider],
        )

        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

        logger.info(
            "ONNX Runtime session created: %d inputs, %d outputs, provider=%s",
            len(self._input_names),
            len(self._output_names),
            provider,
        )

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run inference via ONNX Runtime."""
        ort_inputs = {}
        for name in self._input_names:
            if name in batch:
                tensor = batch[name]
                if isinstance(tensor, torch.Tensor):
                    ort_inputs[name] = tensor.detach().cpu().numpy()
                else:
                    ort_inputs[name] = tensor
            else:
                raise ValueError(
                    f"Missing required input '{name}' in batch. "
                    f"Available keys: {list(batch.keys())}"
                )

        ort_outputs = self._session.run(self._output_names, ort_inputs)

        if ort_outputs:
            return torch.from_numpy(ort_outputs[0])
        return torch.zeros(1, device=self._device)

    def reset(self) -> None:
        return

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def parameters(self) -> list[nn.Parameter]:
        return []

    @property
    def input_names(self) -> list[str]:
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        return self._output_names


def validate_onnx_model(model_path: str | Path) -> dict[str, Any]:
    """Validate an ONNX model and return metadata."""
    if not HAS_ONNX:
        return {"valid": False, "error": "onnx package not installed"}

    try:
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)

        return {
            "valid": True,
            "opset_version": model.opset_import[0].version if model.opset_import else None,
            "inputs": [
                {"name": inp.name, "shape": [d.dim_value for d in inp.type.tensor_type.shape.dim]}
                for inp in model.graph.input
            ],
            "outputs": [
                {"name": out.name, "shape": [d.dim_value for d in out.type.tensor_type.shape.dim]}
                for out in model.graph.output
            ],
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}
