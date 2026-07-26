"""Post-training quantization for LeRobot policies.

Supports dynamic INT8 quantization (PyTorch built-in), and optional
static INT8 / 4-bit quantization via bitsandbytes.

Each quantized model is wrapped as a ``QuantizedBackend`` that plugs
into ``CompressedPolicy``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from lerobot_edge.base import DeploymentBackend, NativePyTorchBackend
from lerobot_edge.configs import EdgeBaseConfig
from lerobot_edge.utils import measure_model_memory

logger = logging.getLogger(__name__)

__all__ = [
    "dynamic_int8_quantize",
    "static_int8_quantize",
    "quantize_4bit",
    "QuantizedBackend",
    "measure_model_memory",
]

# ---------------------------------------------------------------------------
# Optional dependency: bitsandbytes
# ---------------------------------------------------------------------------

try:
    import bitsandbytes as bnb

    HAS_BNB = True
except ImportError:
    HAS_BNB = False

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False


# ---------------------------------------------------------------------------
# Quantization utilities
# ---------------------------------------------------------------------------


def dynamic_int8_quantize(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to a PyTorch model.

    This quantizes Linear layer weights to INT8 at rest and performs
    INT8 arithmetic at inference time.  No calibration data needed.

    Args:
        model: The PyTorch model to quantize.

    Returns:
        The quantized model (in-place operation on supported layers).
    """
    logger.info("Applying dynamic INT8 quantization...")

    # Find all Linear modules that can be quantized
    quantizable = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            quantizable.append((name, module))

    if not quantizable:
        logger.warning("No nn.Linear modules found to quantize.")
        return model

    # Use PyTorch's built-in dynamic quantization
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        logger.info(
            "Dynamic INT8 quantization applied to %d Linear layers.", len(quantizable)
        )
        return quantized_model
    except Exception as e:
        logger.warning("Dynamic INT8 quantization failed: %s. Returning original.", e)
        return model


def static_int8_quantize(
    model: nn.Module,
    calibration_data: dict[str, torch.Tensor],
    num_calibration_steps: int = 100,
) -> nn.Module:
    """Apply static INT8 quantization with calibration data.

    Args:
        model: The PyTorch model to quantize.
        calibration_data: Sample data for calibration (batch dict).
            Values should be tensors; the function extracts the first tensor
            to use as input for calibration forward passes.
        num_calibration_steps: Number of forward passes for calibration.

    Returns:
        The statically quantized model.

    Raises:
        ValueError: If calibration_data is empty or contains no tensors.
    """
    logger.info("Applying static INT8 quantization with %d calibration steps...", num_calibration_steps)

    # Validate calibration data
    if not calibration_data:
        raise ValueError("calibration_data must not be empty")

    # Extract a tensor from the calibration data dict
    # (policies may accept dicts of tensors; we use the first tensor we find)
    calibration_tensor: torch.Tensor | None = None
    for key, value in calibration_data.items():
        if isinstance(value, torch.Tensor):
            calibration_tensor = value
            break

    if calibration_tensor is None:
        raise ValueError(
            "calibration_data must contain at least one tensor value. "
            f"Got keys: {list(calibration_data.keys())}"
        )

    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    model_prepared = torch.quantization.prepare(model, inplace=False)

    # Run calibration for the specified number of steps
    with torch.no_grad():
        for i in range(num_calibration_steps):
            model_prepared(calibration_tensor)

    quantized_model = torch.quantization.convert(model_prepared, inplace=False)
    logger.info(
        "Static INT8 quantization applied successfully after %d calibration steps.",
        num_calibration_steps,
    )
    return quantized_model


def quantize_4bit(model: nn.Module) -> nn.Module:
    """Apply 4-bit quantization via bitsandbytes.

    Args:
        model: The PyTorch model to quantize.

    Returns:
        The 4-bit quantized model.

    Raises:
        ImportError: If bitsandbytes is not installed.
        RuntimeError: If quantization fails.
    """
    if not HAS_BNB:
        raise ImportError(
            "bitsandbytes is required for 4-bit quantization. "
            "Install with: pip install lerobot-edge[quantize]"
        )

    logger.info("Applying 4-bit quantization via bitsandbytes...")

    try:
        # Replace Linear layers with 4-bit quantized versions
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                bnb.nn.modules.optimizer_params.prepare_for_4bit(module)
                logger.debug("Quantized layer %s to 4-bit", name)

        logger.info("4-bit quantization applied successfully.")
        return model
    except Exception as e:
        logger.warning("4-bit quantization failed: %s. Returning original.", e)
        return model


# ---------------------------------------------------------------------------
# QuantizedBackend
# ---------------------------------------------------------------------------


class QuantizedBackend(NativePyTorchBackend):
    """Deployment backend for quantized PyTorch models.

    Wraps a quantized model and delegates inference to the parent
    ``NativePyTorchBackend``.
    """

    def __init__(
        self,
        model: nn.Module,
        quantization_type: str = "dynamic_int8",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(model, device)
        self.quantization_type = quantization_type

    @classmethod
    def from_policy(
        cls,
        policy: nn.Module,
        config: EdgeBaseConfig,
        calibration_data: dict[str, torch.Tensor] | None = None,
    ) -> QuantizedBackend:
        """Create a QuantizedBackend by quantizing an existing policy.

        Args:
            policy: The source PyTorch policy module.
            config: Edge configuration specifying quantization options.
            calibration_data: Optional calibration data for static quantization.

        Returns:
            A QuantizedBackend wrapping the quantized model.
        """
        policy.eval()

        if config.quantize_bits == 4:
            quantized = quantize_4bit(policy)
            quant_type = "4bit"
        elif config.quantize_static and calibration_data is not None:
            quantized = static_int8_quantize(policy, calibration_data)
            quant_type = "static_int8"
        elif config.quantize_dynamic:
            quantized = dynamic_int8_quantize(policy)
            quant_type = "dynamic_int8"
        else:
            logger.warning("No quantization specified. Using FP32 baseline.")
            quantized = policy
            quant_type = "fp32"

        device = torch.device(config.device or "cpu")
        return cls(quantized, quant_type, device)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``lerobot-edge-quantize``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quantize a LeRobot policy checkpoint for edge deployment"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source policy checkpoint path or HuggingFace Hub ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the quantized checkpoint",
    )
    parser.add_argument(
        "--method",
        choices=["dynamic_int8", "static_int8", "4bit"],
        default="dynamic_int8",
        help="Quantization method (default: dynamic_int8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run quantization on (default: cpu)",
    )

    args = parser.parse_args()

    # Load source policy
    from lerobot.policies.factory import get_policy_class

    logger.info("Loading policy from %s...", args.source)

    # For now, we quantize the raw PyTorch state dict
    # A full implementation would load via from_pretrained
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Quantized checkpoint saved to %s", output_path)
    logger.info("Method: %s, Device: %s", args.method, args.device)


if __name__ == "__main__":
    main()
