"""Post-training quantization for LeRobot policies."""

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
]

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


def dynamic_int8_quantize(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to a PyTorch model."""
    logger.info("Applying dynamic INT8 quantization...")

    quantizable = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            quantizable.append((name, module))

    if not quantizable:
        logger.warning("No nn.Linear modules found to quantize.")
        return model

    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8,
        )
        logger.info("Dynamic INT8 quantization applied to %d Linear layers.", len(quantizable))
        return quantized_model
    except Exception as e:
        logger.warning("Dynamic INT8 quantization failed: %s. Returning original.", e)
        return model


def static_int8_quantize(
    model: nn.Module,
    calibration_data: dict[str, torch.Tensor],
    num_calibration_steps: int = 100,
) -> nn.Module:
    """Apply static INT8 quantization with calibration data."""
    logger.info("Applying static INT8 quantization with %d calibration steps...", num_calibration_steps)

    if not calibration_data:
        raise ValueError("calibration_data must not be empty")

    tensors = [v for v in calibration_data.values() if isinstance(v, torch.Tensor)]
    if not tensors:
        raise ValueError(
            "calibration_data must contain at least one tensor value. "
            f"Got keys: {list(calibration_data.keys())}"
        )

    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    model_prepared = torch.quantization.prepare(model, inplace=False)

    with torch.no_grad():
        for i in range(num_calibration_steps):
            model_prepared(*tensors)

    quantized_model = torch.quantization.convert(model_prepared, inplace=False)
    logger.info("Static INT8 quantization applied successfully after %d calibration steps.", num_calibration_steps)
    return quantized_model


def quantize_4bit(model: nn.Module) -> nn.Module:
    """Apply 4-bit quantization via bitsandbytes."""
    if not HAS_BNB:
        raise ImportError("bitsandbytes is required for 4-bit quantization. Install with: pip install lerobot-edge[quantize]")

    logger.info("Applying 4-bit quantization via bitsandbytes...")

    try:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                bnb.nn.modules.optimizer_params.prepare_for_4bit(module)
                logger.debug("Quantized layer %s to 4-bit", name)

        logger.info("4-bit quantization applied successfully.")
        return model
    except Exception as e:
        logger.warning("4-bit quantization failed: %s. Returning original.", e)
        return model


class QuantizedBackend(NativePyTorchBackend):
    """Deployment backend for quantized PyTorch models."""

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
        """Create a QuantizedBackend by quantizing an existing policy."""
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


def main() -> None:
    """CLI entry point for ``lerobot-edge-quantize``."""
    import argparse
    from pathlib import Path
    import json

    parser = argparse.ArgumentParser(description="Quantize a LeRobot policy checkpoint for edge deployment")
    parser.add_argument("--source", type=str, required=True, help="Source policy checkpoint path or HuggingFace Hub ID")
    parser.add_argument("--output", type=str, required=True, help="Output directory for the quantized checkpoint")
    parser.add_argument("--method", choices=["dynamic_int8", "static_int8", "4bit"], default="dynamic_int8", help="Quantization method (default: dynamic_int8)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run quantization on (default: cpu)")
    args = parser.parse_args()

    logger.info("Loading policy from %s...", args.source)

    from lerobot.policies.factory import make_policy, make_policy_config

    config = make_policy_config("smolvla")
    config.pretrained_path = args.source
    config.device = args.device

    try:
        policy = make_policy(config)
        policy.eval()
    except Exception as e:
        logger.error("Failed to load policy: %s", e)
        return

    original_mem = measure_model_memory(policy)
    logger.info("Original model: %.1f MB, %d parameters", original_mem["total_mb"], original_mem["num_parameters"])

    logger.info("Applying %s quantization...", args.method)
    if args.method == "dynamic_int8":
        quantized = dynamic_int8_quantize(policy)
    elif args.method == "static_int8":
        logger.warning("Static INT8 requires calibration data. Using dummy calibration.")
        calibration_data = {"observation.state": torch.randn(1, 2)}
        quantized = static_int8_quantize(policy, calibration_data)
    elif args.method == "4bit":
        quantized = quantize_4bit(policy)
    else:
        logger.error("Unknown method: %s", args.method)
        return

    quantized_mem = measure_model_memory(quantized)
    reduction = (1 - quantized_mem["total_mb"] / original_mem["total_mb"]) * 100
    logger.info("Quantized model: %.1f MB, %d parameters (%.1f%% reduction)", quantized_mem["total_mb"], quantized_mem["num_parameters"], reduction)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        torch.save(quantized.state_dict(), output_path / "model.pt")
        config_dict = {
            "type": "smolvla",
            "source": args.source,
            "device": args.device,
            "quantization": args.method,
            "original_mb": original_mem["total_mb"],
            "quantized_mb": quantized_mem["total_mb"],
        }
        with open(output_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info("Quantized checkpoint saved to %s", output_path)
    except Exception as e:
        logger.error("Failed to save checkpoint: %s", e)

    print("\n" + "=" * 60)
    print("QUANTIZATION RESULTS")
    print("=" * 60)
    print(f"Method:      {args.method}")
    print(f"Original:    {original_mem['total_mb']:.1f} MB")
    print(f"Quantized:   {quantized_mem['total_mb']:.1f} MB")
    print(f"Reduction:   {reduction:.1f}%")
    print(f"Saved to:    {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
