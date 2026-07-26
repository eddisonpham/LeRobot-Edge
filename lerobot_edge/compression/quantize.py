"""Post-training quantization for LeRobot policies."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from lerobot_edge.core.base import NativePyTorchBackend
from lerobot_edge.core.configs import EdgeBaseConfig
from lerobot_edge.core.utils import measure_model_memory

logger = logging.getLogger(__name__)

__all__ = [
    "dynamic_int8_quantize",
    "static_int8_quantize",
    "quantize_4bit",
    "quantize_bnb_int8",
    "quantize_bnb_fp4",
    "QuantizedBackend",
]

try:
    from torchao.core.config import AOBaseConfig
    from torchao.dtypes import to_affine_quantized_intx_static
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig
    from torchao.quantization import quantize_ as torchao_quantize
    from torchao.quantization.granularity import PerAxis, PerTensor
    from torchao.quantization.observer import AffineQuantizedMinMaxObserver
    from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
    from torchao.quantization.quant_primitives import MappingType
    from torchao.quantization.transform_module import register_quantize_module_handler

    HAS_TORCHAO = True
except ImportError:
    HAS_TORCHAO = False

try:
    import bitsandbytes as bnb  # noqa: F401

    HAS_BNB = True
except ImportError:
    HAS_BNB = False


if HAS_TORCHAO:

    class ObservedLinear(nn.Module):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            act_obs: nn.Module,
            weight_obs: nn.Module,
            bias: bool = True,
            device=None,
            dtype=None,
        ):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias, device, dtype)
            self.act_obs = act_obs
            self.weight_obs = weight_obs

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            observed_input = self.act_obs(x)
            observed_weight = self.weight_obs(self.linear.weight)
            return F.linear(observed_input, observed_weight, self.linear.bias)

        @classmethod
        def from_float(
            cls, float_linear: nn.Linear, act_obs: nn.Module, weight_obs: nn.Module
        ) -> ObservedLinear:
            observed_linear = cls(
                float_linear.in_features,
                float_linear.out_features,
                act_obs,
                weight_obs,
                float_linear.bias is not None,
                device=float_linear.weight.device,
                dtype=float_linear.weight.dtype,
            )
            observed_linear.linear.weight = float_linear.weight
            if float_linear.bias is not None:
                observed_linear.linear.bias = float_linear.bias
            return observed_linear

    class QuantizedLinear(nn.Module):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            act_obs: nn.Module,
            weight_obs: nn.Module,
            weight: torch.Tensor,
            bias: torch.Tensor | None,
            target_dtype: torch.dtype,
        ):
            super().__init__()
            self.act_scale, self.act_zero_point = act_obs.calculate_qparams()
            weight_scale, weight_zero_point = weight_obs.calculate_qparams()
            assert weight.dim() == 2
            block_size = (1, weight.shape[1])
            self.target_dtype = target_dtype
            self.bias = bias
            self.qweight = to_affine_quantized_intx_static(
                weight, weight_scale, weight_zero_point, block_size, self.target_dtype
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            block_size = (1,) + x.shape[1:]
            qinput = to_affine_quantized_intx_static(
                x,
                self.act_scale,
                self.act_zero_point,
                block_size,
                self.target_dtype,
            )
            return F.linear(qinput, self.qweight, self.bias)

        @classmethod
        def from_observed(
            cls, observed_linear: ObservedLinear, target_dtype: torch.dtype
        ) -> QuantizedLinear:
            return cls(
                observed_linear.linear.in_features,
                observed_linear.linear.out_features,
                observed_linear.act_obs,
                observed_linear.weight_obs,
                observed_linear.linear.weight,
                observed_linear.linear.bias,
                target_dtype,
            )

    @dataclass
    class StaticQuantConfig(AOBaseConfig):
        target_dtype: torch.dtype = torch.uint8

    @register_quantize_module_handler(StaticQuantConfig)
    def _apply_static_quant(module: nn.Module, config: StaticQuantConfig) -> QuantizedLinear:
        return QuantizedLinear.from_observed(module, config.target_dtype)

    def _insert_observers(model: nn.Module, act_obs: nn.Module, weight_obs: nn.Module) -> None:
        def _is_linear(m: nn.Module, fqn: str) -> bool:
            return isinstance(m, nn.Linear)

        def replacement_fn(m: nn.Module) -> ObservedLinear:
            return ObservedLinear.from_float(m, copy.deepcopy(act_obs), copy.deepcopy(weight_obs))

        _replace_with_custom_fn_if_matches_filter(model, replacement_fn, _is_linear)


def dynamic_int8_quantize(model: nn.Module) -> nn.Module:
    quantizable = [m for _, m in model.named_modules() if isinstance(m, nn.Linear)]
    if not quantizable:
        logger.warning("No nn.Linear modules found to quantize.")
        return model

    if HAS_TORCHAO:
        try:
            config = Int8DynamicActivationInt8WeightConfig()
            torchao_quantize(model, config)
            logger.info("Dynamic INT8 quantization applied to %d Linear layers.", len(quantizable))
            return model
        except Exception as e:
            logger.warning("Dynamic INT8 quantization failed: %s. Returning original.", e)
            return model

    raise ImportError(
        "torchao is required for dynamic INT8 quantization. Install with: pip install torchao"
    )


def static_int8_quantize(
    model: nn.Module,
    calibration_data: dict[str, torch.Tensor],
    num_calibration_steps: int = 100,
) -> nn.Module:
    if not calibration_data:
        raise ValueError("calibration_data must not be empty")

    tensors = [v for v in calibration_data.values() if isinstance(v, torch.Tensor)]
    if not tensors:
        raise ValueError(
            "calibration_data must contain at least one tensor value. "
            f"Got keys: {list(calibration_data.keys())}"
        )

    model.eval()

    if HAS_TORCHAO:
        try:
            act_obs = AffineQuantizedMinMaxObserver(
                MappingType.ASYMMETRIC,
                torch.uint8,
                granularity=PerTensor(),
                eps=torch.finfo(torch.float32).eps,
                scale_dtype=torch.float32,
                zero_point_dtype=torch.float32,
            )
            weight_obs = AffineQuantizedMinMaxObserver(
                MappingType.ASYMMETRIC,
                torch.uint8,
                granularity=PerAxis(axis=0),
                eps=torch.finfo(torch.float32).eps,
                scale_dtype=torch.float32,
                zero_point_dtype=torch.float32,
            )
            _insert_observers(model, act_obs, weight_obs)

            logger.info("Calibrating with %d steps...", num_calibration_steps)
            with torch.no_grad():
                for _ in range(num_calibration_steps):
                    model(*tensors)

            def _is_observed(m: nn.Module, _fqn: str) -> bool:
                return isinstance(m, ObservedLinear)

            torchao_quantize(model, StaticQuantConfig(torch.uint8), _is_observed)
            logger.info("Static INT8 quantization applied successfully.")
            return model
        except Exception as e:
            raise RuntimeError(f"torchao static quantization failed: {e}") from e

    raise ImportError(
        "torchao is required for static INT8 quantization. Install with: pip install torchao"
    )


def quantize_4bit(model: nn.Module) -> nn.Module:
    if not HAS_BNB:
        raise ImportError(
            "bitsandbytes is required for 4-bit quantization. Install with: pip install lerobot-edge[quantize]"
        )

    logger.info("Applying 4-bit quantization via bitsandbytes...")

    try:
        from bitsandbytes.nn.modules import Linear4bit
    except ImportError as e:
        raise ImportError(
            "bitsandbytes version too old: Linear4bit not found. Upgrade with: pip install --upgrade bitsandbytes"
        ) from e

    try:
        linear_layers = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        ]
        if not linear_layers:
            logger.warning("No nn.Linear modules found to quantize.")
            return model

        modules_dict = dict(model.named_modules())
        for name, module in linear_layers:
            parent_name, _, child_name = name.rpartition(".")
            parent = model if not parent_name else modules_dict[parent_name]
            new_layer = Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=module.weight.dtype,
                compress_statistics=True,
                quant_type="nf4",
            )
            new_layer.weight = nn.Parameter(module.weight, requires_grad=False)
            if module.bias is not None:
                new_layer.bias = nn.Parameter(module.bias, requires_grad=False)
            setattr(parent, child_name, new_layer)
            logger.debug("Replaced layer %s with Linear4bit", name)

        logger.info("4-bit quantization applied to %d Linear layers.", len(linear_layers))
        return model
    except ImportError:
        raise
    except Exception as e:
        logger.warning("4-bit quantization failed: %s. Returning original.", e)
        return model


def quantize_bnb_int8(model: nn.Module) -> nn.Module:
    if not HAS_BNB:
        raise ImportError(
            "bitsandbytes is required for INT8 quantization. Install with: pip install lerobot-edge[quantize]"
        )

    try:
        from bitsandbytes.nn import Linear8bitLt
    except ImportError as e:
        raise ImportError(
            "bitsandbytes version too old: Linear8bitLt not found. Upgrade with: pip install --upgrade bitsandbytes"
        ) from e

    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    if not linear_layers:
        logger.warning("No nn.Linear modules found to quantize.")
        return model

    modules_dict = dict(model.named_modules())
    for name, module in linear_layers:
        parent_name, _, child_name = name.rpartition(".")
        parent = model if not parent_name else modules_dict[parent_name]
        new_layer = Linear8bitLt(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            has_fp16_weights=True,
            threshold=6.0,
        )
        new_layer.weight = nn.Parameter(module.weight.float(), requires_grad=False)
        if module.bias is not None:
            new_layer.bias = nn.Parameter(module.bias.float(), requires_grad=False)
        setattr(parent, child_name, new_layer)

    logger.info("bitsandbytes INT8 quantization applied to %d Linear layers.", len(linear_layers))
    return model


def quantize_bnb_fp4(model: nn.Module) -> nn.Module:
    if not HAS_BNB:
        raise ImportError(
            "bitsandbytes is required for FP4 quantization. Install with: pip install lerobot-edge[quantize]"
        )

    from bitsandbytes.nn.modules import Linear4bit

    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    if not linear_layers:
        logger.warning("No nn.Linear modules found to quantize.")
        return model

    modules_dict = dict(model.named_modules())
    for name, module in linear_layers:
        parent_name, _, child_name = name.rpartition(".")
        parent = model if not parent_name else modules_dict[parent_name]
        new_layer = Linear4bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            compute_dtype=module.weight.dtype,
            compress_statistics=True,
            quant_type="fp4",
        )
        new_layer.weight = nn.Parameter(module.weight, requires_grad=False)
        if module.bias is not None:
            new_layer.bias = nn.Parameter(module.bias, requires_grad=False)
        setattr(parent, child_name, new_layer)

    logger.info("bitsandbytes FP4 quantization applied to %d Linear layers.", len(linear_layers))
    return model


class QuantizedBackend(NativePyTorchBackend):
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
        policy.eval()

        config_type = getattr(config, "type", "")

        if config_type == "edge_quant_bnb_int8":
            quantized = quantize_bnb_int8(policy)
            quant_type = "bnb_int8"
        elif config_type == "edge_quant_bnb_fp4":
            quantized = quantize_bnb_fp4(policy)
            quant_type = "bnb_fp4"
        elif config_type == "edge_quant_bnb_nf4":
            quantized = quantize_4bit(policy)
            quant_type = "nf4"
        elif config.quantize_bits == 4:
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
    import argparse
    import json
    from pathlib import Path

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
        "--output", type=str, required=True, help="Output directory for the quantized checkpoint"
    )
    parser.add_argument(
        "--method",
        choices=["dynamic_int8", "static_int8", "4bit", "bnb_int8", "nf4", "bnb_fp4"],
        default="dynamic_int8",
        help="Quantization method (default: dynamic_int8)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run quantization on (default: cpu)"
    )
    args = parser.parse_args()

    logger.info("Loading policy from %s...", args.source)

    from lerobot_edge.core.utils import load_policy_from_checkpoint

    try:
        policy = load_policy_from_checkpoint(args.source, "smolvla", args.device)
    except Exception as e:
        logger.error("Failed to load policy: %s", e)
        return

    original_mem = measure_model_memory(policy)
    logger.info(
        "Original model: %.1f MB, %d parameters",
        original_mem["total_mb"],
        original_mem["num_parameters"],
    )

    logger.info("Applying %s quantization...", args.method)
    if args.method == "dynamic_int8":
        quantized = dynamic_int8_quantize(policy)
    elif args.method == "static_int8":
        logger.warning("Static INT8 requires calibration data. Using dummy calibration.")
        calibration_data = {"observation.state": torch.randn(1, 2)}
        quantized = static_int8_quantize(policy, calibration_data)
    elif args.method == "4bit":
        quantized = quantize_4bit(policy)
    elif args.method == "bnb_int8":
        quantized = quantize_bnb_int8(policy)
    elif args.method == "nf4":
        quantized = quantize_4bit(policy)
    elif args.method == "bnb_fp4":
        quantized = quantize_bnb_fp4(policy)
    else:
        logger.error("Unknown method: %s", args.method)
        return

    quantized_mem = measure_model_memory(quantized)
    reduction = (1 - quantized_mem["total_mb"] / original_mem["total_mb"]) * 100
    logger.info(
        "Quantized model: %.1f MB, %d parameters (%.1f%% reduction)",
        quantized_mem["total_mb"],
        quantized_mem["num_parameters"],
        reduction,
    )

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
