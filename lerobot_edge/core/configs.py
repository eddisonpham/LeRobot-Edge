"""Configuration dataclasses for lerobot_edge policy variants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import PreTrainedConfig

__all__ = [
    "EdgeBaseConfig",
    "EdgeIdentityConfig",
    "EdgeQuantInt8Config",
    "EdgeQuantBnbInt8Config",
    "EdgeQuantBnbNf4Config",
    "EdgeQuantBnbFp4Config",
    "EdgeOnnxFp32Config",
    "EdgeOnnxInt8Config",
    "EdgeDistilledConfig",
    "EdgeDistilledOnnxInt8Config",
]


@dataclass
class EdgeBaseConfig(PreTrainedConfig):
    """Base configuration for all lerobot_edge variants."""

    source_pretrained_path: str | None = None
    source_policy_type: str = "smolvla"
    deploy_device: str | None = None

    # Quantization
    quantize_dynamic: bool = True
    quantize_static: bool = False
    quantize_bits: int = 8

    # ONNX
    onnx_opset: int = 17
    onnx_input_names: list[str] = field(default_factory=list)
    onnx_output_names: list[str] = field(default_factory=list)
    onnx_dynamic_axes: dict[str, dict[int, str]] = field(default_factory=dict)

    # Distillation
    distill_epochs: int = 10
    distill_lr: float = 1e-4
    distill_temperature: float = 2.0
    distill_alpha: float = 0.5

    # Benchmark
    benchmark_warmup: int = 10
    benchmark_num_runs: int = 100

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def get_optimizer_preset(self) -> Any:
        return {"optimizer_cls": "AdamW", "lr": 1e-4, "weight_decay": 1e-4}

    def get_scheduler_preset(self) -> Any | None:
        return None

    def validate_features(self) -> None:
        return


@PreTrainedConfig.register_subclass("edge_identity")
@dataclass
class EdgeIdentityConfig(EdgeBaseConfig):
    """Passthrough wrapper — no compression."""

    type: str = "edge_identity"


@PreTrainedConfig.register_subclass("edge_quant_int8")
@dataclass
class EdgeQuantInt8Config(EdgeBaseConfig):
    """Dynamic INT8 quantized variant."""

    type: str = "edge_quant_int8"
    quantize_dynamic: bool = True
    quantize_bits: int = 8


@PreTrainedConfig.register_subclass("edge_quant_bnb_int8")
@dataclass
class EdgeQuantBnbInt8Config(EdgeBaseConfig):
    """bitsandbytes INT8 quantized variant (Linear8bitLt)."""

    type: str = "edge_quant_bnb_int8"
    quantize_dynamic: bool = False
    quantize_bits: int = 8


@PreTrainedConfig.register_subclass("edge_quant_bnb_nf4")
@dataclass
class EdgeQuantBnbNf4Config(EdgeBaseConfig):
    """bitsandbytes NF4 4-bit quantized variant."""

    type: str = "edge_quant_bnb_nf4"
    quantize_dynamic: bool = False
    quantize_bits: int = 4


@PreTrainedConfig.register_subclass("edge_quant_bnb_fp4")
@dataclass
class EdgeQuantBnbFp4Config(EdgeBaseConfig):
    """bitsandbytes FP4 4-bit quantized variant."""

    type: str = "edge_quant_bnb_fp4"
    quantize_dynamic: bool = False
    quantize_bits: int = 4


@PreTrainedConfig.register_subclass("edge_onnx_fp32")
@dataclass
class EdgeOnnxFp32Config(EdgeBaseConfig):
    """ONNX Runtime inference with FP32 weights."""

    type: str = "edge_onnx_fp32"
    onnx_opset: int = 17


@PreTrainedConfig.register_subclass("edge_onnx_int8")
@dataclass
class EdgeOnnxInt8Config(EdgeBaseConfig):
    """ONNX Runtime inference with INT8 quantized weights."""

    type: str = "edge_onnx_int8"
    onnx_opset: int = 17
    quantize_dynamic: bool = True


@PreTrainedConfig.register_subclass("edge_distilled")
@dataclass
class EdgeDistilledConfig(EdgeBaseConfig):
    """Teacher → student distilled variant."""

    type: str = "edge_distilled"
    distill_epochs: int = 10
    distill_lr: float = 1e-4
    distill_temperature: float = 2.0


@PreTrainedConfig.register_subclass("edge_distilled_onnx_int8")
@dataclass
class EdgeDistilledOnnxInt8Config(EdgeBaseConfig):
    """Distilled + ONNX + INT8 (combined pipeline)."""

    type: str = "edge_distilled_onnx_int8"
    distill_epochs: int = 10
    onnx_opset: int = 17
    quantize_dynamic: bool = True
