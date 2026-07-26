"""Configuration dataclasses for lerobot_edge policy variants.

Each config registers itself with LeRobot's ``PreTrainedConfig`` draccus
ChoiceRegistry via ``@PreTrainedConfig.register_subclass(name)``.

LeRobot's factory fallback (``_get_policy_cls_from_policy_name``) will
then discover these configs and dynamically import the matching policy class
using naming conventions: ``configuration_<type>`` → ``modeling_<type>``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.configs import PreTrainedConfig

__all__ = [
    "EdgeBaseConfig",
    "EdgeIdentityConfig",
    "EdgeQuantInt8Config",
    "EdgeOnnxFp32Config",
    "EdgeOnnxInt8Config",
    "EdgeDistilledConfig",
    "EdgeDistilledOnnxInt8Config",
]


# ---------------------------------------------------------------------------
# Base edge config (shared fields)
# ---------------------------------------------------------------------------

@dataclass
class EdgeBaseConfig(PreTrainedConfig):
    """Shared configuration for all lerobot_edge variants.

    Subclasses must set ``type`` via ``@PreTrainedConfig.register_subclass``.
    """

    # Path to the original (uncompressed) policy checkpoint.  The edge
    # variant wraps / transforms this checkpoint.
    source_pretrained_path: str | None = None

    # Device to run inference on (inherited from PreTrainedConfig).
    # We add a convenience alias used by the benchmark harness.
    deploy_device: str | None = None

    # -- quantization options (used by edge_quant_int8) --
    quantize_dynamic: bool = True
    quantize_static: bool = False
    quantize_bits: int = 8  # 8 or 4

    # -- ONNX options (used by edge_onnx_*) --
    onnx_opset: int = 17
    onnx_input_names: list[str] = field(default_factory=list)
    onnx_output_names: list[str] = field(default_factory=list)
    onnx_dynamic_axes: dict[str, dict[int, str]] = field(default_factory=dict)

    # -- distillation options (used by edge_distilled_*) --
    distill_epochs: int = 10
    distill_lr: float = 1e-4
    distill_temperature: float = 2.0
    distill_alpha: float = 0.5  # weight for KL vs MSE loss

    # -- benchmark options --
    benchmark_warmup: int = 10
    benchmark_num_runs: int = 100

    # ---------- required abstract implementations from PreTrainedConfig ----------

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
        return {
            "optimizer_cls": "AdamW",
            "lr": 1e-4,
            "weight_decay": 1e-4,
        }

    def get_scheduler_preset(self) -> Any | None:
        return None

    def validate_features(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Concrete registered configs
# ---------------------------------------------------------------------------

@PreTrainedConfig.register_subclass("edge_identity")
@dataclass
class EdgeIdentityConfig(EdgeBaseConfig):
    """Identity passthrough – wraps the original policy unchanged.

    This variant proves the plugin registration hook works end-to-end
    without any actual compression or transformation.
    """

    type: str = "edge_identity"


@PreTrainedConfig.register_subclass("edge_quant_int8")
@dataclass
class EdgeQuantInt8Config(EdgeBaseConfig):
    """Dynamic INT8 quantized variant."""

    type: str = "edge_quant_int8"
    quantize_dynamic: bool = True
    quantize_bits: int = 8


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
