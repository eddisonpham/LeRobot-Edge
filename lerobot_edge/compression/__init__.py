"""Compression methods: quantization and distillation."""

from lerobot_edge.compression.distill import (
    DistillationLoss,
    DistilledBackend,
    distill,
)
from lerobot_edge.compression.quantize import (
    QuantizedBackend,
    dynamic_int8_quantize,
    quantize_4bit,
    quantize_bnb_fp4,
    quantize_bnb_int8,
    quantize_int4_weight_only,
    static_int8_quantize,
)

__all__ = [
    "dynamic_int8_quantize",
    "static_int8_quantize",
    "quantize_int4_weight_only",
    "quantize_4bit",
    "quantize_bnb_int8",
    "quantize_bnb_fp4",
    "QuantizedBackend",
    "DistilledBackend",
    "DistillationLoss",
    "distill",
]
