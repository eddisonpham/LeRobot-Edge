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
    static_int8_quantize,
)

__all__ = [
    "dynamic_int8_quantize",
    "static_int8_quantize",
    "quantize_4bit",
    "QuantizedBackend",
    "DistilledBackend",
    "DistillationLoss",
    "distill",
]
