"""Backward-compatible re-export — actual code in lerobot_edge/compression/."""
from lerobot_edge.compression.quantize import *  # noqa: F401,F403
from lerobot_edge.compression.quantize import dynamic_int8_quantize, static_int8_quantize, quantize_4bit, QuantizedBackend
from lerobot_edge.core.utils import measure_model_memory
