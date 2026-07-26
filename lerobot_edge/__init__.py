"""lerobot_edge: Policy compression and edge deployment plugin for LeRobot.

This package registers compressed/deployed policy variants with LeRobot's
plugin system so they can be used via ``--policy.type=<variant>`` without
any modifications to LeRobot's source code.

Variants registered:
    edge_identity   – passthrough wrapper (proves the plugin hook works)
    edge_quant_int8 – dynamic INT8 quantization
    edge_onnx_fp32  – ONNX Runtime inference (FP32 weights)
    edge_onnx_int8  – ONNX Runtime inference (INT8 quantized)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registration with LeRobot's plugin system
# ---------------------------------------------------------------------------
# LeRobot uses draccus.ChoiceRegistry on PreTrainedConfig.  The fallback in
# ``lerobot.policies.factory.get_policy_class`` calls
# ``PreTrainedConfig.get_known_choices()`` and dynamically imports the
# corresponding policy class via naming conventions.  We participate in this
# by decorating our config classes with ``@PreTrainedConfig.register_subclass``.
# ---------------------------------------------------------------------------

# Import config classes so the register_subclass decorators execute at import
# time.  The actual policy classes are imported lazily in factory.py.
from lerobot_edge.configs import (  # noqa: F401
    EdgeIdentityConfig,
    EdgeOnnxFp32Config,
    EdgeOnnxInt8Config,
    EdgeQuantInt8Config,
)

# Re-export for public API
__all__ = [
    "EdgeIdentityConfig",
    "EdgeQuantInt8Config",
    "EdgeOnnxFp32Config",
    "EdgeOnnxInt8Config",
]

logger.info(
    "lerobot_edge registered %d policy variants with LeRobot: %s",
    len(__all__),
    ", ".join(__all__),
)
