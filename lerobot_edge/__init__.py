"""lerobot_edge: Policy compression and edge deployment plugin for LeRobot."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Register configs with LeRobot's plugin system via register_subclass decorators.
from lerobot_edge.core.configs import (  # noqa: F401
    EdgeIdentityConfig,
    EdgeOnnxFp32Config,
    EdgeOnnxInt8Config,
    EdgeQuantInt8Config,
)

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
