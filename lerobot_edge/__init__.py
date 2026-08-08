"""lerobot_edge: Policy compression and edge deployment plugin for LeRobot."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from lerobot_edge.core.configs import (  # noqa: F401, E402
    EdgeIdentityConfig,
    EdgeOnnxFp32Config,
    EdgeOnnxInt8Config,
    EdgeQuantInt4Config,
    EdgeQuantInt8Config,
)
from lerobot_edge.optimization import (  # noqa: F401, E402
    QuantizedKVCache,
    configure_sdpa_backend,
    get_attention_info,
    get_kv_cache_stats,
    optimize_kv_cache,
    optimize_model_attention,
    optimize_policy_for_inference,
)

__all__ = [
    "EdgeIdentityConfig",
    "EdgeQuantInt8Config",
    "EdgeQuantInt4Config",
    "EdgeOnnxFp32Config",
    "EdgeOnnxInt8Config",
    "configure_sdpa_backend",
    "get_attention_info",
    "optimize_model_attention",
    "optimize_policy_for_inference",
    "QuantizedKVCache",
    "optimize_kv_cache",
    "get_kv_cache_stats",
]

logger.info(
    "lerobot_edge registered %d policy variants with LeRobot: %s",
    5,
    ", ".join(__all__[:5]),
)
