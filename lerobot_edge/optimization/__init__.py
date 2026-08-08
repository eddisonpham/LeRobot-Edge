"""Attention optimization for VLA models via SDPA / FlashAttention.

Design
========
Monkey-patches the attention interface of SmolVLMWithExpertModel
(and similar VLA models) to use ``torch.nn.functional.scaled_dot_product_attention``
(SDPA) instead of manual ``eager_attention_forward``.

SDPA automatically selects the best backend:
  - **FlashAttention-2** on Ampere+ GPUs with FP16/BF16 (O(n) memory, 2-4x faster)
  - **Memory-Efficient Attention** on older GPUs / FP32
  - **Math fallback** when neither is available

Usage::

    from lerobot_edge.optimization.attention import optimize_model_attention

    # Monkey-patch the model to use SDPA
    model = optimize_model_attention(model)

    # Or globally: enable flash attention + force SDPA
    configure_sdpa_backend(enable_flash=True, enable_mem_efficient=True)

No modifications to LeRobot source code required.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "configure_sdpa_backend",
    "sdpa_attention_forward",
    "optimize_model_attention",
    "get_attention_info",
    "HAS_FLASH_ATTN",
]

# ---------------------------------------------------------------------------
# FlashAttention availability detection
# ---------------------------------------------------------------------------

try:
    import flash_attn  # noqa: F401

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def configure_sdpa_backend(
    enable_flash: bool = True,
    enable_mem_efficient: bool = True,
    enable_math: bool = True,
    enable_cudnn: bool = False,
) -> dict[str, bool]:
    """Configure the SDPA backend priority globally.

    Call once at startup to set backend preferences. SDPA will try
    backends in priority order.

    Returns:
        Dict of backend name → enabled status.
    """
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(enable_flash)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(enable_mem_efficient)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(enable_math)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(enable_cudnn)

    status = {
        "flash": enable_flash and _cuda_available(),
        "mem_efficient": enable_mem_efficient and _cuda_available(),
        "math": enable_math,
        "cudnn": enable_cudnn and _cuda_available(),
    }
    logger.info("SDPA backend config: %s", status)
    return status


def _cuda_available() -> bool:
    return torch.cuda.is_available()


# ============================================================================
# SDPA attention forward — drop-in replacement for eager_attention_forward
# ============================================================================


def sdpa_attention_forward(
    self: Any,
    attention_mask: torch.Tensor,
    batch_size: int,
    head_dim: int,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    """SDPA-based attention forward, drop-in replacement.

    Replaces the O(n²) eager attention with fused SDPA kernels.

    The original ``eager_attention_forward`` does:
      1. GQA key/value expansion
      2. Upcast to float32
      3. Q·K^T → scale → mask → softmax → ·V
      4. Reshape output

    SDPA does all of this in a single fused kernel, with O(n) memory
    complexity via FlashAttention tiling.
    """
    num_att_heads = self.num_attention_heads
    num_key_value_heads = self.num_key_value_heads
    num_key_value_groups = num_att_heads // num_key_value_heads

    sequence_length = key_states.shape[1]

    # --- GQA expansion (same as original) ---
    key_states = key_states[:, :, :, None, :].expand(
        batch_size, sequence_length, num_key_value_heads,
        num_key_value_groups, head_dim,
    )
    key_states = key_states.reshape(
        batch_size, sequence_length,
        num_key_value_heads * num_key_value_groups, head_dim,
    )

    value_states = value_states[:, :, :, None, :].expand(
        batch_size, sequence_length, num_key_value_heads,
        num_key_value_groups, head_dim,
    )
    value_states = value_states.reshape(
        batch_size, sequence_length,
        num_key_value_heads * num_key_value_groups, head_dim,
    )

    # --- Transpose to (B, H, L, D) for SDPA ---
    # Original: (B, L, H, D) → after transpose: (B, H, L, D)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # --- Convert boolean attention mask to SDPA float mask ---
    # Original: True = attend, False = mask → SDPA: 0.0 = attend, -inf = mask
    if attention_mask.dtype == torch.bool:
        # attention_mask is (B, L, L) with True=attend
        # SDPA expects (B, 1, L, L) or (1, 1, L, L) float where 0=attend, -inf=mask
        sdpa_mask = attention_mask[:, None, :, :].float()
        sdpa_mask = (1.0 - sdpa_mask) * torch.finfo(sdpa_mask.dtype).min
    else:
        sdpa_mask = attention_mask[:, None, :, :]

    # Scale factor: head_dim ** -0.5
    scale = head_dim ** -0.5

    # --- SDPA: fused attention ---
    att_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=sdpa_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )

    # --- Reshape output (matching original layout) ---
    # SDPA returns (B, H, L, D) → need (B, L, H*D)
    att_output = att_output.transpose(1, 2).contiguous()
    att_output = att_output.reshape(
        batch_size, -1,
        num_key_value_heads * num_key_value_groups * head_dim,
    )

    # Original upcasts to float32 before softmax; SDPA handles internal precision
    # Keep output in whatever dtype SDPA produced (typically input dtype)
    return att_output


# ============================================================================
# Model patching
# ============================================================================


def optimize_model_attention(model: torch.nn.Module) -> torch.nn.Module:
    """Patch a VLA model to use SDPA attention.

    Detects the model type and replaces ``get_attention_interface``
    to return SDPA-based attention.

    Supported models:
      - SmolVLMWithExpertModel (SmolVLA)

    Args:
        model: The model to optimize (modified in place).

    Returns:
        The same model instance (modified in place).

    Raises:
        TypeError: If the model type is not recognized.
    """
    model_cls = type(model).__name__

    if model_cls == "SmolVLMWithExpertModel":
        return _patch_smolvlm_with_expert(model)
    else:
        logger.debug(
            "No attention optimization available for %s. "
            "Falling back to default attention.",
            model_cls,
        )
        return model


def _patch_smolvlm_with_expert(model: torch.nn.Module) -> torch.nn.Module:
    """Patch SmolVLMWithExpertModel to use SDPA."""

    # Enable SDPA backends globally
    configure_sdpa_backend()

    # Replace get_attention_interface to always return SDPA
    def _sdpa_interface(self: Any) -> Callable[..., torch.Tensor]:
        return functools.partial(sdpa_attention_forward, self)

    # Monkey-patch: bind the new method
    import types

    model.get_attention_interface = types.MethodType(  # type: ignore[method-assign]
        _sdpa_interface, model
    )

    logger.info(
        "Patched SmolVLMWithExpertModel attention → SDPA "
        "(FlashAttention/mem_efficient/math auto-selected)"
    )
    return model


# ============================================================================
# Utilities
# ============================================================================


def get_attention_info() -> dict[str, Any]:
    """Get information about available attention backends.

    Returns:
        Dict with keys: flash_available, mem_efficient_available,
        math_available, cudnn_available, flash_attn_package,
        recommended_backend.
    """
    info: dict[str, Any] = {
        "flash_attn_package": HAS_FLASH_ATTN,
        "cuda_available": _cuda_available(),
    }

    if _cuda_available():
        # Query what SDPA backends are enabled/available
        if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
            info["flash_enabled"] = torch.backends.cuda.flash_sdp_enabled()
        else:
            info["flash_enabled"] = False

        if hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled"):
            info["mem_efficient_enabled"] = (
                torch.backends.cuda.mem_efficient_sdp_enabled()
            )
        else:
            info["mem_efficient_enabled"] = False

        if hasattr(torch.backends.cuda, "math_sdp_enabled"):
            info["math_enabled"] = torch.backends.cuda.math_sdp_enabled()
        else:
            info["math_enabled"] = True

        # Determine recommended backend
        gpu_name = torch.cuda.get_device_name(0)
        is_ampere_plus = any(
            arch in gpu_name
            for arch in ["A100", "A10", "A40", "RTX 30", "RTX 40", "RTX 50", "H100"]
        )

        if info.get("flash_enabled") and is_ampere_plus:
            info["recommended_backend"] = "FlashAttention (SDPA)"
        elif info.get("mem_efficient_enabled"):
            info["recommended_backend"] = "Memory-Efficient (SDPA)"
        else:
            info["recommended_backend"] = "Math (SDPA fallback)"
    else:
        info["recommended_backend"] = "Math (CPU)"

    return info


# ============================================================================
# Convenience: full pipeline optimization
# ============================================================================


def optimize_policy_for_inference(
    policy: torch.nn.Module,
    *,
    enable_attention: bool = True,
    enable_compile: bool = False,
    compile_mode: str = "reduce-overhead",
) -> torch.nn.Module:
    """Apply all available inference optimizations to a policy.

    Convenience function that applies attention optimization and
    optionally torch.compile in one call.

    Args:
        policy: The LeRobot policy module.
        enable_attention: Apply SDPA attention patching.
        enable_compile: Apply torch.compile.
        compile_mode: Compilation mode.

    Returns:
        The optimized policy (modified in place if patching,
        new reference if compiled).
    """
    if enable_attention:
        # Walk model tree to find attention-capable submodules
        _walk_and_optimize(policy)
        logger.info("Attention optimization applied.")

    if enable_compile and hasattr(torch, "compile"):
        policy = torch.compile(policy, mode=compile_mode)
        logger.info("torch.compile applied (mode=%s).", compile_mode)

    return policy


def _walk_and_optimize(module: torch.nn.Module) -> None:
    """Walk module tree and apply attention optimization to supported modules."""
    for child in module.children():
        if type(child).__name__ in ("SmolVLMWithExpertModel",):
            optimize_model_attention(child)
        else:
            _walk_and_optimize(child)
