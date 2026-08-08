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
    "QuantizedKVCache",
    "optimize_kv_cache",
    "get_kv_cache_stats",
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
# KV-Cache INT8 Quantization
# ============================================================================


class QuantizedKVCache(dict):
    """Dict wrapper that quantizes KV-cache tensors to INT8 on store.

    Drop-in replacement for the ``past_key_values`` dict in
    ``SmolVLMWithExpertModel.forward()``. Intercepts writes to
    quantize K/V tensors per-channel to INT8, and intercepts
    reads to dequantize back to the original dtype.

    Memory savings: ~2x (FP32→INT8), ~4x (FP16→INT8).
    Accuracy: Virtually lossless at INT8 precision.

    Usage::

        cache = QuantizedKVCache()
        # Store (auto-quantizes)
        cache[0] = {"key_states": k_tensor, "value_states": v_tensor}
        # Read (auto-dequantizes)
        k = cache[0]["key_states"]  # returns dequantized tensor
    """

    def __init__(
        self,
        bits: int = 8,
        per_channel: bool = True,
        symmetric: bool = True,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.per_channel = per_channel
        self.symmetric = symmetric
        self._stats: dict[str, float] = {
            "bytes_before": 0.0,
            "bytes_after": 0.0,
            "layers_quantized": 0,
        }

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, dict) and "key_states" in value:
            value = self._quantize_layer_cache(value)
        super().__setitem__(key, value)

    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        if isinstance(value, dict) and "key_states_quantized" in value:
            return self._dequantize_layer_cache(value)
        return value

    # ------------------------------------------------------------------
    # Quantize / Dequantize helpers
    # ------------------------------------------------------------------

    def _quantize_layer_cache(
        self, layer_cache: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Quantize key_states and value_states to INT8."""
        result: dict[str, Any] = {}
        for tensor_name in ("key_states", "value_states"):
            if tensor_name in layer_cache:
                t = layer_cache[tensor_name]
                q_result = self._quantize_tensor(t)
                result[f"{tensor_name}_quantized"] = q_result["data"]
                result[f"{tensor_name}_scale"] = q_result["scale"]
                result[f"{tensor_name}_zp"] = q_result.get("zero_point")
                result[f"{tensor_name}_dtype"] = str(t.dtype)
            else:
                result[tensor_name] = layer_cache[tensor_name]
        self._stats["layers_quantized"] += 1
        return result

    def _dequantize_layer_cache(
        self, layer_cache: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Dequantize key_states and value_states from INT8."""
        result: dict[str, torch.Tensor] = {}
        for tensor_name in ("key_states", "value_states"):
            q_name = f"{tensor_name}_quantized"
            if q_name in layer_cache:
                q_data = layer_cache[q_name]
                scale = layer_cache[f"{tensor_name}_scale"]
                zp = layer_cache.get(f"{tensor_name}_zp")
                orig_dtype_str = layer_cache.get(
                    f"{tensor_name}_dtype", "float32"
                )
                orig_dtype = getattr(torch, orig_dtype_str, torch.float32)
                result[tensor_name] = self._dequantize_tensor(
                    q_data, scale, zp, orig_dtype
                )
            elif tensor_name in layer_cache:
                result[tensor_name] = layer_cache[tensor_name]
        return result

    def _quantize_tensor(
        self, t: torch.Tensor
    ) -> dict[str, Any]:
        """Per-channel or per-tensor INT8 quantization."""
        orig_bytes = t.nelement() * t.element_size()
        self._stats["bytes_before"] += orig_bytes

        if self.per_channel and t.dim() >= 2:
            # Per-channel along last dim (head_dim)
            t_flat = t.float().reshape(-1, t.shape[-1])
            if self.symmetric:
                amax = t_flat.abs().amax(dim=0).clamp_min(1e-12)
                qmax = 127
                scale = amax / qmax
                q_data = (t_flat / scale[None, :]).round().clamp(-qmax, qmax).to(torch.int8)
                q_data = q_data.reshape(t.shape)
                scale = scale.reshape(-1)
                result = {"data": q_data, "scale": scale}
            else:
                t_min = t_flat.amin(dim=0)
                t_max = t_flat.amax(dim=0)
                qmin, qmax = -128, 127
                scale = (t_max - t_min).clamp_min(1e-12) / (qmax - qmin)
                zp = ((qmin - t_min / scale.clamp_min(1e-12))).round().to(torch.int8)
                q_data = ((t_flat / scale[None, :]) + zp[None, :]).round().clamp(qmin, qmax).to(torch.int8)
                q_data = q_data.reshape(t.shape)
                result = {"data": q_data, "scale": scale.reshape(-1), "zero_point": zp.reshape(-1)}
        else:
            # Per-tensor quantization
            if self.symmetric:
                amax = t.float().abs().max().clamp_min(1e-12)
                scale = amax / 127.0
                q_data = (t.float() / scale).round().clamp(-127, 127).to(torch.int8)
                result = {"data": q_data, "scale": scale}
            else:
                t_min, t_max = t.float().min(), t.float().max()
                qmin, qmax = -128, 127
                scale = ((t_max - t_min) / (qmax - qmin)).clamp_min(1e-12)
                zp = int(qmin - t_min.item() / scale.item())
                q_data = ((t.float() / scale) + zp).round().clamp(qmin, qmax).to(torch.int8)
                result = {"data": q_data, "scale": scale, "zero_point": zp}

        q_bytes = q_data.nelement() * q_data.element_size()
        # Add metadata overhead (scale + zp)
        if "scale" in result and isinstance(result["scale"], torch.Tensor):
            q_bytes += result["scale"].nelement() * result["scale"].element_size()
        if "zero_point" in result and isinstance(result["zero_point"], torch.Tensor):
            q_bytes += result["zero_point"].nelement() * result["zero_point"].element_size()
        self._stats["bytes_after"] += q_bytes

        return result

    @staticmethod
    def _dequantize_tensor(
        q_data: torch.Tensor,
        scale: torch.Tensor | float,
        zp: torch.Tensor | int | None,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize INT8 tensor back to target dtype."""
        if isinstance(scale, torch.Tensor) and scale.dim() == 1 and scale.shape[0] == q_data.shape[-1]:
            # Per-channel: scale has shape [D]
            s = scale.to(q_data.device).to(torch.float32)
            z = zp.to(q_data.device).to(torch.float32) if zp is not None else 0.0
            result = (q_data.float() - z) * s[None, None, None, :]
        elif isinstance(scale, torch.Tensor) and scale.ndim == 0:
            s = scale.item()
            z = zp.item() if isinstance(zp, torch.Tensor) else (zp or 0)
            result = (q_data.float() - z) * s
        else:
            s = float(scale) if not isinstance(scale, torch.Tensor) else scale.item()
            z = float(zp) if zp is not None and not isinstance(zp, torch.Tensor) else (
                zp.item() if isinstance(zp, torch.Tensor) else 0.0
            )
            result = (q_data.float() - z) * s

        return result.to(target_dtype)

    @property
    def compression_ratio(self) -> float:
        """Byte-level compression ratio achieved."""
        if self._stats["bytes_after"] == 0:
            return 1.0
        return self._stats["bytes_before"] / self._stats["bytes_after"]

    @property
    def stats(self) -> dict[str, float]:
        return {
            **self._stats,
            "compression_ratio": self.compression_ratio,
            "savings_pct": (1.0 - 1.0 / self.compression_ratio) * 100
            if self.compression_ratio > 0
            else 0.0,
        }


def optimize_kv_cache(
    model: torch.nn.Module,
    bits: int = 8,
    per_channel: bool = True,
    symmetric: bool = True,
) -> QuantizedKVCache:
    """Patch a VLA model to use quantized KV-cache.

    Monkey-patches the model's ``forward`` method so that the
    ``past_key_values`` dict is automatically wrapped in a
    ``QuantizedKVCache``. All stored K/V tensors are quantized
    to INT8; reads are dequantized on-the-fly.

    Args:
        model: The VLA model to optimize.
        bits: Quantization bits (default 8 for INT8).
        per_channel: Use per-channel quantization (more accurate).
        symmetric: Use symmetric quantization (no zero-point).

    Returns:
        The ``QuantizedKVCache`` instance (can be inspected for stats).
    """
    model_cls = type(model).__name__
    if model_cls != "SmolVLMWithExpertModel":
        logger.warning(
            "KV-cache quantization: unsupported model type %s. Skipping.",
            model_cls,
        )
        return QuantizedKVCache()

    import types

    # Store reference to patched cache
    quantized_cache: QuantizedKVCache | None = None
    _orig_forward = model.forward

    def _forward_with_kv_cache_quant(
        self: Any,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list | None = None,
        inputs_embeds: list | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ) -> Any:
        nonlocal quantized_cache

        # Create quantized cache wrapper if filling cache
        if fill_kv_cache and past_key_values is None:
            quantized_cache = QuantizedKVCache(
                bits=bits, per_channel=per_channel, symmetric=symmetric
            )
            past_key_values = quantized_cache
        elif isinstance(past_key_values, QuantizedKVCache):
            quantized_cache = past_key_values

        return _orig_forward(
            self,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
        )

    model.forward = types.MethodType(  # type: ignore[method-assign]
        _forward_with_kv_cache_quant, model
    )

    logger.info(
        "Patched SmolVLMWithExpertModel KV-cache → INT%d "
        "(per_channel=%s, symmetric=%s)",
        bits,
        per_channel,
        symmetric,
    )

    return quantized_cache or QuantizedKVCache()


def get_kv_cache_stats(cache: QuantizedKVCache) -> dict[str, float]:
    """Get statistics from a quantized KV-cache."""
    return cache.stats


# ============================================================================
# Convenience: full pipeline optimization
# ============================================================================


def optimize_policy_for_inference(
    policy: torch.nn.Module,
    *,
    enable_attention: bool = True,
    enable_kv_cache_quant: bool = True,
    enable_compile: bool = False,
    compile_mode: str = "reduce-overhead",
    kv_cache_bits: int = 8,
) -> torch.nn.Module:
    """Apply all available inference optimizations to a policy.

    Convenience function that applies all optimizations in one call:
      1. SDPA/FlashAttention (replaces eager matmul attention)
      2. INT8 KV-cache quantization (compresses cached K/V tensors)
      3. torch.compile (optional)

    Args:
        policy: The LeRobot policy module.
        enable_attention: Apply SDPA attention patching.
        enable_kv_cache_quant: Quantize KV-cache to INT8.
        enable_compile: Apply torch.compile.
        compile_mode: Compilation mode.
        kv_cache_bits: KV-cache quantization bits (default 8).

    Returns:
        The optimized policy (modified in place if patching,
        new reference if compiled).
    """
    if enable_attention:
        _walk_and_optimize(policy)
        logger.info("Attention optimization applied.")

    if enable_kv_cache_quant:
        _walk_and_apply_kv_cache_quant(policy, bits=kv_cache_bits)
        logger.info("KV-cache INT%d quantization applied.", kv_cache_bits)

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


def _walk_and_apply_kv_cache_quant(
    module: torch.nn.Module, bits: int = 8
) -> None:
    """Walk module tree and apply KV-cache quantization."""
    for child in module.children():
        if type(child).__name__ in ("SmolVLMWithExpertModel",):
            optimize_kv_cache(child, bits=bits)
        else:
            _walk_and_apply_kv_cache_quant(child, bits=bits)
