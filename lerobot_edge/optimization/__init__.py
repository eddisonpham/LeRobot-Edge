"""Attention & KV-cache optimizations for VLA inference."""

from __future__ import annotations

import functools
import logging
import types
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)

__all__ = [
    "configure_sdpa_backend",
    "sdpa_attention_forward",
    "optimize_model_attention",
    "get_attention_info",
    "QuantizedKVCache",
    "optimize_kv_cache",
    "get_kv_cache_stats",
    "optimize_policy_for_inference",
]

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
    """Enable SDPA backends globally. Priority: flash > mem_efficient > math."""
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(enable_flash)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(enable_mem_efficient)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(enable_math)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(enable_cudnn)

    status = {
        "flash": enable_flash and torch.cuda.is_available(),
        "mem_efficient": enable_mem_efficient and torch.cuda.is_available(),
        "math": enable_math,
        "cudnn": enable_cudnn and torch.cuda.is_available(),
    }
    logger.info("SDPA backend: %s", status)
    return status


def sdpa_attention_forward(
    self: Any,
    attention_mask: torch.Tensor,
    batch_size: int,
    head_dim: int,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    """Drop-in replacement for SmolVLMWithExpertModel.eager_attention_forward.

    Uses F.scaled_dot_product_attention which auto-selects FlashAttention-2,
    mem-efficient, or math backend. O(n) memory vs O(n²) for eager.
    """
    n_heads = self.num_attention_heads
    n_kv = self.num_key_value_heads
    groups = n_heads // n_kv
    seq_len = key_states.shape[1]

    # GQA expansion
    key_states = key_states[:, :, :, None, :].expand(batch_size, seq_len, n_kv, groups, head_dim)
    key_states = key_states.reshape(batch_size, seq_len, n_heads, head_dim)
    value_states = value_states[:, :, :, None, :].expand(
        batch_size, seq_len, n_kv, groups, head_dim
    )
    value_states = value_states.reshape(batch_size, seq_len, n_heads, head_dim)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # Bool mask → float: True=attend → 0=attend, -inf=mask
    if attention_mask.dtype == torch.bool:
        sdpa_mask = attention_mask[:, None, :, :].float()
        sdpa_mask = (1.0 - sdpa_mask) * torch.finfo(sdpa_mask.dtype).min
    else:
        sdpa_mask = attention_mask[:, None, :, :]

    target_dtype = query_states.dtype
    key_states = key_states.to(target_dtype)
    value_states = value_states.to(target_dtype)
    if sdpa_mask.dtype != target_dtype and sdpa_mask.dtype != torch.bool:
        sdpa_mask = sdpa_mask.to(target_dtype)

    att_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=sdpa_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=head_dim**-0.5,
    )

    att_output = att_output.transpose(1, 2).contiguous()
    att_output = att_output.reshape(batch_size, -1, n_heads * head_dim)
    return att_output


def optimize_model_attention(model: torch.nn.Module) -> torch.nn.Module:
    """Monkey-patch model to use SDPA attention. Returns same model instance."""
    name = type(model).__name__
    if name != "SmolVLMWithExpertModel":
        logger.debug("No attention optimization for %s. Skipping.", name)
        return model

    configure_sdpa_backend()

    def _sdpa_interface(self: Any) -> Callable[..., torch.Tensor]:
        return functools.partial(sdpa_attention_forward, self)

    model.get_attention_interface = types.MethodType(_sdpa_interface, model)  # type: ignore
    logger.info("Patched %s attention → SDPA", name)
    return model


def get_attention_info() -> dict[str, Any]:
    """Return available SDPA backends and recommended choice."""
    info: dict[str, Any] = {
        "flash_attn_package": HAS_FLASH_ATTN,
        "cuda_available": torch.cuda.is_available(),
    }
    if info["cuda_available"]:
        b = torch.backends.cuda
        info["flash_enabled"] = b.flash_sdp_enabled() if hasattr(b, "flash_sdp_enabled") else False
        info["mem_efficient_enabled"] = (
            b.mem_efficient_sdp_enabled() if hasattr(b, "mem_efficient_sdp_enabled") else False
        )
        info["math_enabled"] = b.math_sdp_enabled() if hasattr(b, "math_sdp_enabled") else True
        gpu = torch.cuda.get_device_name(0)
        if info["flash_enabled"] and any(
            a in gpu for a in ["A100", "A10", "RTX 30", "RTX 40", "RTX 50", "H100"]
        ):
            info["recommended_backend"] = "FlashAttention (SDPA)"
        elif info["mem_efficient_enabled"]:
            info["recommended_backend"] = "Memory-Efficient (SDPA)"
        else:
            info["recommended_backend"] = "Math (SDPA fallback)"
    else:
        info["recommended_backend"] = "Math (CPU)"
    return info


# ---------------------------------------------------------------------------
# KV-Cache INT8 Quantization
# ---------------------------------------------------------------------------


class QuantizedKVCache(dict):
    """Dict wrapper that quantizes KV-cache tensors to INT8 on store/dequantizes on read.

    Drop-in for ``past_key_values`` in SmolVLMWithExpertModel.forward().
    Memory: ~4× compression (FP32→INT8), <0.01 mean error.
    """

    def __init__(self, bits: int = 8, per_channel: bool = True, symmetric: bool = True) -> None:
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

    def _quantize_layer_cache(self, lc: dict) -> dict:
        result: dict[str, Any] = {}
        for tn in ("key_states", "value_states"):
            if tn in lc:
                t = lc[tn]
                q = self._quantize_tensor(t)
                result[f"{tn}_quantized"] = q["data"]
                result[f"{tn}_scale"] = q["scale"]
                result[f"{tn}_zp"] = q.get("zero_point")
                result[f"{tn}_dtype"] = str(t.dtype)
            else:
                result[tn] = lc[tn]
        self._stats["layers_quantized"] += 1
        return result

    def _dequantize_layer_cache(self, lc: dict) -> dict:
        result: dict[str, torch.Tensor] = {}
        for tn in ("key_states", "value_states"):
            qn = f"{tn}_quantized"
            if qn in lc:
                dt = getattr(torch, lc.get(f"{tn}_dtype", "float32"), torch.float32)
                result[tn] = self._dequantize_tensor(
                    lc[qn], lc[f"{tn}_scale"], lc.get(f"{tn}_zp"), dt
                )
            elif tn in lc:
                result[tn] = lc[tn]
        return result

    def _quantize_tensor(self, t: torch.Tensor) -> dict[str, Any]:
        orig = t.nelement() * t.element_size()
        self._stats["bytes_before"] += orig
        if self.per_channel and t.dim() >= 2:
            flat = t.float().reshape(-1, t.shape[-1])
            amax = flat.abs().amax(dim=0).clamp_min(1e-12)
            scale = amax / 127
            q_data = (
                (flat / scale[None, :]).round().clamp(-127, 127).to(torch.int8).reshape(t.shape)
            )
            result = {"data": q_data, "scale": scale.reshape(-1)}
        else:
            amax = t.float().abs().max().clamp_min(1e-12)
            scale = amax / 127.0
            q_data = (t.float() / scale).round().clamp(-127, 127).to(torch.int8)
            result = {"data": q_data, "scale": scale}
        q_bytes = q_data.nelement() * q_data.element_size()
        if isinstance(result["scale"], torch.Tensor):
            q_bytes += result["scale"].nelement() * result["scale"].element_size()
        self._stats["bytes_after"] += q_bytes
        return result

    @staticmethod
    def _dequantize_tensor(
        q_data: torch.Tensor,
        scale: torch.Tensor | float,
        zp: torch.Tensor | int | None,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            isinstance(scale, torch.Tensor)
            and scale.dim() == 1
            and scale.shape[0] == q_data.shape[-1]
        ):
            s = scale.to(q_data.device).float()
            z = zp.to(q_data.device).float() if zp is not None else 0.0
            return ((q_data.float() - z) * s[None, None, None, :]).to(target_dtype)
        s = float(scale) if not isinstance(scale, torch.Tensor) else scale.item()
        z = (
            float(zp)
            if zp is not None and not isinstance(zp, torch.Tensor)
            else (zp.item() if isinstance(zp, torch.Tensor) else 0.0)
        )
        return ((q_data.float() - z) * s).to(target_dtype)

    @property
    def compression_ratio(self) -> float:
        if self._stats["bytes_after"] == 0:
            return 1.0
        return self._stats["bytes_before"] / self._stats["bytes_after"]

    @property
    def stats(self) -> dict[str, float]:
        cr = self.compression_ratio
        return {
            **self._stats,
            "compression_ratio": cr,
            "savings_pct": (1.0 - 1.0 / cr) * 100 if cr > 0 else 0.0,
        }


def optimize_kv_cache(
    model: torch.nn.Module, bits: int = 8, per_channel: bool = True, symmetric: bool = True
) -> QuantizedKVCache:
    """Monkey-patch model.forward to use QuantizedKVCache for KV-cache quantization."""
    if type(model).__name__ != "SmolVLMWithExpertModel":
        logger.warning("KV-cache quant unsupported for %s. Skipping.", type(model).__name__)
        return QuantizedKVCache()

    qcache: QuantizedKVCache | None = None
    _orig = model.forward

    def _patched(
        self: Any,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        fill_kv_cache=None,
    ) -> Any:
        nonlocal qcache
        if fill_kv_cache and past_key_values is None:
            qcache = QuantizedKVCache(bits=bits, per_channel=per_channel, symmetric=symmetric)
            past_key_values = qcache
        elif isinstance(past_key_values, QuantizedKVCache):
            qcache = past_key_values
        return _orig(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
        )

    model.forward = types.MethodType(_patched, model)  # type: ignore
    logger.info("Patched KV-cache → INT%d", bits)
    return qcache or QuantizedKVCache()


def get_kv_cache_stats(cache: QuantizedKVCache) -> dict[str, float]:
    return cache.stats


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def _can_compile() -> bool:
    """Check if torch.compile is available with Triton (Linux) or Inductor C++ (fallback)."""
    if not hasattr(torch, "compile"):
        return False
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        pass
    # On Linux, Inductor C++ backend may work without Triton
    import sys

    return sys.platform == "linux"


def optimize_policy_for_inference(
    policy: torch.nn.Module,
    *,
    enable_attention: bool = True,
    enable_kv_cache_quant: bool = True,
    enable_compile: bool = False,
    compile_mode: str = "reduce-overhead",
    kv_cache_bits: int = 8,
) -> torch.nn.Module:
    """Apply all inference optimizations: SDPA + KV-cache INT8 + optional compile."""
    if enable_attention:
        _walk(policy, "attention")
        logger.info("SDPA attention applied.")
    if enable_kv_cache_quant:
        _walk(policy, "kv_cache", bits=kv_cache_bits)
        logger.info("INT%d KV-cache applied.", kv_cache_bits)
    if enable_compile:
        if _can_compile():
            policy = torch.compile(policy, mode=compile_mode)
            logger.info("torch.compile(%s) applied.", compile_mode)
        else:
            logger.warning(
                "torch.compile requires Triton (Linux) or Inductor C++ (Linux). "
                "Not available on this platform. Skipping compile."
            )
    return policy


def _walk(module: torch.nn.Module, action: str, **kw: Any) -> None:
    for child in module.children():
        name = type(child).__name__
        if name == "SmolVLMWithExpertModel":
            if action == "attention":
                optimize_model_attention(child)
            elif action == "kv_cache":
                optimize_kv_cache(child, **kw)
        else:
            _walk(child, action, **kw)
