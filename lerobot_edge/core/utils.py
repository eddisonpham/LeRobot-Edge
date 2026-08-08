"""Shared utilities for lerobot_edge."""

from __future__ import annotations

import importlib
import logging
import subprocess

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "get_git_commit_hash",
    "get_platform_info",
    "measure_model_memory",
    "measure_peak_memory_mb",
    "measure_cuda_memory_mb",
    "build_dummy_input",
    "load_policy_from_checkpoint",
]


def get_git_commit_hash() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_platform_info() -> dict[str, object]:
    """Return dict of platform capabilities for cross-platform awareness."""
    import platform
    import sys

    has_cuda = torch.cuda.is_available()
    has_triton = False
    triton_version = None
    if has_cuda and hasattr(torch, "compile"):
        try:
            import triton

            has_triton = True
            triton_version = triton.__version__
        except ImportError:
            pass

    has_compile = hasattr(torch, "compile")
    compile_ready = has_compile and (has_triton or sys.platform == "linux")

    return {
        "os": platform.system(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": has_cuda,
        "gpu": torch.cuda.get_device_name(0) if has_cuda else None,
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        if has_cuda
        else None,
        "triton": has_triton,
        "triton_version": triton_version,
        "torch_compile": has_compile,
        "compile_ready": compile_ready,
    }


def _param_byte_size(p: torch.nn.Parameter) -> int:
    """Get true byte size, handling quantized tensor subclasses."""
    cls_name = type(p).__name__

    # torchao AffineQuantizedTensor
    if "AffineQuantized" in cls_name and hasattr(p, "quantized_data"):
        return p.quantized_data.nelement() * p.quantized_data.element_size()

    # torchao quantized tensors have _data or int_data
    if hasattr(p, "_data"):
        inner = p._data
        if isinstance(inner, torch.Tensor):
            return inner.nelement() * inner.element_size()

    # bitsandbytes Int8Params / Params4bit
    if cls_name in ("Int8Params", "Params4bit") and hasattr(p, "data"):
        inner = p.data
        if isinstance(inner, torch.Tensor):
            return inner.nelement() * inner.element_size()

    # Default: element_size may be wrong for subclasses, compute from dtype
    if hasattr(p, "dtype"):
        try:
            bits = p.dtype.itemsize * 8 if hasattr(p.dtype, "itemsize") else 0
            if bits == 0 and hasattr(p, "quant_state"):
                # bnb 4-bit: elements in quant_state
                bits = 4
                return (p.nelement() * bits) // 8
        except Exception:
            pass

    return p.nelement() * p.element_size()


def measure_model_memory(model: nn.Module) -> dict[str, float]:
    """Measure model memory, handling quantized tensor subclasses."""
    param_bytes = sum(_param_byte_size(p) for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_bytes = param_bytes + buffer_bytes
    num_params = sum(p.nelement() for p in model.parameters())

    return {
        "param_bytes": param_bytes,
        "buffer_bytes": buffer_bytes,
        "total_bytes": total_bytes,
        "param_mb": param_bytes / (1024 * 1024),
        "buffer_mb": buffer_bytes / (1024 * 1024),
        "total_mb": total_bytes / (1024 * 1024),
        "num_parameters": num_params,
    }


def measure_peak_memory_mb() -> float:
    """Peak memory in MB (CUDA driver-level, CPU best-effort)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined]
            # macOS returns bytes, Linux returns KB
            rss_kb = float(usage.ru_maxrss)
            # Heuristic: if RSS > 10M, likely bytes (macOS)
            if hasattr(resource, "RUSAGE_SELF") and rss_kb > 10_000_000:
                rss_kb /= 1024
            return rss_kb / 1024
        except (ImportError, AttributeError):
            return 0.0


def measure_cuda_memory_mb() -> float:
    """Current CUDA memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def build_dummy_input(policy: nn.Module, device: torch.device) -> dict[str, torch.Tensor]:
    """Build dummy input batch from policy's expected features."""
    dummy_input = {}
    if hasattr(policy, "config") and hasattr(policy.config, "input_features"):
        for name, feature in policy.config.input_features.items():  # type: ignore[union-attr,operator]
            shape = list(feature.shape) if hasattr(feature, "shape") else [1, 3, 224, 224]
            if len(shape) == 0:
                shape = [1]
            elif shape[0] != 1:
                shape.insert(0, 1)

            if "language" in name and "tokens" in name:
                dummy_input[name] = torch.randint(0, 32000, shape, device=device)
            elif "language" in name and "attention_mask" in name:
                dummy_input[name] = torch.ones(shape, dtype=torch.bool, device=device)
            else:
                dummy_input[name] = torch.randn(shape, device=device)

    if not dummy_input:
        dummy_input = {
            "observation.images.front": torch.randn(1, 3, 224, 224, device=device),
            "observation.state": torch.randn(1, 2, device=device),
        }

    policy_name = type(policy).__name__
    if policy_name == "SmolVLAPolicy" and "observation.language.tokens" not in dummy_input:
        dummy_input["observation.language.tokens"] = torch.randint(0, 32000, (1, 16), device=device)
        dummy_input["observation.language.attention_mask"] = torch.ones(
            1, 16, dtype=torch.bool, device=device
        )

    return dummy_input


def load_policy_from_checkpoint(
    checkpoint: str,
    policy_type: str = "smolvla",
    device: str = "cpu",
) -> nn.Module:
    """Load a policy from checkpoint or HuggingFace Hub."""
    known_arch: dict[str, str] = {
        "smolvla": "lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy",
    }

    if policy_type in known_arch:
        try:
            mod_path, cls_name = known_arch[policy_type].rsplit(".", 1)
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            logger.info("Loading %s via from_pretrained(%s)", policy_type, checkpoint)
            model = cls.from_pretrained(checkpoint)
            model.to(device)
            model.eval()
            return model  # type: ignore[no-any-return]
        except Exception as e:
            logger.debug("from_pretrained failed for %s: %s", policy_type, e)

    logger.info("Loading %s via factory for %s", policy_type, checkpoint)
    from lerobot.policies.factory import make_policy, make_policy_config

    config = make_policy_config(policy_type)
    config.pretrained_path = checkpoint
    config.device = device
    model = make_policy(config)
    model.to(device)
    model.eval()
    return model  # type: ignore[no-any-return]
