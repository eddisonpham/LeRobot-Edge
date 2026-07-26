"""Shared utilities for lerobot_edge."""

from __future__ import annotations

import importlib
import logging
import math
import subprocess
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "get_git_commit_hash",
    "measure_model_memory",
    "measure_peak_memory_mb",
    "sigmoid_scalar",
    "build_dummy_input",
    "load_policy_from_checkpoint",
]


def get_git_commit_hash() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def measure_model_memory(model: nn.Module) -> dict[str, float]:
    """Measure the memory footprint of a model."""
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
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
    """Measure current peak memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024
        except (ImportError, AttributeError):
            return 0.0


def sigmoid_scalar(x: float) -> float:
    """Compute sigmoid of a scalar without creating tensors."""
    return 1.0 / (1.0 + math.exp(-x))


def build_dummy_input(policy: nn.Module, device: torch.device) -> dict[str, torch.Tensor]:
    """Build dummy input batch from policy's expected features."""
    dummy_input = {}
    if hasattr(policy, 'config') and hasattr(policy.config, 'input_features'):
        for name, feature in policy.config.input_features.items():
            shape = list(feature.shape) if hasattr(feature, 'shape') else [1, 3, 224, 224]
            if len(shape) == 0:
                shape = [1]
            elif len(shape) == 1 and shape[0] != 1:
                shape.insert(0, 1)
            dummy_input[name] = torch.randn(shape, device=device)

    if not dummy_input:
        dummy_input = {
            "observation.images.front": torch.randn(1, 3, 224, 224, device=device),
            "observation.state": torch.randn(1, 2, device=device),
        }

    return dummy_input


def load_policy_from_checkpoint(
    checkpoint: str,
    policy_type: str = "smolvla",
    device: str = "cpu",
) -> nn.Module:
    """Load a LeRobot policy from a checkpoint path or HuggingFace Hub ID.

    Tries from_pretrained for known architectures first, then falls back
    to the factory method for other policy types.
    """
    _KNOWN_ARCH = {
        "smolvla": "lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy",
    }

    if policy_type in _KNOWN_ARCH:
        try:
            mod_path, cls_name = _KNOWN_ARCH[policy_type].rsplit(".", 1)
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            logger.info("Loading %s via from_pretrained(%s)", policy_type, checkpoint)
            model = cls.from_pretrained(checkpoint)
            model.to(device)
            model.eval()
            return model
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
    return model
