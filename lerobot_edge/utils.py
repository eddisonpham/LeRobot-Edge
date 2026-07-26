"""Shared utilities for lerobot_edge."""

from __future__ import annotations

import math
import subprocess
import torch
import torch.nn as nn

__all__ = [
    "get_git_commit_hash",
    "measure_model_memory",
    "measure_peak_memory_mb",
    "sigmoid_scalar",
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
