"""Core abstractions for lerobot_edge: backends, configs, and utilities."""

from __future__ import annotations

from lerobot_edge.core.base import (
    CUDAGraphBackend,
    CompiledBackend,
    CompressedPolicy,
    DeploymentBackend,
    IdentityBackend,
    NativePyTorchBackend,
)
from lerobot_edge.core.configs import (
    EdgeBaseConfig,
    EdgeDistilledConfig,
    EdgeDistilledOnnxInt8Config,
    EdgeIdentityConfig,
    EdgeOnnxFp32Config,
    EdgeOnnxInt8Config,
    EdgeQuantInt4Config,
    EdgeQuantInt8Config,
    EdgeQuantBnbInt8Config,
    EdgeQuantBnbNf4Config,
    EdgeQuantBnbFp4Config,
)
from lerobot_edge.core.router import ConfidenceRouter
from lerobot_edge.core.utils import (
    build_dummy_input,
    get_git_commit_hash,
    load_policy_from_checkpoint,
    measure_cuda_memory_mb,
    measure_model_memory,
    measure_peak_memory_mb,
)

__all__ = [
    "DeploymentBackend",
    "NativePyTorchBackend",
    "IdentityBackend",
    "CompiledBackend",
    "CUDAGraphBackend",
    "CompressedPolicy",
    "EdgeBaseConfig",
    "EdgeIdentityConfig",
    "EdgeQuantInt8Config",
    "EdgeQuantInt4Config",
    "EdgeQuantBnbInt8Config",
    "EdgeQuantBnbNf4Config",
    "EdgeQuantBnbFp4Config",
    "EdgeOnnxFp32Config",
    "EdgeOnnxInt8Config",
    "EdgeDistilledConfig",
    "EdgeDistilledOnnxInt8Config",
    "ConfidenceRouter",
    "build_dummy_input",
    "get_git_commit_hash",
    "load_policy_from_checkpoint",
    "measure_model_memory",
    "measure_peak_memory_mb",
    "measure_cuda_memory_mb",
]
