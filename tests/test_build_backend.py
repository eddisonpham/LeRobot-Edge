"""Edge case tests for _build_backend_from_checkpoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from lerobot_edge.core.base import (
    CompressedPolicy,
    IdentityBackend,
    _PlaceholderBackend,
)
from lerobot_edge.core.configs import (
    EdgeBaseConfig,
    EdgeIdentityConfig,
    EdgeOnnxFp32Config,
    EdgeOnnxInt8Config,
    EdgeQuantInt8Config,
)


class SimplePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(7, 2)

    def forward(self, x):
        return self.linear(x)

    def select_action(self, batch):
        for val in batch.values():
            if isinstance(val, torch.Tensor) and val.dim() == 2:
                return self.linear(val)
        return self.linear(list(batch.values())[0])

    def reset(self):
        pass


class TestBuildBackendNoSource:
    def test_returns_placeholder_when_no_source(self):
        config = EdgeIdentityConfig(device="cpu", source_pretrained_path=None)
        policy = CompressedPolicy.__new__(CompressedPolicy)
        policy.config = config
        backend = policy._build_backend_from_checkpoint("", config)
        assert isinstance(backend, _PlaceholderBackend)

    def test_returns_placeholder_when_path_empty(self):
        config = EdgeIdentityConfig(device="cpu")
        policy = CompressedPolicy.__new__(CompressedPolicy)
        policy.config = config
        backend = policy._build_backend_from_checkpoint("", config)
        assert isinstance(backend, _PlaceholderBackend)

    def test_returns_placeholder_when_make_policy_fails(self):
        config = EdgeIdentityConfig(device="cpu", source_pretrained_path="nonexistent")
        policy = CompressedPolicy.__new__(CompressedPolicy)
        policy.config = config
        with patch("lerobot.policies.factory.make_policy", side_effect=RuntimeError("load failed")):
            backend = policy._build_backend_from_checkpoint("nonexistent", config)
        assert isinstance(backend, _PlaceholderBackend)

    def test_returns_placeholder_when_factory_import_fails(self):
        config = EdgeIdentityConfig(device="cpu", source_pretrained_path="dummy")
        policy = CompressedPolicy.__new__(CompressedPolicy)
        policy.config = config
        with patch.dict("sys.modules", {"lerobot.policies.factory": None}):
            backend = policy._build_backend_from_checkpoint("dummy", config)
        assert isinstance(backend, _PlaceholderBackend)


class TestBuildBackendQuantization:
    def test_quantize_dynamic_false_creates_identity(self):
        config = EdgeBaseConfig(
            device="cpu",
            source_pretrained_path="dummy",
            quantize_dynamic=False,
        )
        policy = CompressedPolicy.__new__(CompressedPolicy)
        policy.config = config
        mock_policy = SimplePolicy()
        with (
            patch("lerobot.policies.factory.make_policy", return_value=mock_policy),
            patch("lerobot.policies.factory.make_policy_config", return_value=MagicMock()),
        ):
            backend = policy._build_backend_from_checkpoint("dummy", config)
        assert isinstance(backend, IdentityBackend)

    def test_identity_config_quantize_dynamic_true_creates_quantized(self):
        config = EdgeIdentityConfig(device="cpu", source_pretrained_path="dummy")
        policy = CompressedPolicy.__new__(CompressedPolicy)
        policy.config = config
        mock_policy = SimplePolicy()
        with (
            patch("lerobot.policies.factory.make_policy", return_value=mock_policy),
            patch("lerobot.policies.factory.make_policy_config", return_value=MagicMock()),
        ):
            backend = policy._build_backend_from_checkpoint("dummy", config)
        from lerobot_edge.compression.quantize import HAS_TORCHAO, QuantizedBackend

        if HAS_TORCHAO:
            assert isinstance(backend, QuantizedBackend)
        else:
            # Without torchao, falls back to IdentityBackend gracefully
            assert isinstance(backend, IdentityBackend)


class TestBuildBackendFailureFallback:
    def test_onnx_export_failure_falls_back_to_identity(self):
        config = EdgeOnnxInt8Config(device="cpu", source_pretrained_path="dummy")
        policy = CompressedPolicy.__new__(CompressedPolicy)
        policy.config = config
        mock_policy = SimplePolicy()
        with (
            patch("lerobot.policies.factory.make_policy", return_value=mock_policy),
            patch("lerobot.policies.factory.make_policy_config", return_value=MagicMock()),
            patch(
                "lerobot_edge.export.onnx.export_policy_to_onnx",
                side_effect=RuntimeError("export failed"),
            ),
        ):
            backend = policy._build_backend_from_checkpoint("dummy", config)
        assert isinstance(backend, IdentityBackend)

    def test_fp32_onnx_export_failure_falls_back_to_identity(self):
        config = EdgeOnnxFp32Config(device="cpu", source_pretrained_path="dummy")
        policy = CompressedPolicy.__new__(CompressedPolicy)
        policy.config = config
        mock_policy = SimplePolicy()
        with (
            patch("lerobot.policies.factory.make_policy", return_value=mock_policy),
            patch("lerobot.policies.factory.make_policy_config", return_value=MagicMock()),
            patch(
                "lerobot_edge.export.onnx.export_policy_to_onnx",
                side_effect=RuntimeError("export failed"),
            ),
        ):
            backend = policy._build_backend_from_checkpoint("dummy", config)
        assert isinstance(backend, IdentityBackend)

    def test_quantization_failure_falls_back_to_identity(self):
        config = EdgeQuantInt8Config(device="cpu", source_pretrained_path="dummy")
        policy = CompressedPolicy.__new__(CompressedPolicy)
        policy.config = config
        mock_policy = SimplePolicy()
        with (
            patch("lerobot.policies.factory.make_policy", return_value=mock_policy),
            patch("lerobot.policies.factory.make_policy_config", return_value=MagicMock()),
            patch(
                "lerobot_edge.compression.quantize.QuantizedBackend.from_policy",
                side_effect=RuntimeError("quantize failed"),
            ),
        ):
            backend = policy._build_backend_from_checkpoint("dummy", config)
        assert isinstance(backend, IdentityBackend)


class TestBuildBackendDeviceHandling:
    def test_cpu_device(self):
        config = EdgeIdentityConfig(device="cpu")
        policy = CompressedPolicy.__new__(CompressedPolicy)
        policy.config = config
        backend = policy._build_backend_from_checkpoint("", config)
        assert backend.device == torch.device("cpu")

    def test_default_device_when_none(self):
        config = EdgeIdentityConfig()
        config.device = None
        policy = CompressedPolicy.__new__(CompressedPolicy)
        policy.config = config
        backend = policy._build_backend_from_checkpoint("", config)
        assert backend.device == torch.device("cpu")
