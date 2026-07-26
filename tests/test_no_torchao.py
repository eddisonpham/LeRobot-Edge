"""Tests verifying behavior when torchao is NOT installed.

These tests mock HAS_TORCHAO=False to ensure the fallback code paths
work correctly even when torchao is absent.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from lerobot_edge.compression.quantize import HAS_TORCHAO


class SimpleModel(nn.Module):
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
        return


@pytest.fixture
def simple_model():
    return SimpleModel()


class TestNoTorchaoPaths:
    """Verify code paths when torchao is not available."""

    def test_dynamic_int8_raises_import_error(self, simple_model):
        """dynamic_int8_quantize should raise ImportError without torchao."""
        from lerobot_edge.compression.quantize import dynamic_int8_quantize

        with (
            patch("lerobot_edge.compression.quantize.HAS_TORCHAO", False),
            pytest.raises(ImportError, match="torchao is required"),
        ):
            dynamic_int8_quantize(simple_model)

    def test_static_int8_raises_import_error(self, simple_model):
        """static_int8_quantize should raise ImportError without torchao."""
        from lerobot_edge.compression.quantize import static_int8_quantize

        with (
            patch("lerobot_edge.compression.quantize.HAS_TORCHAO", False),
            pytest.raises(ImportError, match="torchao is required"),
        ):
            static_int8_quantize(simple_model, {"x": torch.randn(1, 7)})

    def test_quantized_backend_raises_import_error_without_torchao(self, simple_model):
        """QuantizedBackend.from_policy should raise ImportError when torchao is absent."""
        from lerobot_edge.compression.quantize import QuantizedBackend
        from lerobot_edge.core.configs import EdgeQuantInt8Config

        config = EdgeQuantInt8Config(device="cpu")
        with (
            patch("lerobot_edge.compression.quantize.HAS_TORCHAO", False),
            pytest.raises(ImportError, match="torchao is required"),
        ):
            QuantizedBackend.from_policy(simple_model, config)

    def test_has_torchao_flag_reflects_reality(self):
        """HAS_TORCHAO should match actual torchao availability."""
        try:
            import torchao  # noqa: F401

            assert HAS_TORCHAO is True
        except ImportError:
            assert HAS_TORCHAO is False

    def test_dynamic_int8_succeeds_with_torchao(self, simple_model):
        """dynamic_int8_quantize should succeed when HAS_TORCHAO is True."""
        if not HAS_TORCHAO:
            pytest.skip("torchao not installed")
        from lerobot_edge.compression.quantize import dynamic_int8_quantize

        with patch("lerobot_edge.compression.quantize.HAS_TORCHAO", True):
            result = dynamic_int8_quantize(simple_model)
            assert result is not None

    def test_model_forward_works_with_mocked_no_torchao(self, simple_model):
        """Model forward pass should work even when HAS_TORCHAO is False."""
        with patch("lerobot_edge.compression.quantize.HAS_TORCHAO", False):
            x = torch.randn(1, 7)
            with torch.no_grad():
                out = simple_model(x)
            assert out.shape == (1, 2)
            assert not torch.isnan(out).any()

    def test_select_action_works_with_mocked_no_torchao(self, simple_model):
        """select_action should work even when HAS_TORCHAO is False."""
        with patch("lerobot_edge.compression.quantize.HAS_TORCHAO", False):
            batch = {"observation.state": torch.randn(1, 7)}
            with torch.no_grad():
                out = simple_model.select_action(batch)
            assert out.shape == (1, 2)
            assert not torch.isnan(out).any()
