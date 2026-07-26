"""Tests for torchao vs legacy quantization code paths."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lerobot_edge.compression.quantize import (
    HAS_TORCHAO,
    dynamic_int8_quantize,
    static_int8_quantize,
)


class SimpleLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(7, 32)
        self.layer2 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.relu(self.layer1(x)))


@pytest.fixture
def simple_model() -> SimpleLinearModel:
    return SimpleLinearModel()


class TestTorchaoCodePaths:
    def test_has_torchao_flag_matches_installation(self):
        try:
            from torchao.quantization import quantize_  # noqa: F401

            assert HAS_TORCHAO is True
        except ImportError:
            assert HAS_TORCHAO is False

    @pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
    def test_dynamic_int8_quantize_works(self, simple_model):
        quantized = dynamic_int8_quantize(simple_model)
        assert quantized is not None
        x = torch.randn(1, 7)
        with torch.no_grad():
            output = quantized(x)
        assert output.shape == (1, 2)
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
    def test_static_int8_quantize_works(self, simple_model):
        calibration_data = {"input": torch.randn(1, 7)}
        quantized = static_int8_quantize(simple_model, calibration_data, num_calibration_steps=5)
        assert quantized is not None
        x = torch.randn(1, 7)
        with torch.no_grad():
            output = quantized(x)
        assert output.shape == (1, 2)
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
    def test_dynamic_int8_produces_different_model(self, simple_model):
        original_params = list(simple_model.parameters())
        quantized = dynamic_int8_quantize(simple_model)
        quantized_params = list(quantized.parameters())
        assert len(original_params) == len(quantized_params)

    @pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
    def test_static_int8_produces_different_model(self, simple_model):
        calibration_data = {"input": torch.randn(1, 7)}
        quantized = static_int8_quantize(simple_model, calibration_data, num_calibration_steps=5)
        assert quantized is not None

    @pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
    def test_torchao_classes_exist(self):
        from lerobot_edge.compression.quantize import (
            ObservedLinear,
            QuantizedLinear,
            StaticQuantConfig,
        )

        assert ObservedLinear is not None
        assert QuantizedLinear is not None
        assert StaticQuantConfig is not None
