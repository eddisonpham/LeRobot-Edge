"""Tests for evaluation metrics module."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lerobot_edge.evaluation import (
    OutputDivergence,
    QuantizationQualityReport,
    bootstrap_confidence_interval,
    compare_backends,
    measure_output_divergence,
)


class _SelectActionModel(nn.Module):
    """Minimal model with select_action for testing compare_backends."""

    def __init__(self, dim_in: int = 4, dim_out: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        for key in ("observation.state", "observation.images.front", "observation.images"):
            if key in batch:
                return self.linear(batch[key])
        return self.linear(next(iter(batch.values())))

    def reset(self) -> None:
        pass


class TestOutputDivergence:
    def test_perfect_match(self):
        outputs = [torch.randn(1, 2) for _ in range(3)]
        div = measure_output_divergence(outputs, [o.clone() for o in outputs])

        assert div.mse == pytest.approx(0.0, abs=1e-8)
        assert div.mae == pytest.approx(0.0, abs=1e-8)
        assert div.cosine_similarity == pytest.approx(1.0, abs=1e-5)
        assert div.max_abs_error == pytest.approx(0.0, abs=1e-8)
        assert div.degradation_pct == pytest.approx(0.0, abs=0.01)

    def test_known_divergence(self):
        orig = [torch.tensor([[1.0, 2.0, 3.0]])]
        quant = [torch.tensor([[1.1, 2.1, 3.1]])]

        div = measure_output_divergence(orig, quant)
        assert div.mse > 0
        assert div.mae > 0
        assert 0 < div.cosine_similarity < 1.0
        assert div.num_samples == 3

    def test_empty_outputs(self):
        div = measure_output_divergence([], [])
        assert div.mse == 0.0
        assert div.cosine_similarity == 0.0
        assert div.num_samples == 0

    def test_length_mismatch_raises(self):
        a = [torch.randn(1, 2)]
        b = [torch.randn(1, 2), torch.randn(1, 2)]
        with pytest.raises(ValueError, match="same length"):
            measure_output_divergence(a, b)

    def test_shape_mismatch_skips_pair(self):
        a = [torch.randn(1, 2)]
        b = [torch.randn(1, 3)]
        div = measure_output_divergence(a, b)
        assert div.num_samples == 0

    def test_degradation_pct_property(self):
        div = OutputDivergence(cosine_similarity=0.95)
        assert div.degradation_pct == pytest.approx(5.0)


class TestQuantizationQualityReport:
    def test_compression_ratio(self):
        report = QuantizationQualityReport(
            variant_name="test",
            original_size_mb=10.0,
            quantized_size_mb=5.0,
        )
        assert report.compression_ratio == pytest.approx(2.0)

    def test_memory_savings_pct(self):
        report = QuantizationQualityReport(
            variant_name="test",
            memory_original_mb=100.0,
            memory_quantized_mb=75.0,
        )
        assert report.memory_savings_pct == pytest.approx(25.0)

    def test_speedup_ratio(self):
        report = QuantizationQualityReport(
            variant_name="test",
            latency_original_ms=100.0,
            latency_quantized_ms=50.0,
        )
        assert report.speedup_ratio == pytest.approx(2.0)

    def test_to_dict(self):
        report = QuantizationQualityReport(
            variant_name="test",
            original_size_mb=10.0,
            quantized_size_mb=5.0,
            memory_original_mb=100.0,
            memory_quantized_mb=75.0,
            latency_original_ms=100.0,
            latency_quantized_ms=50.0,
            original_params=1000,
            quantized_params=1000,
        )
        d = report.to_dict()
        assert d["variant"] == "test"
        assert d["compression_ratio"] == 2.0
        assert d["memory_savings_pct"] == 25.0

    def test_zero_original_size(self):
        report = QuantizationQualityReport(variant_name="test")
        assert report.compression_ratio == 0.0
        assert report.memory_savings_pct == 0.0
        assert report.speedup_ratio == 0.0


class TestCompareBackends:
    def test_compare_same_model(self):
        model = _SelectActionModel(dim_in=4, dim_out=2)
        dummy_input = {"observation.state": torch.randn(1, 4)}

        report = compare_backends(model, model, dummy_input, num_samples=3)
        assert report.divergence.mse == pytest.approx(0.0, abs=1e-8)
        assert report.divergence.cosine_similarity == pytest.approx(1.0, abs=1e-5)
        assert report.original_params == report.quantized_params

    def test_compare_different_models(self):
        m1 = _SelectActionModel(dim_in=4, dim_out=2)
        m2 = _SelectActionModel(dim_in=4, dim_out=2)
        m2.load_state_dict(m1.state_dict())
        m2.linear.weight.data.add_(0.1)

        dummy_input = {"observation.state": torch.randn(1, 4)}
        report = compare_backends(m1, m2, dummy_input, num_samples=5)
        assert report.divergence.mse > 0
        assert report.divergence.cosine_similarity < 1.0


class TestBootstrapCI:
    def test_basic_ci(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, lower, upper = bootstrap_confidence_interval(data, confidence=0.95)
        assert mean == pytest.approx(3.0)
        assert lower < mean < upper

    def test_empty_data(self):
        mean, lower, upper = bootstrap_confidence_interval([])
        assert mean == 0.0
        assert lower == 0.0
        assert upper == 0.0

    def test_single_value(self):
        mean, lower, upper = bootstrap_confidence_interval([5.0])
        assert mean == pytest.approx(5.0)
        assert lower == pytest.approx(5.0)
        assert upper == pytest.approx(5.0)

    def test_narrow_ci_for_consistent_data(self):
        data = [1.0] * 100
        mean, lower, upper = bootstrap_confidence_interval(data, confidence=0.99)
        assert mean == pytest.approx(1.0)
        assert upper - lower < 0.01
