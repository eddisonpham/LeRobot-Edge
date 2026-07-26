"""Tests for the quality divergence gate."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lerobot_edge.evaluation.gate import QualityGate, QualityGateResult


class SimplePolicy(nn.Module):
    def __init__(self, dim: int = 7, out: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        for key, val in batch.items():
            if "state" in key.lower():
                return self.forward(val)
        return self.forward(next(iter(batch.values())))

    def reset(self) -> None:
        pass


@pytest.fixture
def simple_policy() -> SimplePolicy:
    return SimplePolicy()


@pytest.fixture
def identity_policy() -> SimplePolicy:
    return SimplePolicy()


@pytest.fixture
def dummy_batch() -> dict[str, torch.Tensor]:
    return {"observation.state": torch.randn(1, 7)}


class TestQualityGate:
    def test_identical_models_pass(self, simple_policy, dummy_batch):
        gate = QualityGate(min_cosine_similarity=0.999, num_samples=5)
        result = gate.check(simple_policy, simple_policy, dummy_batch)
        assert result.passed
        assert result.cosine_similarity == pytest.approx(1.0, abs=1e-5)
        assert "PASSED" in result.message

    def test_different_models_fail(self, simple_policy, dummy_batch):
        other = SimplePolicy()
        other.linear.weight.data.add_(1.0)
        gate = QualityGate(min_cosine_similarity=0.999, num_samples=5)
        result = gate.check(simple_policy, other, dummy_batch)
        assert not result.passed
        assert result.cosine_similarity < 0.999
        assert "FAILED" in result.message

    def test_threshold_configurable(self, simple_policy, dummy_batch):
        gate = QualityGate(min_cosine_similarity=0.0, num_samples=5)
        result = gate.check(simple_policy, simple_policy, dummy_batch)
        assert result.passed
        assert result.threshold_cosine == 0.0

    def test_mse_threshold_passes(self, simple_policy, dummy_batch):
        gate = QualityGate(min_cosine_similarity=0.0, max_mse=0.0, num_samples=5)
        result = gate.check(simple_policy, simple_policy, dummy_batch)
        assert result.passed
        assert result.mse < 1e-10

    def test_mse_threshold_fails(self, simple_policy, dummy_batch):
        other = SimplePolicy()
        other.linear.weight.data.add_(5.0)
        gate = QualityGate(min_cosine_similarity=0.0, max_mse=1e-10, num_samples=5)
        result = gate.check(simple_policy, other, dummy_batch)
        assert not result.passed
        assert "FAILED" in result.message

    def test_result_fields(self, simple_policy, dummy_batch):
        gate = QualityGate(min_cosine_similarity=0.999, num_samples=5)
        result = gate.check(simple_policy, simple_policy, dummy_batch)
        assert isinstance(result, QualityGateResult)
        assert isinstance(result.passed, bool)
        assert isinstance(result.cosine_similarity, float)
        assert isinstance(result.mse, float)
        assert isinstance(result.message, str)

    def test_gate_with_backend_predict(self, simple_policy, dummy_batch):
        from lerobot_edge.core.base import NativePyTorchBackend

        backend = NativePyTorchBackend(simple_policy)
        gate = QualityGate(min_cosine_similarity=0.999, num_samples=5)
        result = gate.check(backend._policy, backend._policy, dummy_batch)
        assert result.passed
        assert result.cosine_similarity == pytest.approx(1.0, abs=1e-5)
