"""Tests for lerobot_edge distillation module."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lerobot_edge.distill import DistillationLoss, DistilledBackend, distill
from lerobot_edge.configs import EdgeDistilledConfig


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class SimplePolicy(nn.Module):
    """A minimal policy for distillation testing."""

    def __init__(self, input_dim: int = 7, output_dim: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        for key, val in batch.items():
            if "state" in key.lower():
                return self.forward(val)
        first_key = next(iter(batch))
        return self.forward(batch[first_key])

    def reset(self) -> None:
        pass


@pytest.fixture
def teacher() -> SimplePolicy:
    return SimplePolicy()


@pytest.fixture
def student() -> SimplePolicy:
    return SimplePolicy()


@pytest.fixture
def distill_config() -> EdgeDistilledConfig:
    return EdgeDistilledConfig(device="cpu")


# ---------------------------------------------------------------------------
# DistillationLoss tests
# ---------------------------------------------------------------------------


class TestDistillationLoss:
    """Test distillation loss computation."""

    def test_mse_only_loss(self):
        """Loss with alpha=0 should be pure MSE."""
        criterion = DistillationLoss(temperature=2.0, alpha=0.0)
        student_actions = torch.randn(4, 2)
        teacher_actions = torch.randn(4, 2)

        loss = criterion(student_actions, teacher_actions)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_kl_only_loss(self):
        """Loss with alpha=1 and logits should be pure KL."""
        criterion = DistillationLoss(temperature=2.0, alpha=1.0)
        student_actions = torch.randn(4, 2)
        teacher_actions = torch.randn(4, 2)
        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)

        loss = criterion(
            student_actions, teacher_actions,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
        )
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_combined_loss(self):
        """Loss with alpha between 0 and 1 should combine MSE and KL."""
        criterion = DistillationLoss(temperature=2.0, alpha=0.5)
        student_actions = torch.randn(4, 2)
        teacher_actions = torch.randn(4, 2)
        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)

        loss = criterion(
            student_actions, teacher_actions,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
        )
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_loss_grad_computation(self):
        """Loss should be differentiable."""
        criterion = DistillationLoss(temperature=2.0, alpha=0.5)
        student_actions = torch.randn(4, 2, requires_grad=True)
        teacher_actions = torch.randn(4, 2)

        loss = criterion(student_actions, teacher_actions)
        loss.backward()
        assert student_actions.grad is not None


# ---------------------------------------------------------------------------
# DistilledBackend tests
# ---------------------------------------------------------------------------


class TestDistilledBackend:
    """Test DistilledBackend functionality."""

    def test_distilled_backend_creation(self, teacher, student):
        """DistilledBackend should be creatable."""
        backend = DistilledBackend(student, teacher)
        assert backend is not None
        assert backend.teacher is teacher

    def test_distilled_backend_predict(self, teacher, student):
        """DistilledBackend predict should return valid actions."""
        backend = DistilledBackend(student, teacher)
        batch = {"observation.state": torch.randn(1, 7)}
        result = backend.predict(batch)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 2)

    def test_distilled_backend_teacher_frozen(self, teacher, student):
        """Teacher parameters should be frozen in DistilledBackend."""
        backend = DistilledBackend(student, teacher)
        for param in backend.teacher.parameters():
            assert not param.requires_grad

    def test_distilled_backend_device(self, teacher, student):
        """DistilledBackend should report correct device."""
        backend = DistilledBackend(student, teacher)
        assert backend.device == torch.device("cpu")

    def test_distilled_backend_reset(self, teacher, student):
        """DistilledBackend reset should not raise."""
        backend = DistilledBackend(student, teacher)
        backend.reset()  # Should not raise
