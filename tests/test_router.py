"""Tests for lerobot_edge router module."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lerobot_edge.router import ConfidenceRouter
from lerobot_edge.base import NativePyTorchBackend


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class SimplePolicy(nn.Module):
    """A minimal policy for router testing."""

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
def edge_policy() -> SimplePolicy:
    return SimplePolicy()


@pytest.fixture
def cloud_policy() -> SimplePolicy:
    return SimplePolicy()


@pytest.fixture
def dummy_batch() -> dict[str, torch.Tensor]:
    return {"observation.state": torch.randn(1, 7)}


# ---------------------------------------------------------------------------
# ConfidenceRouter tests
# ---------------------------------------------------------------------------


class TestConfidenceRouter:
    """Test ConfidenceRouter functionality."""

    def test_router_creation(self, edge_policy, cloud_policy):
        """Router should be creatable with edge and cloud backends."""
        edge_backend = NativePyTorchBackend(edge_policy)
        cloud_backend = NativePyTorchBackend(cloud_policy)

        router = ConfidenceRouter(
            edge_backend=edge_backend,
            cloud_backend=cloud_backend,
            confidence_threshold=0.5,
        )

        assert router is not None
        assert router._threshold == 0.5

    def test_router_predict_returns_tuple(self, edge_policy, cloud_policy, dummy_batch):
        """Router predict should return (actions, info) tuple."""
        edge_backend = NativePyTorchBackend(edge_policy)
        cloud_backend = NativePyTorchBackend(cloud_policy)

        router = ConfidenceRouter(
            edge_backend=edge_backend,
            cloud_backend=cloud_backend,
        )

        result = router.predict(dummy_batch)
        assert isinstance(result, tuple)
        assert len(result) == 2

        actions, info = result
        assert isinstance(actions, torch.Tensor)
        assert isinstance(info, dict)

    def test_router_info_contains_expected_keys(self, edge_policy, cloud_policy, dummy_batch):
        """Router info dict should contain routing metadata."""
        edge_backend = NativePyTorchBackend(edge_policy)
        cloud_backend = NativePyTorchBackend(cloud_policy)

        router = ConfidenceRouter(
            edge_backend=edge_backend,
            cloud_backend=cloud_backend,
        )

        _, info = router.predict(dummy_batch)

        assert "confidence" in info
        assert "escalated" in info
        assert "escalation_rate" in info
        assert "total_inferences" in info
        assert "total_escalations" in info

    def test_router_stats(self, edge_policy, cloud_policy, dummy_batch):
        """Router stats should track routing statistics."""
        edge_backend = NativePyTorchBackend(edge_policy)
        cloud_backend = NativePyTorchBackend(cloud_policy)

        router = ConfidenceRouter(
            edge_backend=edge_backend,
            cloud_backend=cloud_backend,
        )

        # Run some inferences
        for _ in range(5):
            router.predict(dummy_batch)

        stats = router.stats
        assert stats["total_inferences"] == 5
        assert isinstance(stats["escalation_rate"], float)

    def test_router_reset(self, edge_policy, cloud_policy, dummy_batch):
        """Router reset should clear statistics."""
        edge_backend = NativePyTorchBackend(edge_policy)
        cloud_backend = NativePyTorchBackend(cloud_policy)

        router = ConfidenceRouter(
            edge_backend=edge_backend,
            cloud_backend=cloud_backend,
        )

        # Run some inferences
        router.predict(dummy_batch)
        assert router._total_inferences == 1

        # Reset
        router.reset()
        assert router._total_inferences == 0
        assert router._escalations == 0

    def test_router_without_cloud(self, edge_policy, dummy_batch):
        """Router should work without cloud backend (no escalation)."""
        edge_backend = NativePyTorchBackend(edge_policy)

        router = ConfidenceRouter(
            edge_backend=edge_backend,
            cloud_backend=None,
        )

        actions, info = router.predict(dummy_batch)
        assert isinstance(actions, torch.Tensor)
        assert info["escalated"] is False
        assert info["total_escalations"] == 0

    def test_compute_confidence(self, edge_policy):
        """_compute_confidence should return a value between 0 and 1."""
        edge_backend = NativePyTorchBackend(edge_policy)
        router = ConfidenceRouter(edge_backend=edge_backend)

        # Low variance -> high confidence
        actions_low_var = torch.ones(1, 2) * 0.5
        conf_low = router._compute_confidence(actions_low_var)
        assert 0 <= conf_low <= 1

        # High variance -> low confidence
        actions_high_var = torch.randn(10, 2) * 100
        conf_high = router._compute_confidence(actions_high_var)
        assert 0 <= conf_high <= 1
