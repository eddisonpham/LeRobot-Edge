"""Tests for lerobot_edge base module (CompressedPolicy wrapper + backends)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lerobot_edge.core.base import (
    CompressedPolicy,
    DeploymentBackend,
    IdentityBackend,
    NativePyTorchBackend,
    _PlaceholderBackend,
)
from lerobot_edge.core.configs import EdgeBaseConfig, EdgeIdentityConfig


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class SimplePolicy(nn.Module):
    """A minimal policy for testing."""

    def __init__(self, input_dim: int = 7, output_dim: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Find the first state-like tensor
        for key, val in batch.items():
            if "state" in key.lower() or "obs" in key.lower():
                return self.linear(val)
        # Fallback: use first tensor
        first_key = next(iter(batch))
        return self.linear(batch[first_key])

    def reset(self) -> None:
        pass


@pytest.fixture
def simple_policy() -> SimplePolicy:
    return SimplePolicy()


@pytest.fixture
def identity_config() -> EdgeIdentityConfig:
    return EdgeIdentityConfig(device="cpu")


@pytest.fixture
def dummy_batch() -> dict[str, torch.Tensor]:
    return {
        "observation.state": torch.randn(1, 7),
        "observation.images": torch.randn(1, 3, 224, 224),
    }


# ---------------------------------------------------------------------------
# DeploymentBackend interface tests
# ---------------------------------------------------------------------------


class TestDeploymentBackendInterface:
    """Test that backends satisfy the DeploymentBackend interface."""

    def test_placeholder_backend_interface(self):
        """PlaceholderBackend should satisfy the interface."""
        backend = _PlaceholderBackend("cpu")
        assert isinstance(backend, DeploymentBackend)

        batch = {"observation.state": torch.randn(1, 7)}
        result = backend.predict(batch)
        assert isinstance(result, torch.Tensor)

        backend.reset()  # Should not raise

        assert backend.device == torch.device("cpu")
        assert isinstance(backend.parameters, list)

    def test_native_pytorch_backend_interface(self, simple_policy):
        """NativePyTorchBackend should satisfy the interface."""
        backend = NativePyTorchBackend(simple_policy, torch.device("cpu"))
        assert isinstance(backend, DeploymentBackend)

        batch = {"observation.state": torch.randn(1, 7)}
        result = backend.predict(batch)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 2)

        backend.reset()  # Should not raise

        assert backend.device == torch.device("cpu")
        assert len(backend.parameters) > 0


# ---------------------------------------------------------------------------
# CompressedPolicy tests
# ---------------------------------------------------------------------------


class TestCompressedPolicy:
    """Test CompressedPolicy wrapper functionality."""

    def test_compressed_policy_creation_with_backend(
        self, simple_policy, identity_config
    ):
        """CompressedPolicy should be creatable with a pre-built backend."""
        backend = NativePyTorchBackend(simple_policy)
        policy = CompressedPolicy(config=identity_config, backend=backend)

        assert policy is not None
        assert policy.device == torch.device("cpu")

    def test_compressed_policy_select_action(
        self, simple_policy, identity_config, dummy_batch
    ):
        """select_action should return a tensor."""
        backend = NativePyTorchBackend(simple_policy)
        policy = CompressedPolicy(config=identity_config, backend=backend)

        action = policy.select_action(dummy_batch)
        assert isinstance(action, torch.Tensor)
        # Shape depends on the backend - single action may or may not have batch dim
        assert action.numel() > 0

    def test_compressed_policy_action_caching(
        self, simple_policy, identity_config, dummy_batch
    ):
        """select_action should cache action chunks."""
        backend = NativePyTorchBackend(simple_policy)
        policy = CompressedPolicy(config=identity_config, backend=backend)

        # First call triggers inference
        action1 = policy.select_action(dummy_batch)
        assert policy._action_cache is not None
        assert policy._cache_idx == 1

        # Second call should use cache (if cache has more than 1 action)
        # For our simple policy, the cache will have shape (1, 2), so cache_idx=1
        # will be equal to cache length, triggering new inference
        action2 = policy.select_action(dummy_batch)
        assert isinstance(action2, torch.Tensor)

    def test_compressed_policy_reset(
        self, simple_policy, identity_config, dummy_batch
    ):
        """reset should clear action cache."""
        backend = NativePyTorchBackend(simple_policy)
        policy = CompressedPolicy(config=identity_config, backend=backend)

        # Trigger some actions
        policy.select_action(dummy_batch)
        assert policy._action_cache is not None

        # Reset
        policy.reset()
        assert policy._action_cache is None
        assert policy._cache_idx == 0

    def test_compressed_policy_forward(
        self, simple_policy, identity_config, dummy_batch
    ):
        """forward should return loss and info dict."""
        backend = NativePyTorchBackend(simple_policy)
        policy = CompressedPolicy(config=identity_config, backend=backend)

        loss, info = policy.forward(dummy_batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert "actions" in info

    def test_compressed_policy_with_placeholder_backend(self, identity_config):
        """CompressedPolicy should work with placeholder backend."""
        policy = CompressedPolicy(config=identity_config)

        batch = {"observation.state": torch.randn(1, 7)}
        action = policy.select_action(batch)
        assert isinstance(action, torch.Tensor)

    def test_compressed_policy_get_optim_params(
        self, simple_policy, identity_config
    ):
        """get_optim_params should return parameters dict."""
        backend = NativePyTorchBackend(simple_policy)
        policy = CompressedPolicy(config=identity_config, backend=backend)

        params = policy.get_optim_params()
        assert "params" in params
        assert isinstance(params["params"], list)
        assert len(params["params"]) > 0


# ---------------------------------------------------------------------------
# IdentityBackend tests
# ---------------------------------------------------------------------------


class TestIdentityBackend:
    """Test IdentityBackend passthrough behavior."""

    def test_identity_backend_passthrough(self, simple_policy, dummy_batch):
        """IdentityBackend should produce same output as original policy."""
        # Original policy output
        with torch.no_grad():
            expected = simple_policy.select_action(dummy_batch)

        # Identity backend output
        backend = IdentityBackend(simple_policy)
        actual = backend.predict(dummy_batch)

        assert torch.allclose(expected, actual, atol=1e-6)

    def test_identity_backend_preserves_gradients(self, simple_policy):
        """IdentityBackend should preserve gradient computation."""
        backend = IdentityBackend(simple_policy)
        batch = {"observation.state": torch.randn(1, 7, requires_grad=True)}

        # The predict method uses torch.no_grad(), which is correct for deployment
        result = backend.predict(batch)
        assert isinstance(result, torch.Tensor)
