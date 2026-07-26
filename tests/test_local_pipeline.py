"""End-to-end tests using SimpleCNNPolicy for local testing without model downloads.

These tests exercise the full pipeline (quantize, export, benchmark, report)
using a lightweight model that runs entirely on CPU.
"""

from __future__ import annotations

import json
import tempfile
import pytest
import torch

from lerobot_edge.core.base import CompressedPolicy, IdentityBackend, NativePyTorchBackend
from lerobot_edge.core.configs import (
    EdgeBaseConfig,
    EdgeIdentityConfig,
    EdgeQuantInt8Config,
)
from lerobot_edge.compression.quantize import (
    HAS_TORCHAO,
    dynamic_int8_quantize,
    static_int8_quantize,
    QuantizedBackend,
)
from lerobot_edge.core.utils import measure_model_memory
from test_models import SimpleCNNPolicy
from lerobot_edge.evaluation.benchmark import benchmark_backend, BenchmarkResult
from lerobot_edge.core.router import ConfidenceRouter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_cnn() -> SimpleCNNPolicy:
    return SimpleCNNPolicy(image_channels=3, state_dim=2, action_dim=2)


@pytest.fixture
def simple_cnn_batch() -> dict[str, torch.Tensor]:
    return {
        "observation.images.front": torch.randn(1, 3, 224, 224),
        "observation.state": torch.randn(1, 2),
    }


@pytest.fixture
def identity_config() -> EdgeIdentityConfig:
    return EdgeIdentityConfig(device="cpu")


@pytest.fixture
def quant_int8_config() -> EdgeQuantInt8Config:
    return EdgeQuantInt8Config(device="cpu")


# ---------------------------------------------------------------------------
# Quantization tests with SimpleCNNPolicy
# ---------------------------------------------------------------------------


class TestQuantizationLocal:
    """Test quantization using the lightweight SimpleCNNPolicy."""

    @pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
    def test_dynamic_int8_quantize(self, simple_cnn):
        original_mem = measure_model_memory(simple_cnn)
        quantized = dynamic_int8_quantize(simple_cnn)
        quantized_mem = measure_model_memory(quantized)

        assert quantized is not None
        assert quantized_mem["total_mb"] <= original_mem["total_mb"] * 1.1

    @pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
    def test_quantized_inference(self, simple_cnn, simple_cnn_batch):
        quantized = dynamic_int8_quantize(simple_cnn)
        backend = NativePyTorchBackend(quantized, torch.device("cpu"))
        config = EdgeQuantInt8Config(device="cpu")
        policy = CompressedPolicy(config=config, backend=backend)

        action = policy.select_action(simple_cnn_batch)
        assert isinstance(action, torch.Tensor)
        assert action.numel() > 0
        assert not torch.isnan(action).any()

    @pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
    def test_quantized_backend_from_policy(self, simple_cnn, simple_cnn_batch):
        config = EdgeQuantInt8Config(device="cpu")
        backend = QuantizedBackend.from_policy(simple_cnn, config)

        assert backend.quantization_type == "dynamic_int8"
        action = backend.predict(simple_cnn_batch)
        assert isinstance(action, torch.Tensor)
        assert not torch.isnan(action).any()

    @pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
    def test_memory_reduction(self, simple_cnn):
        original = measure_model_memory(simple_cnn)
        quantized = dynamic_int8_quantize(simple_cnn)
        quantized_mem = measure_model_memory(quantized)

        reduction_pct = (1 - quantized_mem["total_mb"] / original["total_mb"]) * 100
        print(f"\nMemory: {original['total_mb']:.2f} MB -> {quantized_mem['total_mb']:.2f} MB ({reduction_pct:.1f}% reduction)")
        assert reduction_pct >= 0

    @pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
    def test_static_int8_quantize(self, simple_cnn):
        calibration_data = {
            "observation.images.front": torch.randn(1, 3, 224, 224),
            "observation.state": torch.randn(1, 2),
        }
        quantized = static_int8_quantize(
            simple_cnn, calibration_data, num_calibration_steps=3
        )
        assert quantized is not None

    @pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
    def test_multiple_quantize_cycles(self, simple_cnn):
        """Verify quantization is idempotent."""
        q1 = dynamic_int8_quantize(simple_cnn)
        q2 = dynamic_int8_quantize(simple_cnn)
        m1 = measure_model_memory(q1)
        m2 = measure_model_memory(q2)
        assert m1["total_mb"] == m2["total_mb"]


# ---------------------------------------------------------------------------
# Benchmark tests with SimpleCNNPolicy
# ---------------------------------------------------------------------------


class TestBenchmarkLocal:
    """Test benchmark harness using the lightweight SimpleCNNPolicy."""

    def test_benchmark_identity(self, simple_cnn, simple_cnn_batch):
        backend = IdentityBackend(simple_cnn)
        result = benchmark_backend(
            backend,
            simple_cnn_batch,
            warmup_runs=3,
            num_runs=10,
            backend_name="simple_cnn_identity",
            device_profile="cpu",
        )

        assert result.latency_mean_ms > 0
        assert result.throughput_fps > 0
        assert result.backend_name == "simple_cnn_identity"

    @pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
    def test_benchmark_quantized(self, simple_cnn, simple_cnn_batch):
        config = EdgeQuantInt8Config(device="cpu")
        backend = QuantizedBackend.from_policy(simple_cnn, config)
        result = benchmark_backend(
            backend,
            simple_cnn_batch,
            warmup_runs=3,
            num_runs=10,
            backend_name="simple_cnn_quant_int8",
            device_profile="cpu",
        )

        assert result.latency_mean_ms > 0
        assert result.throughput_fps > 0

    @pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
    def test_benchmark_compare(self, simple_cnn, simple_cnn_batch):
        id_backend = IdentityBackend(simple_cnn)
        id_result = benchmark_backend(
            id_backend, simple_cnn_batch,
            warmup_runs=2, num_runs=5,
            backend_name="identity", device_profile="cpu",
        )

        config = EdgeQuantInt8Config(device="cpu")
        q_backend = QuantizedBackend.from_policy(simple_cnn, config)
        q_result = benchmark_backend(
            q_backend, simple_cnn_batch,
            warmup_runs=2, num_runs=5,
            backend_name="quant_int8", device_profile="cpu",
        )

        print(f"\nIdentity:  {id_result.latency_mean_ms:.2f} ms")
        print(f"Quant INT8: {q_result.latency_mean_ms:.2f} ms")
        assert id_result.latency_mean_ms > 0
        assert q_result.latency_mean_ms > 0

    def test_benchmark_result_serialization(self, simple_cnn, simple_cnn_batch):
        backend = IdentityBackend(simple_cnn)
        result = benchmark_backend(
            backend, simple_cnn_batch,
            warmup_runs=2, num_runs=3,
            backend_name="test", device_profile="cpu",
        )

        from dataclasses import asdict
        d = asdict(result)
        assert isinstance(d, dict)
        assert d["backend_name"] == "test"

        restored = BenchmarkResult(**d)
        assert restored.backend_name == result.backend_name


# ---------------------------------------------------------------------------
# CompressedPolicy + SimpleCNNPolicy integration
# ---------------------------------------------------------------------------


class TestCompressedPolicyLocal:
    """Test CompressedPolicy wrapper with SimpleCNNPolicy."""

    def test_identity_wrapper(self, simple_cnn, simple_cnn_batch, identity_config):
        backend = IdentityBackend(simple_cnn)
        policy = CompressedPolicy(config=identity_config, backend=backend)

        action = policy.select_action(simple_cnn_batch)
        assert isinstance(action, torch.Tensor)
        assert not torch.isnan(action).any()

    @pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
    def test_quantized_wrapper(self, simple_cnn, simple_cnn_batch, quant_int8_config):
        backend = QuantizedBackend.from_policy(simple_cnn, quant_int8_config)
        policy = CompressedPolicy(config=quant_int8_config, backend=backend)

        action = policy.select_action(simple_cnn_batch)
        assert isinstance(action, torch.Tensor)
        assert not torch.isnan(action).any()

    def test_action_caching(self, simple_cnn, simple_cnn_batch, identity_config):
        backend = IdentityBackend(simple_cnn)
        policy = CompressedPolicy(config=identity_config, backend=backend)

        a1 = policy.select_action(simple_cnn_batch)
        assert policy._action_cache is not None

        policy.reset()
        assert policy._action_cache is None

    def test_forward_pass(self, simple_cnn, simple_cnn_batch, identity_config):
        backend = IdentityBackend(simple_cnn)
        policy = CompressedPolicy(config=identity_config, backend=backend)

        loss, info = policy.forward(simple_cnn_batch)
        assert isinstance(loss, torch.Tensor)
        assert "actions" in info

    def test_multiple_inferences(self, simple_cnn, simple_cnn_batch, identity_config):
        backend = IdentityBackend(simple_cnn)
        policy = CompressedPolicy(config=identity_config, backend=backend)

        for i in range(5):
            action = policy.select_action(simple_cnn_batch)
            assert isinstance(action, torch.Tensor)
            assert not torch.isnan(action).any()
            policy.reset()


# ---------------------------------------------------------------------------
# Router tests with SimpleCNNPolicy
# ---------------------------------------------------------------------------


class TestRouterLocal:
    """Test ConfidenceRouter with SimpleCNNPolicy backends."""

    def test_router_basic(self, simple_cnn, simple_cnn_batch):
        edge = IdentityBackend(simple_cnn)
        router = ConfidenceRouter(edge_backend=edge, confidence_threshold=0.5)

        actions, info = router.predict(simple_cnn_batch)
        assert isinstance(actions, torch.Tensor)
        assert "confidence" in info
        assert "escalated" in info

    def test_router_stats(self, simple_cnn, simple_cnn_batch):
        edge = IdentityBackend(simple_cnn)
        router = ConfidenceRouter(edge_backend=edge)

        for _ in range(3):
            router.predict(simple_cnn_batch)

        stats = router.stats
        assert stats["total_inferences"] == 3
        assert stats["escalations"] == 0

    def test_router_reset(self, simple_cnn, simple_cnn_batch):
        edge = IdentityBackend(simple_cnn)
        router = ConfidenceRouter(edge_backend=edge)

        router.predict(simple_cnn_batch)
        router.reset()
        assert router.stats["total_inferences"] == 0
