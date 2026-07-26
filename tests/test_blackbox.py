"""Blackbox integration tests for lerobot_edge as a library API.

These tests verify that all public interfaces work correctly when the
package is imported and used as a user would use it.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from lerobot_edge.compression.quantize import (
    HAS_TORCHAO,
    QuantizedBackend,
    dynamic_int8_quantize,
)
from lerobot_edge.core.base import (
    CompressedPolicy,
    DeploymentBackend,
    IdentityBackend,
    NativePyTorchBackend,
)
from lerobot_edge.core.configs import (
    EdgeBaseConfig,
    EdgeDistilledConfig,
    EdgeIdentityConfig,
    EdgeOnnxFp32Config,
    EdgeOnnxInt8Config,
    EdgeQuantInt8Config,
)
from lerobot_edge.core.utils import (
    build_dummy_input,
    get_git_commit_hash,
    measure_model_memory,
    measure_peak_memory_mb,
    sigmoid_scalar,
)
from lerobot_edge.evaluation.gate import QualityGate, QualityGateResult
from lerobot_edge.evaluation.metrics import (
    OutputDivergence,
    QuantizationQualityReport,
    bootstrap_confidence_interval,
    compare_backends,
    measure_output_divergence,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class MinimalPolicy(nn.Module):
    """Minimal policy satisfying LeRobot's interface for testing."""

    def __init__(self, input_dim: int = 7, output_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        for val in batch.values():
            if isinstance(val, torch.Tensor) and val.dim() == 2:
                return self.linear(val)
        return self.linear(list(batch.values())[0])

    def reset(self) -> None:
        return


@pytest.fixture
def policy() -> MinimalPolicy:
    return MinimalPolicy()


@pytest.fixture
def batch() -> dict[str, torch.Tensor]:
    return {
        "observation.state": torch.randn(1, 7),
        "observation.images": torch.randn(1, 3, 224, 224),
    }


@pytest.fixture
def config() -> EdgeIdentityConfig:
    return EdgeIdentityConfig(device="cpu")


# ---------------------------------------------------------------------------
# Public API: core.base
# ---------------------------------------------------------------------------


class TestCoreBaseAPI:
    """Verify DeploymentBackend, IdentityBackend, NativePyTorchBackend, CompressedPolicy."""

    def test_deployment_backend_is_abstract(self):
        assert hasattr(DeploymentBackend, "predict")
        assert hasattr(DeploymentBackend, "reset")

    def test_identity_backend_predict(self, policy, batch):
        backend = IdentityBackend(policy)
        out = backend.predict(batch)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 2)

    def test_identity_backend_properties(self, policy):
        backend = IdentityBackend(policy, torch.device("cpu"))
        assert backend.device == torch.device("cpu")
        assert len(backend.parameters) > 0

    def test_native_pytorch_backend(self, policy, batch):
        backend = NativePyTorchBackend(policy)
        out = backend.predict(batch)
        assert isinstance(out, torch.Tensor)

    def test_compressed_policy_select_action(self, policy, config, batch):
        backend = IdentityBackend(policy)
        cp = CompressedPolicy(config=config, backend=backend)
        action = cp.select_action(batch)
        assert isinstance(action, torch.Tensor)
        assert action.numel() > 0

    def test_compressed_policy_forward(self, policy, config, batch):
        backend = IdentityBackend(policy)
        cp = CompressedPolicy(config=config, backend=backend)
        loss, info = cp.forward(batch)
        assert isinstance(loss, torch.Tensor)
        assert "actions" in info

    def test_compressed_policy_reset(self, policy, config, batch):
        backend = IdentityBackend(policy)
        cp = CompressedPolicy(config=config, backend=backend)
        cp.select_action(batch)
        cp.reset()
        assert cp._action_cache is None

    def test_compressed_policy_placeholder(self, config):
        cp = CompressedPolicy(config=config)
        action = cp.select_action({"observation.state": torch.randn(1, 7)})
        assert isinstance(action, torch.Tensor)


# ---------------------------------------------------------------------------
# Public API: core.configs
# ---------------------------------------------------------------------------


class TestCoreConfigsAPI:
    """Verify all config types are registered and instantiable."""

    CONFIG_TYPES = [
        ("edge_identity", EdgeIdentityConfig),
        ("edge_quant_int8", EdgeQuantInt8Config),
        ("edge_onnx_fp32", EdgeOnnxFp32Config),
        ("edge_onnx_int8", EdgeOnnxInt8Config),
        ("edge_distilled", EdgeDistilledConfig),
    ]

    @pytest.mark.parametrize("type_name,cls", CONFIG_TYPES)
    def test_config_instantiation(self, type_name, cls):
        cfg = cls(device="cpu")
        assert cfg.type == type_name

    def test_edge_base_config_properties(self):
        cfg = EdgeBaseConfig(device="cpu")
        assert cfg.observation_delta_indices is None
        assert cfg.action_delta_indices is None
        assert cfg.reward_delta_indices is None


# ---------------------------------------------------------------------------
# Public API: core.utils
# ---------------------------------------------------------------------------


class TestCoreUtilsAPI:
    """Verify utility functions work correctly."""

    def test_get_git_commit_hash(self):
        h = get_git_commit_hash()
        assert isinstance(h, str)

    def test_measure_model_memory(self, policy):
        mem = measure_model_memory(policy)
        assert mem["total_mb"] > 0
        assert mem["num_parameters"] > 0

    def test_measure_peak_memory_mb(self):
        mb = measure_peak_memory_mb()
        assert isinstance(mb, float)

    def test_sigmoid_scalar(self):
        assert abs(sigmoid_scalar(0.0) - 0.5) < 1e-6
        assert sigmoid_scalar(100.0) > 0.99
        assert sigmoid_scalar(-100.0) < 0.01

    def test_build_dummy_input(self, policy):
        dummy = build_dummy_input(policy, torch.device("cpu"))
        assert isinstance(dummy, dict)
        assert all(isinstance(v, torch.Tensor) for v in dummy.values())


# ---------------------------------------------------------------------------
# Public API: compression.quantize
# ---------------------------------------------------------------------------


class TestCompressionQuantizeAPI:
    """Verify quantization functions and QuantizedBackend."""

    @pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
    def test_dynamic_int8_quantize(self, policy):
        quantized = dynamic_int8_quantize(policy)
        assert quantized is not None
        out = quantized(torch.randn(1, 7))
        assert out.shape == (1, 2)

    @pytest.mark.skipif(not HAS_TORCHAO, reason="torchao not installed")
    def test_quantized_backend_from_policy(self, policy):
        cfg = EdgeQuantInt8Config(device="cpu")
        backend = QuantizedBackend.from_policy(policy, cfg)
        assert backend.quantization_type == "dynamic_int8"
        out = backend.predict({"observation.state": torch.randn(1, 7)})
        assert isinstance(out, torch.Tensor)


# ---------------------------------------------------------------------------
# Public API: evaluation.metrics
# ---------------------------------------------------------------------------


class TestEvaluationMetricsAPI:
    """Verify metrics functions and dataclasses."""

    def test_measure_output_divergence(self):
        orig = [torch.randn(2, 5) for _ in range(5)]
        quant = [o + 0.01 * torch.randn_like(o) for o in orig]
        div = measure_output_divergence(orig, quant)
        assert isinstance(div, OutputDivergence)
        assert 0.0 <= div.cosine_similarity <= 1.0
        assert div.mse >= 0.0

    def test_divergence_degradation_pct(self):
        div = OutputDivergence(cosine_similarity=0.995)
        assert abs(div.degradation_pct - 0.5) < 1e-6

    def test_compare_backends(self, policy, batch):
        quantized = dynamic_int8_quantize(policy) if HAS_TORCHAO else policy
        report = compare_backends(policy, quantized, batch, num_samples=3)
        assert isinstance(report, QuantizationQualityReport)
        assert report.latency_original_ms > 0
        assert report.compression_ratio >= 0

    def test_bootstrap_confidence_interval(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, lo, hi = bootstrap_confidence_interval(data, n_bootstrap=100)
        assert 2.0 <= mean <= 4.0
        assert lo <= mean <= hi


# ---------------------------------------------------------------------------
# Public API: evaluation.gate
# ---------------------------------------------------------------------------


class TestEvaluationGateAPI:
    """Verify QualityGate passes/fails correctly."""

    def test_gate_passes_identical_models(self, policy, batch):
        gate = QualityGate(min_cosine_similarity=0.999)
        result = gate.check(policy, policy, batch)
        assert isinstance(result, QualityGateResult)
        assert result.passed
        assert result.cosine_similarity >= 0.999

    def test_gate_fails_on_random_model(self, policy, batch):
        random_policy = MinimalPolicy()
        gate = QualityGate(min_cosine_similarity=0.999)
        result = gate.check(policy, random_policy, batch)
        # Random model should fail the gate
        assert not result.passed or result.cosine_similarity < 0.999

    def test_gate_strict_threshold(self, policy, batch):
        gate = QualityGate(min_cosine_similarity=0.999999)
        result = gate.check(policy, policy, batch)
        assert result.passed


# ---------------------------------------------------------------------------
# Public API: tracking
# ---------------------------------------------------------------------------


class TestTrackingAPI:
    """Verify ExperimentTracker works in local mode."""

    def test_tracker_local_mode(self):
        from lerobot_edge.tracking.tracker import ExperimentTracker, TrackConfig

        config = TrackConfig(log_dir=tempfile.mkdtemp())
        tracker = ExperimentTracker(config=config, enabled=False)
        tracker.init_run("test_run")
        tracker.log_metrics({"loss": 0.5, "lr": 1e-4})
        tracker.log_config({"epochs": 10})
        tracker.finish_run()
        assert tracker.is_active is False


# ---------------------------------------------------------------------------
# End-to-end: quantize → benchmark → gate
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """Verify the full quantize → benchmark → quality gate pipeline."""

    def test_full_pipeline(self, policy, batch):
        measure_model_memory(policy)

        quantized = dynamic_int8_quantize(policy) if HAS_TORCHAO else policy

        measure_model_memory(quantized)

        gate = QualityGate(min_cosine_similarity=0.95)
        gate_result = gate.check(policy, quantized, batch)
        assert gate_result.passed

        report = compare_backends(policy, quantized, batch, num_samples=3)
        assert report.latency_original_ms > 0
        assert report.latency_quantized_ms > 0

        with tempfile.TemporaryDirectory() as tmpdir:
            report_dict = report.to_dict()
            report_path = Path(tmpdir) / "report.json"
            with open(report_path, "w") as f:
                json.dump(report_dict, f, indent=2)
            loaded = json.loads(report_path.read_text())
            assert loaded["variant"] == type(quantized).__name__
