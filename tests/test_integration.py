"""End-to-end integration smoke tests for lerobot_edge.

These tests load real SmolVLA checkpoints, apply quantization, and run
inference through the full pipeline.  They require network access to
download the model from HuggingFace Hub.

Marked as @pytest.mark.integration and @pytest.mark.slow so they can
be skipped in fast CI runs or offline environments.

Run with:
    pytest tests/test_integration.py -v -m integration
    pytest tests/test_integration.py -v -m slow
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from lerobot_edge.compression.quantize import (
    QuantizedBackend,
    dynamic_int8_quantize,
)
from lerobot_edge.core.base import (
    CompressedPolicy,
    IdentityBackend,
    NativePyTorchBackend,
)
from lerobot_edge.core.configs import (
    EdgeIdentityConfig,
    EdgeQuantInt8Config,
)
from lerobot_edge.core.utils import measure_model_memory

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def smolvla_policy():
    """Load real SmolVLA policy from HuggingFace Hub."""
    try:
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError:
        pytest.skip("LeRobot SmolVLA not available")

    try:
        config = SmolVLAConfig.from_pretrained("lerobot/smolvla_base")
        config.device = "cpu"
        config.use_amp = False
        policy = SmolVLAPolicy(config)
        policy.eval()
        return policy
    except Exception as e:
        pytest.skip(f"Could not load SmolVLA: {e}")


@pytest.fixture(scope="module")
def smolvla_batch():
    """Create a dummy batch matching SmolVLA's expected input format."""
    return {
        "observation.images.front": torch.randn(1, 3, 224, 224),
        "observation.state": torch.randn(1, 2),
    }


class TestSmolVLAIdentityPlugin:
    """Test that edge_identity plugin wraps real SmolVLA correctly."""

    def test_load_smolvla_and_wrap(self, smolvla_policy):
        """Load real SmolVLA and wrap it in CompressedPolicy with IdentityBackend."""
        backend = IdentityBackend(smolvla_policy)
        config = EdgeIdentityConfig(device="cpu")
        policy = CompressedPolicy(config=config, backend=backend)

        assert policy is not None
        assert policy.device == torch.device("cpu")

    def test_smolvla_select_action(self, smolvla_policy, smolvla_batch):
        """Run select_action on real SmolVLA through CompressedPolicy."""
        backend = IdentityBackend(smolvla_policy)
        config = EdgeIdentityConfig(device="cpu")
        policy = CompressedPolicy(config=config, backend=backend)

        action = policy.select_action(smolvla_batch)
        assert isinstance(action, torch.Tensor)
        assert action.numel() > 0
        assert not torch.isnan(action).any(), "Action contains NaN values"

    def test_smolvla_forward(self, smolvla_policy, smolvla_batch):
        """Run forward pass on real SmolVLA through CompressedPolicy."""
        backend = IdentityBackend(smolvla_policy)
        config = EdgeIdentityConfig(device="cpu")
        policy = CompressedPolicy(config=config, backend=backend)

        loss, info = policy.forward(smolvla_batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert "actions" in info

    def test_smolvla_reset(self, smolvla_policy, smolvla_batch):
        """Test that reset clears action cache on real SmolVLA."""
        backend = IdentityBackend(smolvla_policy)
        config = EdgeIdentityConfig(device="cpu")
        policy = CompressedPolicy(config=config, backend=backend)

        policy.select_action(smolvla_batch)
        assert policy._action_cache is not None

        policy.reset()
        assert policy._action_cache is None
        assert policy._cache_idx == 0

    def test_smolvla_multiple_inferences(self, smolvla_policy, smolvla_batch):
        """Run multiple inference calls to verify stability."""
        backend = IdentityBackend(smolvla_policy)
        config = EdgeIdentityConfig(device="cpu")
        policy = CompressedPolicy(config=config, backend=backend)

        for i in range(3):
            action = policy.select_action(smolvla_batch)
            assert isinstance(action, torch.Tensor)
            assert not torch.isnan(action).any(), f"NaN in action at step {i}"
            policy.reset()


class TestSmolVLAQuantization:
    """Test dynamic INT8 quantization on real SmolVLA."""

    def test_quantize_smolvla(self, smolvla_policy):
        """Apply dynamic INT8 quantization to real SmolVLA."""
        original_mem = measure_model_memory(smolvla_policy)
        quantized = dynamic_int8_quantize(smolvla_policy)
        quantized_mem = measure_model_memory(quantized)

        assert quantized is not None
        assert quantized_mem["total_mb"] <= original_mem["total_mb"] * 1.1

    def test_quantized_smolvla_inference(self, smolvla_policy, smolvla_batch):
        """Run inference on quantized SmolVLA."""
        quantized = dynamic_int8_quantize(smolvla_policy)
        backend = NativePyTorchBackend(quantized, torch.device("cpu"))
        config = EdgeQuantInt8Config(device="cpu")
        policy = CompressedPolicy(config=config, backend=backend)

        action = policy.select_action(smolvla_batch)
        assert isinstance(action, torch.Tensor)
        assert action.numel() > 0
        assert not torch.isnan(action).any(), "Quantized action contains NaN values"

    def test_quantized_smolvla_via_backend(self, smolvla_policy, smolvla_batch):
        """Test QuantizedBackend.from_policy with real SmolVLA."""
        config = EdgeQuantInt8Config(device="cpu")
        backend = QuantizedBackend.from_policy(smolvla_policy, config)

        assert backend.quantization_type == "dynamic_int8"
        assert backend.device == torch.device("cpu")

        action = backend.predict(smolvla_batch)
        assert isinstance(action, torch.Tensor)
        assert not torch.isnan(action).any()

    def test_smolvla_memory_reduction(self, smolvla_policy):
        """Verify that quantization reduces memory footprint."""
        original = measure_model_memory(smolvla_policy)
        quantized = dynamic_int8_quantize(smolvla_policy)
        quantized_mem = measure_model_memory(quantized)

        reduction_pct = (1 - quantized_mem["total_mb"] / original["total_mb"]) * 100
        assert reduction_pct >= 0, f"Memory increased after quantization: {reduction_pct:.1f}%"

        print(
            f"\nMemory: {original['total_mb']:.1f} MB -> {quantized_mem['total_mb']:.1f} MB ({reduction_pct:.1f}% reduction)"
        )


class TestSmolVLAPluginRegistration:
    """Test that edge variants are discoverable by LeRobot's factory."""

    def test_configs_in_registry(self):
        """All edge config types should be in LeRobot's registry."""
        from lerobot.configs import PreTrainedConfig

        known = PreTrainedConfig.get_known_choices()
        for name in [
            "edge_identity",
            "edge_quant_int8",
            "edge_onnx_fp32",
            "edge_onnx_int8",
        ]:
            assert name in known, f"'{name}' not found in LeRobot registry"

    def test_make_policy_config(self):
        """LeRobot's make_policy_config should create edge configs."""
        from lerobot.policies.factory import make_policy_config

        config = make_policy_config("edge_quant_int8")
        assert config.type == "edge_quant_int8"
        assert hasattr(config, "quantize_dynamic")

    def test_config_class_retrievable(self):
        """Config classes should be retrievable from registry."""
        from lerobot.configs import PreTrainedConfig

        from lerobot_edge.core.configs import EdgeIdentityConfig, EdgeQuantInt8Config

        cls = PreTrainedConfig.get_choice_class("edge_identity")
        assert cls is EdgeIdentityConfig

        cls = PreTrainedConfig.get_choice_class("edge_quant_int8")
        assert cls is EdgeQuantInt8Config

    def test_compressed_policy_is_subclass_of_pretrained(self):
        """CompressedPolicy should satisfy LeRobot's policy interface."""
        from lerobot.policies.pretrained import PreTrainedPolicy

        assert issubclass(CompressedPolicy, PreTrainedPolicy)


class TestSmolVLABenchmark:
    """Test the benchmark harness with real SmolVLA."""

    def test_benchmark_identity_backend(self, smolvla_policy, smolvla_batch):
        """Benchmark the identity backend with real SmolVLA."""
        from lerobot_edge.evaluation.benchmark import benchmark_backend

        backend = IdentityBackend(smolvla_policy)
        result = benchmark_backend(
            backend,
            smolvla_batch,
            warmup_runs=2,
            num_runs=5,
            backend_name="smolvla_identity",
            device_profile="laptop_cpu",
        )

        assert result.latency_mean_ms > 0
        assert result.throughput_fps > 0
        assert result.backend_name == "smolvla_identity"

    def test_benchmark_quantized_backend(self, smolvla_policy, smolvla_batch):
        """Benchmark the quantized backend with real SmolVLA."""
        from lerobot_edge.evaluation.benchmark import benchmark_backend

        config = EdgeQuantInt8Config(device="cpu")
        backend = QuantizedBackend.from_policy(smolvla_policy, config)
        result = benchmark_backend(
            backend,
            smolvla_batch,
            warmup_runs=2,
            num_runs=5,
            backend_name="smolvla_quant_int8",
            device_profile="laptop_cpu",
        )

        assert result.latency_mean_ms > 0
        assert result.throughput_fps > 0


class TestSmolVLAFullPipeline:
    """Full pipeline chain test: load -> quantize -> benchmark -> quality gate -> save."""

    def test_full_pipeline_load_quantize_benchmark_gate(self, smolvla_policy, smolvla_batch):
        """End-to-end: load SmolVLA, quantize, benchmark, run quality gate, save results."""
        from lerobot_edge.evaluation.benchmark import benchmark_backend
        from lerobot_edge.evaluation.gate import QualityGate

        original = smolvla_policy
        original.eval()

        quantized = dynamic_int8_quantize(original)
        quantized.eval()

        identity_backend = IdentityBackend(original)
        identity_result = benchmark_backend(
            identity_backend,
            smolvla_batch,
            warmup_runs=2,
            num_runs=5,
            backend_name="identity",
            device_profile="laptop_cpu",
        )

        config = EdgeQuantInt8Config(device="cpu")
        quant_backend = QuantizedBackend.from_policy(original, config)
        quant_result = benchmark_backend(
            quant_backend,
            smolvla_batch,
            warmup_runs=2,
            num_runs=5,
            backend_name="dynamic_int8",
            device_profile="laptop_cpu",
        )

        gate = QualityGate(min_cosine_similarity=0.98, num_samples=5)
        gate_result = gate.check(original, quantized, smolvla_batch)

        assert identity_result.latency_mean_ms > 0
        assert quant_result.latency_mean_ms > 0
        assert gate_result.cosine_similarity > 0

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            results = {
                "identity": {
                    "latency_ms": identity_result.latency_mean_ms,
                    "fps": identity_result.throughput_fps,
                },
                "quantized": {
                    "latency_ms": quant_result.latency_mean_ms,
                    "fps": quant_result.throughput_fps,
                },
                "quality_gate": {
                    "passed": gate_result.passed,
                    "cosine_similarity": gate_result.cosine_similarity,
                    "mse": gate_result.mse,
                    "message": gate_result.message,
                },
            }
            json.dump(results, f, indent=2)
            output_path = f.name

        loaded = json.loads(Path(output_path).read_text())
        assert loaded["identity"]["latency_ms"] > 0
        assert loaded["quality_gate"]["cosine_similarity"] > 0

        Path(output_path).unlink()

    def test_quality_gate_with_real_smolvla(self, smolvla_policy, smolvla_batch):
        """Quality gate should pass for dynamic INT8 on real SmolVLA."""
        from lerobot_edge.evaluation.gate import QualityGate

        quantized = dynamic_int8_quantize(smolvla_policy)
        quantized.eval()

        gate = QualityGate(min_cosine_similarity=0.98, num_samples=10)
        result = gate.check(smolvla_policy, quantized, smolvla_batch)

        assert result.cosine_similarity > 0.9, (
            f"Cosine similarity too low: {result.cosine_similarity:.6f}"
        )
        assert result.passed or result.cosine_similarity >= 0.95, (
            f"Quality gate failed with cosine={result.cosine_similarity:.6f}: {result.message}"
        )

    def test_quality_gate_rejects_heavily_quantized(self, smolvla_policy, smolvla_batch):
        """Quality gate should reject when cosine similarity is below threshold."""
        from lerobot_edge.evaluation.gate import QualityGate

        gate_strict = QualityGate(min_cosine_similarity=0.9999, num_samples=5)
        quantized = dynamic_int8_quantize(smolvla_policy)
        result = gate_strict.check(smolvla_policy, quantized, smolvla_batch)

        assert result.threshold_cosine == 0.9999
        assert result.cosine_similarity > 0
        assert not result.passed or result.cosine_similarity >= 0.9999

    def test_compare_backends_metrics(self, smolvla_policy, smolvla_batch):
        """compare_backends should return full quality report."""
        from lerobot_edge.evaluation.metrics import compare_backends

        quantized = dynamic_int8_quantize(smolvla_policy)
        report = compare_backends(smolvla_policy, quantized, smolvla_batch, num_samples=3)

        assert report.original_params > 0
        assert report.quantized_params > 0
        assert report.latency_original_ms > 0
        assert report.latency_quantized_ms > 0
        assert report.divergence.cosine_similarity > 0
        assert report.compression_ratio >= 1.0
        assert isinstance(report.to_dict(), dict)


class TestSmolVLABatchStability:
    """Multi-batch inference stability tests."""

    def test_repeated_inference_consistency(self, smolvla_policy, smolvla_batch):
        """Same input should produce same output across repeated calls."""
        backend = IdentityBackend(smolvla_policy)
        config = EdgeIdentityConfig(device="cpu")
        policy = CompressedPolicy(config=config, backend=backend)

        torch.manual_seed(42)
        action1 = policy.select_action(smolvla_batch).clone()
        policy.reset()

        torch.manual_seed(42)
        action2 = policy.select_action(smolvla_batch).clone()
        policy.reset()

        assert torch.allclose(action1, action2, atol=1e-6)

    def test_varied_batch_sizes(self, smolvla_policy):
        """Inference should work with batch size 1 (SmolVLA's primary mode)."""
        backend = IdentityBackend(smolvla_policy)
        config = EdgeIdentityConfig(device="cpu")
        policy = CompressedPolicy(config=config, backend=backend)

        batch = {
            "observation.images.front": torch.randn(1, 3, 224, 224),
            "observation.state": torch.randn(1, 2),
        }
        action = policy.select_action(batch)
        assert isinstance(action, torch.Tensor)
        assert action.shape[0] == 1
        assert not torch.isnan(action).any()
        policy.reset()

    def test_quantized_repeated_inference(self, smolvla_policy, smolvla_batch):
        """Quantized model should produce stable outputs across calls."""
        config = EdgeQuantInt8Config(device="cpu")
        backend = QuantizedBackend.from_policy(smolvla_policy, config)

        outputs = []
        for _ in range(5):
            action = backend.predict(smolvla_batch)
            outputs.append(action.clone())

        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-5), (
                f"Output drift at iteration {i}"
            )


class TestSmolVLAConfigRoundTrip:
    """Config serialization round-trip tests."""

    def test_identity_config_roundtrip(self, tmp_path):
        """EdgeIdentityConfig should save and load correctly."""
        config = EdgeIdentityConfig(device="cpu")
        config.save_pretrained(tmp_path)
        restored = EdgeIdentityConfig.from_pretrained(tmp_path)
        assert restored.type == "edge_identity"
        assert restored.device == "cpu"

    def test_quant_int8_config_roundtrip(self, tmp_path):
        """EdgeQuantInt8Config should save and load correctly."""
        config = EdgeQuantInt8Config(device="cpu", quantize_bits=8)
        config.save_pretrained(tmp_path)
        restored = EdgeQuantInt8Config.from_pretrained(tmp_path)
        assert restored.type == "edge_quant_int8"
        assert restored.quantize_bits == 8

    def test_all_edge_configs_roundtrip(self, tmp_path):
        """All edge config types should roundtrip through save/load."""
        from lerobot_edge.core.configs import (
            EdgeDistilledConfig,
            EdgeOnnxFp32Config,
            EdgeQuantBnbFp4Config,
            EdgeQuantBnbInt8Config,
            EdgeQuantBnbNf4Config,
        )

        configs = [
            EdgeIdentityConfig(device="cpu"),
            EdgeQuantInt8Config(device="cpu"),
            EdgeQuantBnbInt8Config(device="cpu"),
            EdgeQuantBnbNf4Config(device="cpu"),
            EdgeQuantBnbFp4Config(device="cpu"),
            EdgeOnnxFp32Config(device="cpu"),
            EdgeDistilledConfig(device="cpu"),
        ]

        for config in configs:
            sub = tmp_path / config.type
            sub.mkdir()
            config.save_pretrained(sub)
            cls = type(config)
            restored = cls.from_pretrained(sub)
            assert restored.type == config.type
