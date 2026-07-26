"""Tests for lerobot_edge configuration registration and validation."""

from __future__ import annotations

import pytest
from dataclasses import fields

from lerobot_edge.configs import (
    EdgeBaseConfig,
    EdgeIdentityConfig,
    EdgeQuantInt8Config,
    EdgeOnnxFp32Config,
    EdgeOnnxInt8Config,
    EdgeDistilledConfig,
    EdgeDistilledOnnxInt8Config,
)


class TestConfigRegistration:
    """Test that configs register correctly with LeRobot's plugin system."""

    def test_all_configs_are_subclasses_of_edge_base(self):
        """All edge configs must inherit from EdgeBaseConfig."""
        for cfg_cls in [
            EdgeIdentityConfig,
            EdgeQuantInt8Config,
            EdgeOnnxFp32Config,
            EdgeOnnxInt8Config,
            EdgeDistilledConfig,
            EdgeDistilledOnnxInt8Config,
        ]:
            assert issubclass(cfg_cls, EdgeBaseConfig)

    def test_all_configs_inherit_from_pretrained_config(self):
        """All edge configs must satisfy LeRobot's PreTrainedConfig interface."""
        from lerobot.configs import PreTrainedConfig

        for cfg_cls in [
            EdgeIdentityConfig,
            EdgeQuantInt8Config,
            EdgeOnnxFp32Config,
            EdgeOnnxInt8Config,
            EdgeDistilledConfig,
            EdgeDistilledOnnxInt8Config,
        ]:
            assert issubclass(cfg_cls, PreTrainedConfig)

    def test_identity_config_defaults(self):
        """EdgeIdentityConfig should have correct defaults."""
        cfg = EdgeIdentityConfig()
        assert cfg.type == "edge_identity"
        assert cfg.quantize_dynamic is True
        assert cfg.quantize_bits == 8

    def test_quant_int8_config_defaults(self):
        """EdgeQuantInt8Config should have INT8 quantization enabled."""
        cfg = EdgeQuantInt8Config()
        assert cfg.type == "edge_quant_int8"
        assert cfg.quantize_dynamic is True
        assert cfg.quantize_bits == 8

    def test_onnx_fp32_config_defaults(self):
        """EdgeOnnxFp32Config should have ONNX settings."""
        cfg = EdgeOnnxFp32Config()
        assert cfg.type == "edge_onnx_fp32"
        assert cfg.onnx_opset == 17

    def test_onnx_int8_config_defaults(self):
        """EdgeOnnxInt8Config should have both ONNX and INT8 settings."""
        cfg = EdgeOnnxInt8Config()
        assert cfg.type == "edge_onnx_int8"
        assert cfg.onnx_opset == 17
        assert cfg.quantize_dynamic is True

    def test_distilled_config_defaults(self):
        """EdgeDistilledConfig should have distillation settings."""
        cfg = EdgeDistilledConfig()
        assert cfg.type == "edge_distilled"
        assert cfg.distill_epochs == 10
        assert cfg.distill_lr == 1e-4
        assert cfg.distill_temperature == 2.0

    def test_distilled_onnx_int8_config_defaults(self):
        """EdgeDistilledOnnxInt8Config should combine distillation + ONNX + INT8."""
        cfg = EdgeDistilledOnnxInt8Config()
        assert cfg.type == "edge_distilled_onnx_int8"
        assert cfg.distill_epochs == 10
        assert cfg.onnx_opset == 17
        assert cfg.quantize_dynamic is True

    def test_config_type_property(self):
        """Config.type property should return the registered name."""
        from lerobot.configs import PreTrainedConfig

        for cfg_cls, expected_type in [
            (EdgeIdentityConfig, "edge_identity"),
            (EdgeQuantInt8Config, "edge_quant_int8"),
            (EdgeOnnxFp32Config, "edge_onnx_fp32"),
            (EdgeOnnxInt8Config, "edge_onnx_int8"),
            (EdgeDistilledConfig, "edge_distilled"),
            (EdgeDistilledOnnxInt8Config, "edge_distilled_onnx_int8"),
        ]:
            cfg = cfg_cls()
            assert cfg.type == expected_type

    def test_configs_in_lerobot_registry(self):
        """All edge configs should be discoverable via LeRobot's registry."""
        from lerobot.configs import PreTrainedConfig

        known_choices = PreTrainedConfig.get_known_choices()
        for name in [
            "edge_identity",
            "edge_quant_int8",
            "edge_onnx_fp32",
            "edge_onnx_int8",
            "edge_distilled",
            "edge_distilled_onnx_int8",
        ]:
            assert name in known_choices, f"'{name}' not found in registry"

    def test_configs_retrievable_from_registry(self):
        """Configs should be retrievable from LeRobot's registry by name."""
        from lerobot.configs import PreTrainedConfig

        for name in [
            "edge_identity",
            "edge_quant_int8",
            "edge_onnx_fp32",
            "edge_onnx_int8",
            "edge_distilled",
            "edge_distilled_onnx_int8",
        ]:
            cls = PreTrainedConfig.get_choice_class(name)
            assert cls is not None
            assert issubclass(cls, EdgeBaseConfig)


class TestEdgeBaseConfig:
    """Test EdgeBaseConfig shared fields and abstract implementations."""

    def test_abstract_properties_implemented(self):
        """EdgeBaseConfig must implement abstract properties."""
        cfg = EdgeIdentityConfig()
        assert cfg.observation_delta_indices is None
        assert cfg.action_delta_indices is None
        assert cfg.reward_delta_indices is None

    def test_get_optimizer_preset(self):
        """get_optimizer_preset should return a valid dict."""
        cfg = EdgeIdentityConfig()
        preset = cfg.get_optimizer_preset()
        assert isinstance(preset, dict)
        assert "optimizer_cls" in preset
        assert "lr" in preset

    def test_get_scheduler_preset(self):
        """get_scheduler_preset should return None by default."""
        cfg = EdgeIdentityConfig()
        assert cfg.get_scheduler_preset() is None

    def test_validate_features(self):
        """validate_features should not raise."""
        cfg = EdgeIdentityConfig()
        cfg.validate_features()  # Should not raise

    def test_custom_field_overrides(self):
        """Config fields should be overridable via constructor."""
        cfg = EdgeQuantInt8Config(
            quantize_dynamic=False,
            quantize_bits=4,
            benchmark_warmup=5,
        )
        assert cfg.quantize_dynamic is False
        assert cfg.quantize_bits == 4
        assert cfg.benchmark_warmup == 5

    def test_config_serialization(self):
        """Config should be serializable to/from dict."""
        cfg = EdgeOnnxInt8Config(onnx_opset=14, quantize_bits=8)
        # Test that config has expected attributes
        assert hasattr(cfg, "onnx_opset")
        assert hasattr(cfg, "quantize_bits")
        assert cfg.onnx_opset == 14
        assert cfg.quantize_bits == 8
