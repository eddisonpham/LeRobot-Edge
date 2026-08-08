"""Tests for lerobot_edge ONNX export module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from lerobot_edge.core.configs import EdgeOnnxFp32Config

# ---------------------------------------------------------------------------
# Helper functions (MUST be before test classes for pytest.mark.skipif)
# ---------------------------------------------------------------------------


def _has_onnx() -> bool:
    try:
        import onnx  # noqa: F401

        return True
    except ImportError:
        return False


def _has_ort() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class SimplePolicy(nn.Module):
    """A minimal policy for ONNX export testing."""

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
def simple_policy() -> SimplePolicy:
    return SimplePolicy()


@pytest.fixture
def onnx_config() -> EdgeOnnxFp32Config:
    return EdgeOnnxFp32Config(device="cpu")


# ---------------------------------------------------------------------------
# ONNX export tests (skipped if onnx not installed)
# ---------------------------------------------------------------------------


class TestOnnxExport:
    """Test ONNX export functionality."""

    @pytest.mark.skipif(not _has_onnx(), reason="onnx package not installed")
    def test_export_creates_file(self, simple_policy, onnx_config):
        """ONNX export should create a file."""
        from lerobot_edge.export.onnx import export_policy_to_onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"
            result = export_policy_to_onnx(
                simple_policy,
                onnx_config,
                output_path,
                input_names=["observation.state"],
                output_names=["actions"],
            )
            assert result.exists()
            assert result.stat().st_size > 0

    @pytest.mark.skipif(not _has_onnx(), reason="onnx package not installed")
    def test_export_validates_model(self, simple_policy, onnx_config):
        """Exported ONNX model should pass validation."""
        from lerobot_edge.export.onnx import export_policy_to_onnx, validate_onnx_model

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"
            export_policy_to_onnx(
                simple_policy,
                onnx_config,
                output_path,
                input_names=["observation.state"],
                output_names=["actions"],
            )

            info = validate_onnx_model(output_path)
            assert info["valid"] is True
            assert "inputs" in info
            assert "outputs" in info


class TestOnnxRuntimeBackend:
    """Test ONNX Runtime backend."""

    @pytest.mark.skipif(
        not (_has_onnx() and _has_ort()),
        reason="onnx or onnxruntime package not installed",
    )
    def test_backend_predict(self, simple_policy, onnx_config):
        """OnnxRuntimeBackend should produce valid predictions."""
        from lerobot_edge.export.onnx import OnnxRuntimeBackend, export_policy_to_onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"
            export_policy_to_onnx(
                simple_policy,
                onnx_config,
                output_path,
                input_names=["observation.state"],
                output_names=["actions"],
            )

            backend = OnnxRuntimeBackend(output_path)
            batch = {"observation.state": torch.randn(1, 7)}
            result = backend.predict(batch)

            assert isinstance(result, torch.Tensor)
            assert result.shape == (1, 2)

    @pytest.mark.skipif(
        not (_has_onnx() and _has_ort()),
        reason="onnx or onnxruntime package not installed",
    )
    def test_backend_reset(self, simple_policy, onnx_config):
        """OnnxRuntimeBackend reset should not raise."""
        from lerobot_edge.export.onnx import OnnxRuntimeBackend, export_policy_to_onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"
            export_policy_to_onnx(
                simple_policy,
                onnx_config,
                output_path,
                input_names=["observation.state"],
                output_names=["actions"],
            )

            backend = OnnxRuntimeBackend(output_path)
            backend.reset()  # Should not raise

    @pytest.mark.skipif(
        not (_has_onnx() and _has_ort()),
        reason="onnx or onnxruntime package not installed",
    )
    def test_backend_properties(self, simple_policy, onnx_config):
        """OnnxRuntimeBackend should report correct properties."""
        from lerobot_edge.export.onnx import OnnxRuntimeBackend, export_policy_to_onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"
            export_policy_to_onnx(
                simple_policy,
                onnx_config,
                output_path,
                input_names=["observation.state"],
                output_names=["actions"],
            )

            backend = OnnxRuntimeBackend(output_path)
            assert backend.device == torch.device("cpu")
            assert isinstance(backend.parameters, list)
            assert len(backend.input_names) > 0
            assert len(backend.output_names) > 0
