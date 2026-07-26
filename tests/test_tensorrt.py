"""Tests for lerobot_edge TensorRT export module (mocked, no GPU required)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lerobot_edge.export.tensorrt import HAS_TENSORRT, get_tensorrt_info


def _has_tensorrt() -> bool:
    return HAS_TENSORRT


class TestTensorRTAvailability:
    def test_get_tensorrt_info_without_trt(self):
        if HAS_TENSORRT:
            info = get_tensorrt_info()
            assert info["available"] is True
            assert "version" in info
        else:
            info = get_tensorrt_info()
            assert info["available"] is False
            assert "error" in info

    def test_has_tensorrt_flag(self):
        assert isinstance(HAS_TENSORRT, bool)


class TestTensorRTExportGuard:
    def test_export_raises_without_trt(self):
        from lerobot_edge.export.tensorrt import export_onnx_to_tensorrt

        if not HAS_TENSORRT:
            with pytest.raises(ImportError, match="TensorRT is required"):
                export_onnx_to_tensorrt("dummy.onnx", "dummy.engine")


@pytest.mark.skipif(not _has_tensorrt(), reason="TensorRT not installed")
class TestTensorRTBackendWithTrt:
    def test_backend_init_raises_without_pycuda(self):
        from lerobot_edge.export.tensorrt import TensorRTBackend

        with patch.dict("sys.modules", {"pycuda": None, "pycuda.driver": None}):
            with pytest.raises(ImportError, match="pycuda is required"):
                TensorRTBackend("dummy.engine")
