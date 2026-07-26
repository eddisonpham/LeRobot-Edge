"""Comprehensive edge case tests for TensorRT backend (mocked, no GPU required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from lerobot_edge.export.tensorrt import HAS_TENSORRT, get_tensorrt_info


def _has_tensorrt() -> bool:
    return HAS_TENSORRT


class TestTensorRTInfoEdgeCases:
    def test_get_tensorrt_info_returns_dict(self):
        info = get_tensorrt_info()
        assert isinstance(info, dict)
        assert "available" in info

    def test_get_tensorrt_info_has_error_when_unavailable(self):
        if not HAS_TENSORRT:
            info = get_tensorrt_info()
            assert "error" in info
            assert isinstance(info["error"], str)

    def test_get_tensorrt_info_has_version_when_available(self):
        if HAS_TENSORRT:
            info = get_tensorrt_info()
            assert "version" in info


class TestTensorRTExportEdgeCases:
    def test_export_raises_import_error(self):
        from lerobot_edge.export.tensorrt import export_onnx_to_tensorrt
        if not HAS_TENSORRT:
            with pytest.raises(ImportError):
                export_onnx_to_tensorrt("dummy.onnx", "dummy.engine")

    def test_export_raises_with_empty_path(self):
        from lerobot_edge.export.tensorrt import export_onnx_to_tensorrt
        if not HAS_TENSORRT:
            with pytest.raises(ImportError):
                export_onnx_to_tensorrt("", "")

    def test_export_accepts_path_objects(self):
        from pathlib import Path
        from lerobot_edge.export.tensorrt import export_onnx_to_tensorrt
        if not HAS_TENSORRT:
            with pytest.raises(ImportError):
                export_onnx_to_tensorrt(Path("dummy.onnx"), Path("dummy.engine"))


class TestTensorRTBackendEdgeCases:
    @pytest.mark.skipif(not _has_tensorrt(), reason="TensorRT not installed")
    def test_backend_init_raises_without_pycuda(self):
        from lerobot_edge.export.tensorrt import TensorRTBackend
        with patch.dict("sys.modules", {"pycuda": None, "pycuda.driver": None}):
            with pytest.raises(ImportError, match="pycuda is required"):
                TensorRTBackend("dummy.engine")

    @pytest.mark.skipif(not _has_tensorrt(), reason="TensorRT not installed")
    def test_backend_init_raises_with_nonexistent_file(self):
        from lerobot_edge.export.tensorrt import TensorRTBackend
        with pytest.raises(Exception):
            TensorRTBackend("nonexistent/path/engine.engine")

    def test_has_tensorrt_flag_is_bool(self):
        assert isinstance(HAS_TENSORRT, bool)

    def test_tensorrt_is_optional_import(self):
        assert HAS_TENSORRT in (True, False)
