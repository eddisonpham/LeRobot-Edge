"""Tests verifying subpackages work independently."""

from __future__ import annotations

import pytest


class TestCoreSubpackage:
    def test_import_base(self):
        from lerobot_edge.core.base import (
            DeploymentBackend,
            NativePyTorchBackend,
            IdentityBackend,
            CompressedPolicy,
        )
        assert DeploymentBackend is not None
        assert NativePyTorchBackend is not None
        assert IdentityBackend is not None
        assert CompressedPolicy is not None

    def test_import_configs(self):
        from lerobot_edge.core.configs import (
            EdgeBaseConfig,
            EdgeIdentityConfig,
            EdgeQuantInt8Config,
        )
        assert EdgeBaseConfig is not None
        assert EdgeIdentityConfig.type == "edge_identity"
        assert EdgeQuantInt8Config.type == "edge_quant_int8"

    def test_import_router(self):
        from lerobot_edge.core.router import ConfidenceRouter
        assert ConfidenceRouter is not None

    def test_import_utils(self):
        from lerobot_edge.core.utils import (
            measure_model_memory,
            measure_peak_memory_mb,
            sigmoid_scalar,
        )
        assert callable(measure_model_memory)
        assert callable(measure_peak_memory_mb)
        assert callable(sigmoid_scalar)

    def test_sigmoid_scalar_works(self):
        from lerobot_edge.core.utils import sigmoid_scalar
        import math
        assert sigmoid_scalar(0.0) == 0.5
        assert sigmoid_scalar(100.0) == pytest.approx(1.0, abs=1e-10)
        assert sigmoid_scalar(-100.0) == pytest.approx(0.0, abs=1e-10)

    def test_measure_model_memory_works(self):
        import torch.nn as nn
        from lerobot_edge.core.utils import measure_model_memory
        model = nn.Linear(10, 2)
        result = measure_model_memory(model)
        assert result["num_parameters"] == 22
        assert result["total_mb"] > 0

    def test_core_init_exports(self):
        from lerobot_edge.core import (
            DeploymentBackend,
            CompressedPolicy,
            EdgeBaseConfig,
            ConfidenceRouter,
        )
        assert DeploymentBackend is not None


class TestCompressionSubpackage:
    def test_import_quantize(self):
        from lerobot_edge.compression.quantize import (
            dynamic_int8_quantize,
            static_int8_quantize,
            QuantizedBackend,
        )
        assert callable(dynamic_int8_quantize)
        assert callable(static_int8_quantize)

    def test_import_distill(self):
        from lerobot_edge.compression.distill import (
            DistillationLoss,
            DistilledBackend,
            distill,
        )
        assert DistillationLoss is not None
        assert callable(distill)

    def test_compression_init_exports(self):
        from lerobot_edge.compression import (
            dynamic_int8_quantize,
            QuantizedBackend,
            DistillationLoss,
        )
        assert callable(dynamic_int8_quantize)


class TestExportSubpackage:
    def test_import_onnx(self):
        from lerobot_edge.export.onnx import (
            export_policy_to_onnx,
            OnnxRuntimeBackend,
            validate_onnx_model,
        )
        assert callable(export_policy_to_onnx)

    def test_import_tensorrt(self):
        pytest.importorskip("tensorrt", reason="TensorRT not installed")
        from lerobot_edge.export.tensorrt import TensorRTBackend
        assert TensorRTBackend is not None

    def test_export_init_exports(self):
        from lerobot_edge.export import export_policy_to_onnx
        assert callable(export_policy_to_onnx)


class TestEvaluationSubpackage:
    def test_import_metrics(self):
        from lerobot_edge.evaluation.metrics import (
            OutputDivergence,
            QuantizationQualityReport,
            measure_output_divergence,
            compare_backends,
            bootstrap_confidence_interval,
        )
        assert OutputDivergence is not None
        assert callable(compare_backends)

    def test_import_benchmark(self):
        from lerobot_edge.evaluation.benchmark import (
            BenchmarkResult,
            benchmark_backend,
            compare_results,
            load_results,
        )
        assert BenchmarkResult is not None
        assert callable(benchmark_backend)

    def test_import_report(self):
        from lerobot_edge.evaluation.report import (
            aggregate_results,
            generate_report,
            generate_results_table,
            plot_pareto_frontier,
        )
        assert callable(aggregate_results)
        assert callable(generate_report)

    def test_evaluation_init_exports(self):
        from lerobot_edge.evaluation import (
            OutputDivergence,
            BenchmarkResult,
            compare_backends,
        )
        assert OutputDivergence is not None


class TestMonitoringSubpackage:
    def test_import_tracker(self):
        from lerobot_edge.tracking.tracker import (
            ExperimentTracker,
            TrackConfig,
        )
        assert ExperimentTracker is not None
        assert TrackConfig is not None

    def test_tracker_functional(self):
        from lerobot_edge.tracking import ExperimentTracker
        tracker = ExperimentTracker(enabled=False)
        assert not tracker.is_active
        tracker.init_run()
        assert not tracker.is_active

    def test_monitoring_init_exports(self):
        from lerobot_edge.tracking import ExperimentTracker, TrackConfig
        assert ExperimentTracker is not None


class TestBackwardCompatWrappers:
    def test_base_wrapper(self):
        from lerobot_edge.base import CompressedPolicy, DeploymentBackend
        assert CompressedPolicy is not None

    def test_configs_wrapper(self):
        from lerobot_edge.configs import EdgeQuantInt8Config
        assert EdgeQuantInt8Config.type == "edge_quant_int8"

    def test_quantize_wrapper(self):
        from lerobot_edge.quantize import dynamic_int8_quantize, QuantizedBackend
        assert callable(dynamic_int8_quantize)

    def test_benchmark_wrapper(self):
        from lerobot_edge.benchmark import BenchmarkResult, benchmark_backend
        assert BenchmarkResult is not None

    def test_evaluation_wrapper(self):
        from lerobot_edge.evaluation import compare_backends
        assert callable(compare_backends)

    def test_monitoring_wrapper(self):
        from lerobot_edge.monitoring import ExperimentTracker
        assert ExperimentTracker is not None

    def test_router_wrapper(self):
        from lerobot_edge.router import ConfidenceRouter
        assert ConfidenceRouter is not None
