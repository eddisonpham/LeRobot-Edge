"""Tests for lerobot_edge benchmark harness."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from lerobot_edge.evaluation.benchmark import (
    BenchmarkResult,
    benchmark_backend,
    compare_results,
    load_results,
)
from lerobot_edge.core.utils import (
    get_git_commit_hash,
    measure_model_memory,
    measure_peak_memory_mb,
)
from lerobot_edge.core.base import NativePyTorchBackend, _PlaceholderBackend
from lerobot_edge.core.configs import EdgeIdentityConfig


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class SimplePolicy(nn.Module):
    """A minimal policy for benchmark testing."""

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
def dummy_input() -> dict[str, torch.Tensor]:
    return {"observation.state": torch.randn(1, 7)}


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestUtilities:
    """Test benchmark utility functions."""

    def test_get_git_commit_hash(self):
        """get_git_commit_hash should return a string."""
        commit = get_git_commit_hash()
        assert isinstance(commit, str)
        assert len(commit) > 0

    def test_measure_peak_memory_mb(self):
        """measure_peak_memory_mb should return a non-negative number."""
        mem = measure_peak_memory_mb()
        assert isinstance(mem, float)
        assert mem >= 0  # May be 0 on Windows where resource.getrusage is limited

    def test_measure_model_memory(self, simple_policy):
        """measure_model_memory should return expected keys."""
        mem = measure_model_memory(simple_policy)
        assert "param_mb" in mem
        assert "num_parameters" in mem
        assert mem["param_mb"] > 0
        assert mem["num_parameters"] > 0


# ---------------------------------------------------------------------------
# BenchmarkResult tests
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """BenchmarkResult should be creatable with required fields."""
        result = BenchmarkResult(
            backend_name="test",
            device_profile="cpu",
            git_commit="abc123",
            timestamp="2024-01-01T00:00:00",
        )
        assert result.backend_name == "test"
        assert result.device_profile == "cpu"

    def test_benchmark_result_defaults(self):
        """BenchmarkResult should have sensible defaults."""
        result = BenchmarkResult(
            backend_name="test",
            device_profile="cpu",
            git_commit="abc123",
            timestamp="2024-01-01T00:00:00",
        )
        assert result.latency_mean_ms == 0.0
        assert result.throughput_fps == 0.0
        assert result.peak_memory_mb == 0.0
        assert result.success_rate is None


# ---------------------------------------------------------------------------
# benchmark_backend tests
# ---------------------------------------------------------------------------


class TestBenchmarkBackend:
    """Test benchmark_backend function."""

    def test_benchmark_returns_result(self, simple_policy, dummy_input):
        """benchmark_backend should return a BenchmarkResult."""
        backend = NativePyTorchBackend(simple_policy)
        result = benchmark_backend(
            backend,
            dummy_input,
            warmup_runs=2,
            num_runs=5,
            backend_name="test_policy",
            device_profile="cpu",
        )

        assert isinstance(result, BenchmarkResult)
        assert result.backend_name == "test_policy"
        assert result.device_profile == "cpu"
        assert result.warmup_runs == 2
        assert result.benchmark_runs == 5

    def test_benchmark_latency_positive(self, simple_policy, dummy_input):
        """Benchmark should measure positive latency."""
        backend = NativePyTorchBackend(simple_policy)
        result = benchmark_backend(
            backend,
            dummy_input,
            warmup_runs=2,
            num_runs=5,
        )

        assert result.latency_mean_ms > 0
        assert result.latency_p50_ms > 0
        assert result.latency_p95_ms > 0

    def test_benchmark_throughput_positive(self, simple_policy, dummy_input):
        """Benchmark should measure positive throughput."""
        backend = NativePyTorchBackend(simple_policy)
        result = benchmark_backend(
            backend,
            dummy_input,
            warmup_runs=2,
            num_runs=5,
        )

        assert result.throughput_fps > 0

    def test_benchmark_placeholder_backend(self, dummy_input):
        """Benchmark should work with placeholder backend."""
        backend = _PlaceholderBackend("cpu")
        result = benchmark_backend(
            backend,
            dummy_input,
            warmup_runs=2,
            num_runs=5,
            backend_name="placeholder",
        )

        assert isinstance(result, BenchmarkResult)
        assert result.backend_name == "placeholder"


# ---------------------------------------------------------------------------
# compare_results tests
# ---------------------------------------------------------------------------


class TestCompareResults:
    """Test compare_results function."""

    def test_compare_results_with_baseline(self):
        """compare_results should compute ratios against baseline."""
        results = [
            BenchmarkResult(
                backend_name="edge_identity",
                device_profile="cpu",
                git_commit="abc",
                timestamp="2024-01-01",
                latency_mean_ms=10.0,
                peak_memory_mb=100.0,
                throughput_fps=100.0,
            ),
            BenchmarkResult(
                backend_name="edge_quant_int8",
                device_profile="cpu",
                git_commit="abc",
                timestamp="2024-01-01",
                latency_mean_ms=5.0,
                peak_memory_mb=50.0,
                throughput_fps=200.0,
            ),
        ]

        comparisons = compare_results(results, baseline_name="edge_identity")
        assert "edge_quant_int8" in comparisons
        assert comparisons["edge_quant_int8"]["speedup"] == 2.0
        assert comparisons["edge_quant_int8"]["memory_savings"] == 50.0

    def test_compare_results_no_baseline(self):
        """compare_results should handle missing baseline gracefully."""
        results = [
            BenchmarkResult(
                backend_name="other",
                device_profile="cpu",
                git_commit="abc",
                timestamp="2024-01-01",
                latency_mean_ms=10.0,
            ),
        ]

        comparisons = compare_results(results, baseline_name="edge_identity")
        assert comparisons == {}


# ---------------------------------------------------------------------------
# Result I/O tests
# ---------------------------------------------------------------------------


class TestResultIO:
    """Test result saving and loading."""

    def test_save_and_load_json(self):
        """Results should be saveable and loadable from JSON."""
        results = [
            BenchmarkResult(
                backend_name="test",
                device_profile="cpu",
                git_commit="abc",
                timestamp="2024-01-01",
                latency_mean_ms=10.0,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"

            # Save
            from lerobot_edge.evaluation.benchmark import _save_results_json
            _save_results_json(results, path)
            assert path.exists()

            # Load
            loaded = load_results(path)
            assert len(loaded) == 1
            assert loaded[0].backend_name == "test"
            assert loaded[0].latency_mean_ms == 10.0
