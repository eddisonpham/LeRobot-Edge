"""Tests for monitoring/logging module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lerobot_edge.monitoring import ExperimentTracker, TrackConfig


class TestTrackConfig:
    def test_defaults(self):
        config = TrackConfig()
        assert config.project == "lerobot-edge"
        assert config.entity is None
        assert config.tags == []
        assert config.log_model_artifacts is False

    def test_custom(self):
        config = TrackConfig(project="my-proj", tags=["v1"], entity="team")
        assert config.project == "my-proj"
        assert config.tags == ["v1"]
        assert config.entity == "team"


class TestExperimentTrackerLocal:
    """Tests using local JSON fallback (no wandb)."""

    def test_tracker_init_without_wandb(self):
        tracker = ExperimentTracker(enabled=False)
        assert not tracker.is_active

    def test_local_logging(self, tmp_path):
        config = TrackConfig(log_dir=str(tmp_path))
        tracker = ExperimentTracker(config=config, enabled=False)
        tracker.init_run(run_name="test-run")
        tracker.log_metrics({"loss": 0.5, "acc": 0.9})
        tracker.log_metrics({"loss": 0.3}, step=2)
        tracker.finish_run()

        log_files = list(tmp_path.glob("run_*.json"))
        assert len(log_files) == 1

        with open(log_files[0]) as f:
            data = json.load(f)

        assert len(data) >= 3
        assert data[0]["event"] == "init"
        assert data[1]["loss"] == 0.5

    def test_local_artifact_logging(self, tmp_path):
        config = TrackConfig(log_dir=str(tmp_path))
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"test": true}')
            tmp_artifact = f.name

        tracker = ExperimentTracker(config=config, enabled=False)
        tracker.init_run()
        tracker.log_artifact("model", "model", tmp_artifact, description="test")
        tracker.finish_run()

        log_files = list(tmp_path.glob("run_*.json"))
        with open(log_files[0]) as f:
            data = json.load(f)

        artifact_events = [e for e in data if e["event"] == "artifact"]
        assert len(artifact_events) == 1
        assert artifact_events[0]["name"] == "model"

    def test_local_benchmark_logging(self, tmp_path):
        config = TrackConfig(log_dir=str(tmp_path))
        tracker = ExperimentTracker(config=config, enabled=False)
        tracker.init_run()
        tracker.log_benchmark_result({
            "backend_name": "quant_int8",
            "latency_mean_ms": 15.2,
            "throughput_fps": 65.8,
            "peak_memory_mb": 256.0,
            "success_rate": 0.85,
        })
        tracker.finish_run()

        log_files = list(tmp_path.glob("run_*.json"))
        with open(log_files[0]) as f:
            data = json.load(f)

        metric_events = [e for e in data if e["event"] == "metrics"]
        assert len(metric_events) == 1
        assert metric_events[0]["bench/latency_mean_ms"] == 15.2
        assert metric_events[0]["eval/success_rate"] == 0.85

    def test_local_quality_report_logging(self, tmp_path):
        config = TrackConfig(log_dir=str(tmp_path))
        tracker = ExperimentTracker(config=config, enabled=False)
        tracker.init_run()
        tracker.log_quality_report({
            "variant": "int8",
            "compression_ratio": 2.1,
            "memory_savings_pct": 52.3,
            "cosine_similarity": 0.998,
        })
        tracker.finish_run()

        log_files = list(tmp_path.glob("run_*.json"))
        with open(log_files[0]) as f:
            data = json.load(f)

        metric_events = [e for e in data if e["event"] == "metrics"]
        assert metric_events[0]["quality/compression_ratio"] == 2.1

    def test_local_comparison_table(self, tmp_path):
        config = TrackConfig(log_dir=str(tmp_path))
        tracker = ExperimentTracker(config=config, enabled=False)
        tracker.init_run()
        tracker.log_comparison_table([
            {"variant": "fp32", "compression_ratio": 1.0},
            {"variant": "int8", "compression_ratio": 2.0},
        ])
        tracker.finish_run()

        log_files = list(tmp_path.glob("run_*.json"))
        with open(log_files[0]) as f:
            data = json.load(f)

        table_events = [e for e in data if e["event"] == "table"]
        assert len(table_events) == 1
        assert len(table_events[0]["data"]) == 2

    def test_is_active_without_wandb(self):
        tracker = ExperimentTracker(enabled=False)
        tracker.init_run()
        assert not tracker.is_active


class TestExperimentTrackerWandb:
    """Tests using patched wandb module."""

    def test_wandb_init_and_log(self, tmp_path):
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            import lerobot_edge.monitoring as mon_mod
            old_has_wandb = mon_mod.HAS_WANDB
            old_wandb_ref = getattr(mon_mod, "wandb", None)
            mon_mod.HAS_WANDB = True
            mon_mod.wandb = mock_wandb

            try:
                config = TrackConfig(log_dir=str(tmp_path))
                tracker = ExperimentTracker(config=config, enabled=True)
                tracker.init_run(run_name="test-wandb")
                mock_wandb.init.assert_called_once()

                tracker.log_metrics({"loss": 0.5})
                mock_wandb.log.assert_called_with({"loss": 0.5}, step=None)

                tracker.finish_run()
                mock_wandb.finish.assert_called_once()
                assert tracker._run is None
            finally:
                mon_mod.HAS_WANDB = old_has_wandb
                if old_wandb_ref is not None:
                    mon_mod.wandb = old_wandb_ref

    def test_wandb_artifact(self, tmp_path):
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            import lerobot_edge.monitoring as mon_mod
            old_has_wandb = mon_mod.HAS_WANDB
            old_wandb_ref = getattr(mon_mod, "wandb", None)
            mon_mod.HAS_WANDB = True
            mon_mod.wandb = mock_wandb

            try:
                config = TrackConfig(log_dir=str(tmp_path))
                tracker = ExperimentTracker(config=config, enabled=True)
                tracker.init_run()
                tracker.log_artifact("model", "model", "/tmp/model.pt")
                mock_run.log_artifact.assert_called_once()
                tracker.finish_run()
            finally:
                mon_mod.HAS_WANDB = old_has_wandb
                if old_wandb_ref is not None:
                    mon_mod.wandb = old_wandb_ref

    def test_wandb_is_active(self, tmp_path):
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            import lerobot_edge.monitoring as mon_mod
            old_has_wandb = mon_mod.HAS_WANDB
            old_wandb_ref = getattr(mon_mod, "wandb", None)
            mon_mod.HAS_WANDB = True
            mon_mod.wandb = mock_wandb

            try:
                config = TrackConfig(log_dir=str(tmp_path))
                tracker = ExperimentTracker(config=config, enabled=True)
                tracker.init_run()
                assert tracker.is_active
                tracker.finish_run()
                assert not tracker.is_active
            finally:
                mon_mod.HAS_WANDB = old_has_wandb
                if old_wandb_ref is not None:
                    mon_mod.wandb = old_wandb_ref

    def test_wandb_comparison_table(self, tmp_path):
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            import lerobot_edge.monitoring as mon_mod
            old_has_wandb = mon_mod.HAS_WANDB
            old_wandb_ref = getattr(mon_mod, "wandb", None)
            mon_mod.HAS_WANDB = True
            mon_mod.wandb = mock_wandb

            try:
                config = TrackConfig(log_dir=str(tmp_path))
                tracker = ExperimentTracker(config=config, enabled=True)
                tracker.init_run()
                tracker.log_comparison_table([{"variant": "fp32", "ratio": 1.0}])
                mock_run.log.assert_called_once()
                tracker.finish_run()
            finally:
                mon_mod.HAS_WANDB = old_has_wandb
                if old_wandb_ref is not None:
                    mon_mod.wandb = old_wandb_ref
