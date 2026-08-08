"""Experiment tracking (W&B with local JSON fallback)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ExperimentTracker",
    "TrackConfig",
]

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


@dataclass
class TrackConfig:
    """Configuration for experiment tracking."""

    project: str = "lerobot-edge"
    entity: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    log_model_artifacts: bool = False
    log_dir: str = "wandb_logs"


class ExperimentTracker:
    """W&B tracking with local JSON fallback."""

    def __init__(self, config: TrackConfig | None = None, enabled: bool | None = None) -> None:
        self._config = config or TrackConfig()
        self._enabled = enabled if enabled is not None else HAS_WANDB
        self._run: Any = None
        self._local_log: list[dict[str, Any]] = []
        self._finished: bool = False

        if self._enabled:
            logger.info("W&B tracking enabled (project=%s)", self._config.project)
        else:
            logger.info("Local logging mode (wandb not installed or disabled)")

    def init_run(self, run_name: str | None = None, **kwargs: Any) -> None:
        """Start a run."""
        self._finished = False
        if self._enabled:
            self._run = wandb.init(
                project=self._config.project,
                entity=self._config.entity,
                name=run_name,
                tags=self._config.tags,
                notes=self._config.notes,
                config=kwargs,
            )
        else:
            self._local_log.append({"event": "init", "name": run_name, "config": kwargs})

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics."""
        if self._finished:
            logger.warning("log_metrics called after finish_run — data will be lost")
        if self._enabled and self._run is not None:
            wandb.log(metrics, step=step)
        else:
            entry = {"event": "metrics", "step": step, **metrics}
            self._local_log.append(entry)

    def log_config(self, config: dict[str, Any]) -> None:
        """Log config."""
        if self._finished:
            logger.warning("log_config called after finish_run — data will be lost")
        if self._enabled and self._run is not None:
            self._run.config.update(config)
        else:
            self._local_log.append({"event": "config", **config})

    def log_artifact(
        self,
        artifact_name: str,
        artifact_type: str,
        local_path: str | Path,
        description: str = "",
    ) -> None:
        """Log a file artifact."""
        if self._finished:
            logger.warning("log_artifact called after finish_run — data will be lost")
        if self._enabled and self._run is not None:
            artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
            artifact.add_file(str(local_path))
            if description:
                artifact.description = description
            self._run.log_artifact(artifact)
        else:
            self._local_log.append(
                {
                    "event": "artifact",
                    "name": artifact_name,
                    "type": artifact_type,
                    "path": str(local_path),
                }
            )

    def log_benchmark_result(self, result: dict[str, Any]) -> None:
        """Log benchmark result."""
        metrics = {}
        for key in [
            "latency_mean_ms",
            "latency_p50_ms",
            "latency_p95_ms",
            "throughput_fps",
            "peak_memory_mb",
            "param_memory_mb",
        ]:
            if key in result:
                metrics[f"bench/{key}"] = result[key]

        if "success_rate" in result and result["success_rate"] is not None:
            metrics["eval/success_rate"] = result["success_rate"]

        metrics["bench/backend"] = result.get("backend_name", "unknown")
        self.log_metrics(metrics)

    def log_quality_report(self, report: dict[str, Any]) -> None:
        """Log quality report."""
        metrics = {}
        for key in [
            "compression_ratio",
            "memory_savings_pct",
            "quality_degradation_pct",
            "cosine_similarity",
            "mse",
            "mae",
            "speedup_ratio",
        ]:
            if key in report:
                metrics[f"quality/{key}"] = report[key]

        metrics["quality/variant"] = report.get("variant", "unknown")
        self.log_metrics(metrics)

    def log_comparison_table(
        self,
        results: list[dict[str, Any]],
        columns: list[str] | None = None,
    ) -> None:
        """Log comparison table."""
        if self._finished:
            logger.warning("log_comparison_table called after finish_run — data will be lost")
        if not results:
            return

        if columns is None:
            columns = [
                "variant",
                "compression_ratio",
                "memory_savings_pct",
                "quality_degradation_pct",
                "cosine_similarity",
                "speedup_ratio",
            ]

        if self._enabled and self._run is not None and HAS_WANDB:
            table = wandb.Table(columns=columns)
            for r in results:
                row = [r.get(c, "") for c in columns]
                table.add_data(*row)
            self._run.log({"comparison_table": table})
        else:
            self._local_log.append({"event": "table", "columns": columns, "data": results})

    def finish_run(self) -> None:
        """Finish the run."""
        if self._enabled and self._run is not None:
            wandb.finish()
            self._run = None
        elif self._local_log:
            self._save_local_log()
        self._finished = True

    def _save_local_log(self) -> None:
        """Save local JSON log when wandb is unavailable."""
        import json
        import time

        log_dir = Path(self._config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"run_{timestamp}.json"

        with open(log_file, "w") as f:
            json.dump(self._local_log, f, indent=2, default=str)

        logger.info("Local run log saved to %s", log_file)
        self._local_log.clear()

    @property
    def is_active(self) -> bool:
        """Whether actively logging to W&B."""
        return self._enabled and self._run is not None
