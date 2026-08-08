"""Benchmarking, evaluation metrics, and reporting."""

from lerobot_edge.evaluation.benchmark import (
    BenchmarkResult,
    benchmark_backend,
    benchmark_policy_variants,
    compare_results,
    load_results,
)
from lerobot_edge.evaluation.gate import QualityGate, QualityGateResult
from lerobot_edge.evaluation.metrics import (
    OutputDivergence,
    QuantizationQualityReport,
    bootstrap_confidence_interval,
    compare_backends,
    measure_output_divergence,
)
from lerobot_edge.evaluation.experiment import (
    ExperimentResult,
    ExperimentRunner,
    run_experiment_grid,
)
from lerobot_edge.evaluation.report import (
    aggregate_results,
    generate_report,
    generate_results_table,
    plot_pareto_frontier,
)

__all__ = [
    "OutputDivergence",
    "QuantizationQualityReport",
    "bootstrap_confidence_interval",
    "compare_backends",
    "measure_output_divergence",
    "BenchmarkResult",
    "benchmark_backend",
    "benchmark_policy_variants",
    "compare_results",
    "load_results",
    "QualityGate",
    "QualityGateResult",
    "aggregate_results",
    "generate_report",
    "generate_results_table",
    "plot_pareto_frontier",
    "ExperimentRunner",
    "ExperimentResult",
    "run_experiment_grid",
]
