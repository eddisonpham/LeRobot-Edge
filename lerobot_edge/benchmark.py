"""Backward-compatible re-export — actual code in lerobot_edge/evaluation/."""
from lerobot_edge.evaluation.benchmark import (  # noqa: F401
    BenchmarkResult,
    benchmark_backend,
    benchmark_policy_variants,
    compare_results,
    load_results,
)
from lerobot_edge.core.utils import (  # noqa: F401
    get_git_commit_hash,
    measure_model_memory,
    measure_peak_memory_mb,
)
from lerobot_edge.evaluation.benchmark import _save_results_json  # noqa: F401
