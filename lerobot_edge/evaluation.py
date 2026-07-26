"""Backward-compatible re-export — actual code in lerobot_edge/evaluation/."""
from lerobot_edge.evaluation.metrics import *  # noqa: F401,F403
from lerobot_edge.evaluation.metrics import OutputDivergence, QuantizationQualityReport, measure_output_divergence, compare_backends, bootstrap_confidence_interval
