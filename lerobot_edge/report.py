"""Pareto frontier report for lerobot_edge benchmark results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from lerobot_edge.benchmark import BenchmarkResult, load_results

logger = logging.getLogger(__name__)

__all__ = [
    "aggregate_results",
    "generate_results_table",
    "plot_pareto_frontier",
    "generate_report",
]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def aggregate_results(results_dir: str | Path) -> list[BenchmarkResult]:
    """Load and aggregate all benchmark results from a directory."""
    results_dir = Path(results_dir)
    all_results: list[BenchmarkResult] = []

    for results_file in results_dir.rglob("benchmark_results.json"):
        logger.info("Loading results from %s", results_file)
        results = load_results(results_file)
        all_results.extend(results)

    for json_file in results_dir.glob("*.json"):
        if json_file.name == "benchmark_results.json":
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            if isinstance(data, list) and data and "backend_name" in data[0]:
                results = [BenchmarkResult(**item) for item in data]
                all_results.extend(results)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    logger.info("Loaded %d total benchmark results", len(all_results))
    return all_results


def generate_results_table(results: list[BenchmarkResult]) -> str:
    """Generate a markdown table of benchmark results."""
    if not results:
        return "No benchmark results available."

    sorted_results = sorted(results, key=lambda r: r.latency_mean_ms)

    header = (
        "| Backend | Device | Latency (ms) | p50 (ms) | p95 (ms) | "
        "Memory (MB) | Params | Throughput (fps) | Success Rate |\n"
        "|---------|--------|--------------|----------|----------|"
        "------------|--------|------------------|--------------|\n"
    )

    rows = []
    for r in sorted_results:
        success_str = f"{r.success_rate:.1%}" if r.success_rate is not None else "N/A"
        params_str = f"{r.num_parameters:,}" if r.num_parameters > 0 else "N/A"
        rows.append(
            f"| {r.backend_name} | {r.device_profile} | "
            f"{r.latency_mean_ms:.2f} +/- {r.latency_std_ms:.2f} | "
            f"{r.latency_p50_ms:.2f} | {r.latency_p95_ms:.2f} | "
            f"{r.peak_memory_mb:.1f} | {params_str} | "
            f"{r.throughput_fps:.1f} | {success_str} |"
        )

    return header + "\n".join(rows)


def plot_pareto_frontier(
    results: list[BenchmarkResult],
    output_path: str | Path = "pareto_frontier.png",
    *,
    x_metric: str = "latency",
    title: str = "Edge Deployment Pareto Frontier",
) -> Path | None:
    """Plot Pareto frontier: latency/memory vs success rate."""
    if not HAS_MPL:
        logger.warning("matplotlib not available. Skipping Pareto plot.")
        return None

    plotted = [r for r in results if r.success_rate is not None]
    if not plotted:
        logger.warning("No results with success rates available for Pareto plot.")
        return None

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    backend_names = list(set(r.backend_name for r in plotted))
    colors = plt.cm.Set2(np.linspace(0, 1, len(backend_names)))
    color_map = dict(zip(backend_names, colors))

    for r in plotted:
        x = r.latency_mean_ms if x_metric == "latency" else r.peak_memory_mb
        color = color_map[r.backend_name]
        ax.scatter(x, r.success_rate, c=[color], s=100, alpha=0.8, edgecolors="black")
        ax.annotate(r.backend_name, (x, r.success_rate), textcoords="offset points", xytext=(5, 5), fontsize=8)

    if len(plotted) > 1:
        points = np.array([
            (r.latency_mean_ms if x_metric == "latency" else r.peak_memory_mb, r.success_rate)
            for r in plotted
        ])
        points = points[points[:, 0].argsort()]

        frontier = []
        max_y = -float("inf")
        for x, y in points:
            if y > max_y:
                frontier.append((x, y))
                max_y = y

        if frontier:
            frontier = np.array(frontier)
            ax.plot(frontier[:, 0], frontier[:, 1], "r--", alpha=0.5, label="Pareto Frontier")

    x_label = "Latency (ms)" if x_metric == "latency" else "Peak Memory (MB)"
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Task Success Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Pareto plot saved to %s", output_path)
    return output_path


def generate_report(
    results_dir: str | Path,
    output_dir: str | Path = "docs",
) -> dict[str, Any]:
    """Generate a complete benchmark report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = aggregate_results(results_dir)

    if not results:
        logger.warning("No benchmark results found in %s", results_dir)
        return {"error": "No results found"}

    table = generate_results_table(results)

    latency_plot = plot_pareto_frontier(
        results, output_path=output_dir / "pareto_latency.png", x_metric="latency", title="Latency vs Task Success Rate",
    )

    memory_plot = plot_pareto_frontier(
        results, output_path=output_dir / "pareto_memory.png", x_metric="memory", title="Memory vs Task Success Rate",
    )

    report_md = f"""# lerobot_edge Benchmark Report

Generated from {len(results)} benchmark results.

## Results Summary

{table}

## Pareto Frontiers

### Latency vs Success Rate
![Latency Pareto](pareto_latency.png)

### Memory vs Success Rate
![Memory Pareto](pareto_memory.png)

## Methodology

- **Warmup runs**: {results[0].warmup_runs if results else 'N/A'}
- **Benchmark runs**: {results[0].benchmark_runs if results else 'N/A'}
- **Git commit**: {results[0].git_commit if results else 'N/A'}
- **Timestamp**: {results[0].timestamp if results else 'N/A'}

## Notes

- All measurements taken on {results[0].device_profile if results else 'unknown device'}
- Success rates measured on PushT benchmark (unless otherwise noted)
- Memory includes peak GPU/CPU memory during inference
"""

    report_path = output_dir / "RESULTS.md"
    with open(report_path, "w") as f:
        f.write(report_md)

    logger.info("Report generated at %s", report_path)

    return {
        "num_results": len(results),
        "report_path": str(report_path),
        "latency_plot": str(latency_plot) if latency_plot else None,
        "memory_plot": str(memory_plot) if memory_plot else None,
    }


def main() -> None:
    """CLI entry point for ``lerobot-edge-report``."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark report from lerobot_edge results")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing benchmark results")
    parser.add_argument("--output-dir", type=str, default="docs", help="Output directory for the report")
    args = parser.parse_args()

    result = generate_report(args.results_dir, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
