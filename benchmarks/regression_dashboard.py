"""Compare benchmark JSONs across runs and flag regressions > 5%."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# -- Data models --


@dataclass
class NormalizedResult:
    """Single benchmark result in normalized format."""

    source_file: str
    source_mtime: float
    model: str
    method: str
    compile_mode: str | None
    batch_size: int
    latency_mean_ms: float
    throughput_fps: float
    memory_mb: float
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Regression:
    """A detected regression between two runs."""

    model: str
    method: str
    compile_mode: str | None
    batch_size: int
    metric: str  # "latency", "throughput", "memory"
    previous_value: float
    current_value: float
    change_pct: float
    severity: str  # "regression", "improvement", "neutral"


@dataclass
class DashboardReport:
    timestamp: str
    files_scanned: int
    results_compared: int
    regressions: list[Regression]
    summary: dict[str, int]


# -- Normalization --


def _extract_compile_mode(method: str) -> tuple[str, str | None]:
    """Split 'int4+compile=reduce-overhead' into ('int4', 'reduce-overhead')."""
    if "+compile=" in method:
        parts = method.split("+compile=", 1)
        return parts[0], parts[1]
    return method, None


def normalize_results(
    filepath: Path,
) -> list[NormalizedResult]:
    """Load and normalize a benchmark JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    mtime = filepath.stat().st_mtime
    results: list[NormalizedResult] = []

    if isinstance(data, list):
        results.extend(_normalize_list_format(data, filepath.name, mtime))
    elif isinstance(data, dict):
        results.extend(_normalize_dict_format(data, filepath.name, mtime))

    return results


def _normalize_list_format(data: list[dict], fname: str, mtime: float) -> list[NormalizedResult]:
    """Normalize list-of-objects format."""
    out = []
    for item in data:
        method, cmode = _extract_compile_mode(
            item.get("quantization") or item.get("method", "unknown")
        )
        cmode = cmode or item.get("compile_mode")
        out.append(
            NormalizedResult(
                source_file=fname,
                source_mtime=mtime,
                model=item.get("config") or item.get("model", "unknown"),
                method=method,
                compile_mode=cmode,
                batch_size=item.get("batch_size", 1),
                latency_mean_ms=item.get("latency_mean_ms", 0),
                throughput_fps=item.get("throughput_fps", 0),
                memory_mb=item.get("model_memory_mb", 0),
                raw=item,
            )
        )
    return out


def _normalize_dict_format(data: dict, fname: str, mtime: float) -> list[NormalizedResult]:
    """Normalize dict formats (bench_large.py and compare_backends.py)."""
    out = []
    model = data.get("config") or data.get("model", fname)

    # Format A: bench_large.py — has model_params, fp32, fp16, etc.
    if "model_params" in data or "fp32_memory_mb" in data:
        for key in data:
            if key in ("fp32", "fp16", "int8", "nf4", "int4"):
                backend_data = data[key]
                if isinstance(backend_data, dict):
                    for bs_key, metrics in backend_data.items():
                        bs = int(bs_key)
                        out.append(
                            NormalizedResult(
                                source_file=fname,
                                source_mtime=mtime,
                                model=model,
                                method=key,
                                compile_mode=("reduce-overhead" if data.get("compiled") else None),
                                batch_size=bs,
                                latency_mean_ms=metrics.get("mean_ms", 0),
                                throughput_fps=metrics.get("throughput", 0),
                                memory_mb=data.get(f"{key}_memory_mb", 0),
                                raw=metrics,
                            )
                        )
        return out

    # Format B: compare_backends.py — identity, dynamic_int8, etc.
    for key, value in data.items():
        if isinstance(value, dict) and "latency_mean_ms" in value:
            out.append(
                NormalizedResult(
                    source_file=fname,
                    source_mtime=mtime,
                    model=model,
                    method=key,
                    compile_mode=("reduce-overhead" if "compiled" in key else None),
                    batch_size=1,  # compare_backends defaults to bs=1
                    latency_mean_ms=value.get("latency_mean_ms", 0),
                    throughput_fps=value.get("throughput_fps", 0),
                    memory_mb=value.get("memory", {}).get("total_mb", 0)
                    if isinstance(value.get("memory"), dict)
                    else 0,
                    raw=value,
                )
            )

    return out


# -- Diff engine --


def _make_key(r: NormalizedResult) -> tuple:
    """Create a grouping key for comparison."""
    return (
        r.model,
        r.method,
        r.compile_mode or "",
        r.batch_size,
    )


def compute_regressions(
    results: list[NormalizedResult],
    threshold_pct: float = 5.0,
) -> tuple[list[Regression], dict[str, int]]:
    """Compare latest two runs per group. Returns (regressions, summary)."""
    # Group by key
    from collections import defaultdict

    groups: dict[tuple, list[NormalizedResult]] = defaultdict(list)
    for r in results:
        groups[_make_key(r)].append(r)

    regressions: list[Regression] = []
    summary = {"regression": 0, "improvement": 0, "neutral": 0}

    for key, group in groups.items():
        if len(group) < 2:
            continue

        # Sort by timestamp (most recent last)
        group.sort(key=lambda r: r.source_mtime)

        prev = group[-2]
        curr = group[-1]

        model, method, cmode, bs = key
        cmode = cmode or None

        # Check latency
        if prev.latency_mean_ms > 0 and curr.latency_mean_ms > 0:
            pct = (curr.latency_mean_ms - prev.latency_mean_ms) / prev.latency_mean_ms * 100
            severity = _classify(pct, threshold_pct, higher_is_worse=True)
            regressions.append(
                Regression(
                    model=model,
                    method=method,
                    compile_mode=cmode,
                    batch_size=bs,
                    metric="latency",
                    previous_value=prev.latency_mean_ms,
                    current_value=curr.latency_mean_ms,
                    change_pct=pct,
                    severity=severity,
                )
            )
            summary[severity] += 1

        # Check throughput
        if prev.throughput_fps > 0 and curr.throughput_fps > 0:
            pct = (curr.throughput_fps - prev.throughput_fps) / prev.throughput_fps * 100
            severity = _classify(pct, threshold_pct, higher_is_worse=False)
            regressions.append(
                Regression(
                    model=model,
                    method=method,
                    compile_mode=cmode,
                    batch_size=bs,
                    metric="throughput",
                    previous_value=prev.throughput_fps,
                    current_value=curr.throughput_fps,
                    change_pct=pct,
                    severity=severity,
                )
            )
            summary[severity] += 1

        # Check memory
        if prev.memory_mb > 0 and curr.memory_mb > 0:
            pct = (curr.memory_mb - prev.memory_mb) / prev.memory_mb * 100
            severity = _classify(pct, threshold_pct, higher_is_worse=True)
            regressions.append(
                Regression(
                    model=model,
                    method=method,
                    compile_mode=cmode,
                    batch_size=bs,
                    metric="memory",
                    previous_value=prev.memory_mb,
                    current_value=curr.memory_mb,
                    change_pct=pct,
                    severity=severity,
                )
            )
            summary[severity] += 1

    return regressions, summary


def _classify(pct: float, threshold: float, higher_is_worse: bool) -> str:
    """Classify a percentage change."""
    abs_pct = abs(pct)
    if abs_pct <= threshold:
        return "neutral"

    if higher_is_worse:
        return "regression" if pct > 0 else "improvement"
    else:
        return "regression" if pct < 0 else "improvement"


# -- Printer --


# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _color(val: float, severity: str) -> str:
    """Color-code a value based on severity."""
    if severity == "regression":
        return f"{RED}{val:.2f}{RESET}"
    elif severity == "improvement":
        return f"{GREEN}{val:.2f}{RESET}"
    return f"{val:.2f}"


def print_dashboard(
    regressions: list[Regression],
    summary: dict[str, int],
    threshold_pct: float,
    files_scanned: int,
    results_compared: int,
) -> None:
    """Print color-coded regression dashboard."""
    print(f"\n{BOLD}{'=' * 110}{RESET}")
    print(
        f"{BOLD}  PERFORMANCE REGRESSION DASHBOARD  "
        f"|  Threshold: >{threshold_pct}%  "
        f"|  Files: {files_scanned}  "
        f"|  Comparisons: {results_compared}{RESET}"
    )
    print(f"{BOLD}{'=' * 110}{RESET}")

    if not regressions:
        print(f"\n  {GREEN}No regressions or improvements detected.{RESET}\n")
        return

    # Group by model
    models = sorted(set(r.model for r in regressions))
    for model in models:
        model_regs = [r for r in regressions if r.model == model]
        print(f"\n  {CYAN}{BOLD}Model: {model}{RESET}")
        print(
            f"  {'Method':<25} {'Compile':<18} {'bs':<5} "
            f"{'Metric':<12} {'Previous':<12} {'Current':<12} "
            f"{'Change':<10} {'Flag':<12}"
        )
        print(f"  {'-' * 100}")

        # Sort: regressions first, then by abs change
        model_regs.sort(
            key=lambda r: (
                0 if r.severity == "regression" else 1 if r.severity == "improvement" else 2,
                -abs(r.change_pct),
            )
        )

        for r in model_regs:
            cmode = r.compile_mode or "none"
            flag = ""
            if r.severity == "regression":
                flag = f"{RED}REGRESSION{RESET}"
            elif r.severity == "improvement":
                flag = f"{GREEN}improved{RESET}"

            unit = "ms" if r.metric == "latency" else "fps" if r.metric == "throughput" else "MB"
            prev_str = f"{r.previous_value:.2f} {unit}"
            curr_str = f"{r.current_value:.2f} {unit}"
            change_str = f"{r.change_pct:+.1f}%"

            print(
                f"  {r.method:<25} {cmode:<18} {r.batch_size:<5} "
                f"{r.metric:<12} {prev_str:<12} {curr_str:<12} "
                f"{_color_str(change_str, r.severity):<10} {flag:<12}"
            )

    # Summary
    print(f"\n{BOLD}{'=' * 110}{RESET}")
    print(
        f"  {BOLD}Summary:{RESET} "
        f"{RED}{summary['regression']} regressions{RESET}, "
        f"{GREEN}{summary['improvement']} improvements{RESET}, "
        f"{summary['neutral']} neutral"
    )
    print(f"{'=' * 110}\n")


def _color_str(s: str, severity: str) -> str:
    if severity == "regression":
        return f"{RED}{s}{RESET}"
    elif severity == "improvement":
        return f"{GREEN}{s}{RESET}"
    return s


# -- Runner --


def run_dashboard(
    dirs: list[str | Path] | None = None,
    threshold_pct: float = 5.0,
    output_path: str | Path | None = None,
) -> DashboardReport:
    """Scan dirs, compute regressions, print dashboard, optionally save JSON."""
    if dirs is None:
        dirs = ["benchmark_results"]

    import time as time_module

    # Collect all JSON files
    all_files: list[Path] = []
    for d in dirs:
        dpath = Path(d)
        if dpath.is_dir():
            all_files.extend(sorted(dpath.glob("*.json")))

    # Normalize
    all_results: list[NormalizedResult] = []
    for fp in all_files:
        try:
            all_results.extend(normalize_results(fp))
        except Exception as e:
            logger.debug("Skipping %s: %s", fp.name, e)

    # Compute regressions
    regressions, summary = compute_regressions(all_results, threshold_pct)

    report = DashboardReport(
        timestamp=time_module.strftime("%Y-%m-%dT%H:%M:%S"),
        files_scanned=len(all_files),
        results_compared=len(regressions),
        regressions=regressions,
        summary=summary,
    )

    # Print dashboard
    print_dashboard(
        regressions,
        summary,
        threshold_pct,
        len(all_files),
        len(regressions),
    )

    # Save report
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(
                {
                    "timestamp": report.timestamp,
                    "files_scanned": report.files_scanned,
                    "results_compared": report.results_compared,
                    "summary": report.summary,
                    "regressions": [
                        {
                            "model": r.model,
                            "method": r.method,
                            "compile_mode": r.compile_mode,
                            "batch_size": r.batch_size,
                            "metric": r.metric,
                            "previous": round(r.previous_value, 2),
                            "current": round(r.current_value, 2),
                            "change_pct": round(r.change_pct, 2),
                            "severity": r.severity,
                        }
                        for r in regressions
                    ],
                },
                f,
                indent=2,
            )
        logger.info("Report saved to %s", output)

    return report


# -- CLI --


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Performance regression dashboard for LeRobot Edge benchmarks"
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["benchmark_results"],
        help="Directories to scan (default: benchmark_results)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Regression threshold in percent (default: 5.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save JSON report to file",
    )
    args = parser.parse_args()

    run_dashboard(
        dirs=args.dirs,
        threshold_pct=args.threshold,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
