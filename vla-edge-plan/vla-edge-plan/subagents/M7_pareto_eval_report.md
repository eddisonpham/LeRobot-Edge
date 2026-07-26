# M7 — Efficiency/Accuracy Pareto Report

## Objective
Aggregate every variant benchmarked so far (whatever subset of M2–M6 was
completed) into a single, honest report: a Pareto plot and results table that
becomes the centerpiece of the README.

## Inputs
- Every `results_*.json` produced by M1–M6.

## Steps
1. Implement `lerobot_edge/report.py`: reads all results JSON files, produces:
   - A table: backend | device profile | latency (p50/p95) | memory | success
     rate | notes.
   - A Pareto plot: x-axis latency (or memory, pick whichever is more
     interesting given your actual results), y-axis success rate, one point
     per backend/device combination, FP32 baseline clearly marked.
2. If LeRobot's `Robometer` or `TOPReward` (reward models shipped in v0.6.0)
   are available in the installed `lerobot` package, use one of them as an
   automatic scorer to
   cross-check hand-counted success rates on a subset of episodes — this
   both saves laptop compute (fewer full manual rollouts needed) and adds a
   second, independent signal to the results.
3. Write up the results honestly in `docs/RESULTS.md` — including any
   variant that underperformed or failed to build. A tradeoff that didn't pay
   off is still a legitimate, discussable result; do not omit it.

## Acceptance criteria
- `docs/RESULTS.md` and an accompanying plot (checked into the repo, e.g. as
  a PNG or an interactive artifact) exist and reflect real numbers from real
  runs — no placeholder/estimated figures.
- The report clearly states which numbers came from laptop CPU vs. cloud GPU
  vs. (if applicable) edge hardware, per the reproducibility rule.

## Handoff to M8
The finished results table + plot — this is the content M8 uses to write the
final README.
