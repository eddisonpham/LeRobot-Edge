# M1 — Environment & Baseline Reproduction

## Objective
Get a real, unmodified LeRobot baseline running end-to-end on the laptop,
using plain `pip install lerobot` (no source install, no fork) plus your new
`lerobot_edge` repo scaffolded alongside it, before writing any new logic.
This establishes the number every later "compression" claim is measured
against.

## Inputs
- `docs/agent-notes/api-map.md` from M0.
- `../04_INFRA_LOCAL_AND_CLOUD.md`.

## Steps
1. In your new repo's environment: `pip install "lerobot>=0.6.0"` (plain PyPI
   install — you are a user of this library, not a maintainer of a fork of it).
2. Download `lerobot/smolvla_base` from the HF Hub.
3. Run `lerobot-eval` against SmolVLA on PushT (or the lightest sim benchmark
   from M0) for a small number of episodes (e.g. 10), using LeRobot exactly as
   installed. Record: success rate, wall-clock time per episode, peak memory.
4. Set up `configs/device/laptop_cpu.yaml` and `configs/device/cloud_gpu.yaml`
   in your own repo, as described in `04_INFRA_LOCAL_AND_CLOUD.md`.
5. Configure (but don't necessarily run yet) an HF Jobs cloud training config
   for later use in M5, following LeRobot's own documentation for it.
6. Write baseline numbers to `docs/agent-notes/baseline-results.json` in your
   repo — the reference point for every later Pareto point.

## Acceptance criteria
- A real `lerobot-eval` run completes locally against unmodified, pip-
  installed SmolVLA and produces trustworthy success-rate and timing numbers.
- `docs/agent-notes/baseline-results.json` exists with device profile, git
  commit hash (of *your* repo), config, and results recorded.
- This step required installing exactly one package (`lerobot`) from PyPI —
  no source checkout, no build step, no patched files. If it required more
  than that, something upstream of this milestone went wrong.

## Handoff to M2
Baseline results file, plus confirmation the vanilla, pip-installed pipeline
runs cleanly on the laptop — the safety net M2 onward must not break.
