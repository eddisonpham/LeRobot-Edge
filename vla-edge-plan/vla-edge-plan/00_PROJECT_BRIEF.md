# Project Brief — Edge-Efficient Deployment Extension for LeRobot

> Working title: `lerobot-edge` (repo name) / `lerobot_edge` (package name).
> Rename to whatever you like — the planning docs use these as placeholders.

## 1. Starting point — no fork

**Do not fork `huggingface/lerobot`.** Instead:

- `pip install lerobot` (pin `>=0.6.0`) as a normal dependency inside your own,
  brand-new repository.
- Your repo starts empty. Every commit in it, from commit 1, is authored by you.
- You depend on and cite LeRobot; you do not copy or inherit its history.

**Why this matters:** a fork of a 26k-star, ~1500-commit monorepo will always
show that full inherited history and a "forked from huggingface/lerobot" label,
no matter how much you add on top — your work always reads as a fraction of the
total, regardless of its actual quality. A standalone package that *depends on*
LeRobot has no such baseline to be measured against: 100% of the repository is
yours, the dependency is honest and clearly declared, and this is exactly how
most real ML tooling is built (packages on top of `transformers`, `torch`,
`diffusers`, etc.) — it's a well-regarded pattern, not a workaround.

- Base model to build around: **SmolVLA (~450M params)**, `lerobot/smolvla_base`
  on the HF Hub — documented as trainable on one consumer GPU and runnable on a
  CPU/laptop.
- Upstream project you depend on (cite it, don't fork it):
  https://github.com/huggingface/lerobot (Apache-2.0)

## 2. The problem this project solves

LeRobot (as of v0.6.0, July 2026) ships state-of-the-art policies, VLA models,
world models, and reward models — but it is a **train-and-evaluate** library. It
has no story for what happens after you have a checkpoint: no quantization, no
ONNX/TensorRT export, no distillation, no latency/memory benchmarking, no
efficiency-vs-accuracy reporting.

This is not cosmetic. Every 2026 industry writeup on VLA deployment says the
same thing: getting a multi-hundred-million-to-billion-parameter policy to run
at useful control rates (5–30 Hz) on a power-and-latency-constrained board is
the actual bottleneck between a lab demo and a shipped robot. NVIDIA sells an
entire product line (Jetson + TensorRT Edge-LLM) to address exactly this. A
recent practitioner roadmap explicitly lists VLA quantization/edge deployment
as "unsettled as of mid-2026."

**This is the gap: a standalone package that takes any LeRobot policy
checkpoint and produces a compressed, benchmarked, deployment-ready variant —
installed as a plugin, without touching LeRobot's own source.**

## 3. What you're building (one sentence)

> Build `lerobot_edge`, a standalone, pip-installable extension package that
> registers with LeRobot's public policy plugin system to add quantization,
> ONNX export, teacher-student distillation, and an automated latency/memory-
> vs-task-success benchmark — so that a VLA trained in LeRobot can be evaluated
> not just on "does it work" but on "does it work fast enough, small enough,
> and cheap enough to actually ship."

## 4. Why this matters for the resume / interview

- Directly maps to real job families: NVIDIA Jetson/Isaac/TensorRT teams,
  Google on-device model teams, "robotics ML infra" / "embodied AI systems"
  roles.
- It is a **systems** contribution, not a "trained a model" contribution.
- Extending a major library through its public plugin API (rather than forking
  and patching internals) is itself a signal of engineering maturity —
  it shows you understood the library's architecture well enough to work with
  it, not just inside it.
- Produces a concrete, defensible artifact: a Pareto chart of
  latency/memory/cost vs. task success rate.

## 5. Non-goals

- Not attempting a new SOTA policy architecture.
- Not requiring a physical robot (sim benchmarks — PushT, LIBERO — are
  sufficient and are what LeRobot itself uses for CI-grade evaluation).
- Not requiring a multi-GPU training cluster. Distillation and any TensorRT
  step are the only parts that benefit from a cloud GPU.
- Not modifying LeRobot's own source at all, anywhere, for any reason. If a
  milestone seems to require that, stop and reconsider the design — see
  `03_EXTENSION_PLAN.md`.

## 6. Success metrics (fill in real numbers as you go — don't guess ahead)

Quantitative:
- Memory footprint reduction: FP32 baseline vs. INT8 vs. distilled student.
- Latency / throughput (Hz) on CPU (laptop) and, if pursued, cloud GPU / edge.
- Task success-rate retention on PushT and/or a LIBERO subset, per variant.
- A single Pareto plot: x = latency (or memory), y = success rate.

Qualitative:
- Code clean enough to plausibly review as a real package release.
- A README that reads like a small systems report.
- A repo where `git log` shows nothing but your own commits, because that's
  literally true.

## 7. Constraints

- **Primary development happens on a laptop, CPU-only** (or a modest consumer
  GPU if present). Every milestone needs a laptop-runnable path.
- **Cloud is opportunistic**, used only for: distillation training runs too
  slow on CPU, and the TensorRT export step (needs a real NVIDIA GPU driver
  stack most laptops don't have).
- Build against LeRobot's *public* API only: `lerobot.policies.factory`,
  `lerobot.policies.pretrained.PreTrainedPolicy`, the documented plugin/
  registration system, the `lerobot-eval` / `lerobot-train` CLIs. Never import
  from or depend on LeRobot's private/internal modules in a way that would
  break on a routine version bump.

## 8. Document map

| File | Purpose |
|---|---|
| `01_API_INGESTION.md` | How to study LeRobot's public API and plugin system before writing any code |
| `02_TARGET_ARCHITECTURE.md` | The `lerobot_edge` package: modules, interfaces, data flow |
| `03_EXTENSION_PLAN.md` | How to register with LeRobot's plugin system with zero changes to LeRobot itself |
| `04_INFRA_LOCAL_AND_CLOUD.md` | How the same code runs on a laptop and on cloud |
| `05_RESUME_AND_ATTRIBUTION.md` | Citing LeRobot correctly, and how to talk about this project |
| `subagents/M0..M8_*.md` | Sequential, scoped task files, one per milestone — execute in order |

**Note on naming:** these docs were revised from an earlier fork-based draft —
if you see any stray reference to "forking" or "the fork" anywhere below,
treat it as an error to fix; the current design is a standalone package
depending on `lerobot` via pip, per §1 above.

Read the docs once, in order, then execute `subagents/` in order.
