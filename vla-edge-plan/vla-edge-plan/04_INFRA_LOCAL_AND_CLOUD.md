# Infra — Laptop-First, Cloud-Optional

## Guiding rule

Every milestone must define a **laptop path** that actually runs, even if the
numbers are worse than what cloud compute would give you. Cloud is for the two
steps that genuinely need it: (1) any training run too slow on CPU (distillation,
teacher fine-tuning), and (2) the TensorRT export step, which needs an NVIDIA GPU
driver stack most laptops don't have.

## Device profiles

Add two small config files (exact format depends on what M0 finds — draccus
dataclasses or YAML):

```
configs/device/laptop_cpu.yaml     # torch device=cpu, small batch/episode counts
configs/device/cloud_gpu.yaml      # torch device=cuda, full batch/episode counts
```

Same code path, different config — this is the point. A reviewer should be able
to run the identical command on a laptop and on a cloud box and only change
`--device_profile=laptop_cpu` → `--device_profile=cloud_gpu`.

## Local (laptop) — what actually runs here

- SmolVLA (~450M params) — documented as CPU-runnable, this is your baseline.
- PushT — lightweight 2D physics sim, no GPU required, fast episodes. This is
  your primary dev-loop benchmark; treat it like a unit test suite you run
  constantly.
- ONNX Runtime with the CPU execution provider for the exported/quantized
  policy.
- Quantization (dynamic INT8 at minimum — this needs no GPU).
- A small LIBERO task subset for periodic (not every-commit) sanity checks,
  since LIBERO episodes are heavier than PushT.

## Cloud — what needs it, and what to use

LeRobot v0.6.0 added **native HF Jobs cloud training** — use this first before
reaching for anything else, since it's already wired into the library you're
extending and costs you no new integration work:
- Teacher fine-tuning (if you fine-tune a larger VLA as the distillation
  teacher rather than using an off-the-shelf checkpoint).
- The distillation training loop itself, if laptop CPU speed makes it
  impractically slow (verify with a short run before assuming this — a
  450M-param student on a small dataset may be more tractable on CPU than it
  looks).

If HF Jobs doesn't fit your needs (e.g. you want a specific GPU generation for
the TensorRT step, which needs a real NVIDIA driver stack):
- A single on-demand GPU instance (Lambda, RunPod, or a cloud provider's spot
  GPU instance) for a few hours is enough — you do not need a cluster.
- SkyPilot is worth knowing about if you want a cloud-agnostic launch config
  instead of hand-writing instance setup per-provider, but it's optional
  infrastructure, not a requirement — don't let learning a new orchestration
  tool become the bottleneck on the actual ML/systems work.

## Docker

Your repo is standalone, so write your own Dockerfiles rather than modifying
anything of LeRobot's:
- `docker/dev.Dockerfile`: a plain base image (e.g. `python:3.11-slim` or a
  CUDA base if you want GPU support) that runs `pip install lerobot
  lerobot-edge` — this is your proof, runnable by anyone, that the whole thing
  installs cleanly on top of stock LeRobot with no source patching.
- `docker/edge.Dockerfile`: adds `onnxruntime` (and, in a separate
  cloud/edge-only variant, `tensorrt`, which needs an NVIDIA base image and
  isn't installable on a laptop without an NVIDIA GPU anyway).
- Both Dockerfiles should work by pulling `lerobot` from PyPI — never `COPY`
  or vendor LeRobot's source into your image.

## Reproducibility rule (non-negotiable)

Every benchmark run's output JSON must record:
- device profile used
- git commit hash of your repo at the time of the run
- full resolved config
- wall-clock timestamp

This is what turns "I got a chart" into "here's a reproducible result" — the
difference matters a lot in how the project reads to a technical reviewer.
