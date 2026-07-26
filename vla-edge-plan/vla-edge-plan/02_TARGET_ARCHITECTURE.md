# Target Architecture — `lerobot_edge`

## Design principle

**A standalone package, zero changes to LeRobot's own source.** Everything below
lives in your own repo, in a `lerobot_edge/` package that depends on `lerobot`
via pip. Any compressed/exported policy must subclass (or wrap and satisfy the
same interface as) `lerobot.policies.pretrained.PreTrainedPolicy` (confirmed in
`01_API_INGESTION.md`, Step 2.1) and self-register through LeRobot's public
plugin mechanism (Step 2.2), so `lerobot-eval` and `lerobot-record` — installed
from plain, unmodified `pip install lerobot` — pick it up automatically once
`lerobot_edge` is also installed.

## Module layout (your own repo)

```
lerobot-edge/                    # repo root
├── pyproject.toml               # depends on lerobot>=0.6.0,<0.7
├── lerobot_edge/
│   ├── __init__.py               # registers CompressedPolicy variants on import
│   ├── base.py                   # DeploymentBackend interface + CompressedPolicy wrapper
│   ├── quantize.py               # Post-training quantization (dynamic INT8, static INT8, 4-bit)
│   ├── export_onnx.py            # Export policy backbone -> ONNX, ONNX Runtime session wrapper
│   ├── export_tensorrt.py        # ONNX -> TensorRT engine build (cloud/edge GPU only, guarded)
│   ├── distill.py                # Teacher -> student distillation training loop
│   ├── benchmark.py              # Latency / memory / throughput harness, CSV+JSON output
│   ├── router.py                 # (stretch goal) simple edge/cloud confidence-based router
│   └── report.py                 # Pareto frontier plotting (latency|memory vs. success rate)
├── tests/
├── docs/agent-notes/
└── README.md
```

## Core interface (sketch — adapt names to match what M0 found in the real repo)

```python
# base.py
class CompressedPolicy:
    """
    Wraps any deployment backend (quantized weights, ONNX Runtime session,
    TensorRT engine, distilled student model) behind the exact call signature
    the existing eval/record scripts expect from a native LeRobot policy.
    """
    def __init__(self, backend, config): ...
    def select_action(self, observation): ...  # match the real interface found in M0
    def reset(self): ...
    @property
    def device(self): ...
```

Everything else in the package produces objects that get wrapped in
`CompressedPolicy` before being registered and handed to the eval loop. This is
the single most important design decision in the project — it's what lets you
say "I built a plugin that LeRobot's own `lerobot-eval` runs unmodified"
rather than "I built a parallel eval script," which is a meaningfully stronger
claim, and it's also what makes the "zero fork" approach actually work end to
end rather than just avoiding the fork cosmetically.

## Pipeline (conceptual data flow)

```
checkpoint (FP32, e.g. lerobot/smolvla_base)
        │
        ├─► quantize.py ──► INT8 / 4-bit weights ─┐
        │                                          │
        ├─► distill.py  ──► smaller student ───────┤─► export_onnx.py ──► ONNX Runtime session
        │   (teacher: FP32 or a larger VLA)         │        │
        │                                          │        └─► export_tensorrt.py (cloud/edge only)
        └──────────────────────────────────────────┘
                     │
                     ▼
         benchmark.py (latency, memory, throughput; N=laptop CPU, cloud GPU, edge if available)
                     │
                     ▼
      lerobot-eval on PushT / LIBERO subset, once per variant
      (reuse Robometer/TOPReward if available from the repo, to
       auto-score rollouts instead of hand-labeling success)
                     │
                     ▼
           report.py -> Pareto plot + results table -> README
```

## Config additions

Follow whatever extension mechanism M0 found for exposing new CLI-visible
config fields from a third-party package (likely a `draccus`-compatible config
class of your own that LeRobot's parser can resolve once `lerobot_edge` is
installed and imported). The goal is the same CLI surface as if this were
built into LeRobot itself, e.g.:

```
lerobot-eval \
  --policy.path=lerobot/smolvla_base \
  --policy.deploy_backend=onnx_int8 \
  --env.type=pusht \
  --eval.n_episodes=10
```

`deploy_backend` values to support, roughly in build order:
`none` (baseline) → `quant_int8` → `onnx_fp32` → `onnx_int8` → `distilled` →
`distilled_onnx_int8` → (`tensorrt_*`, cloud/edge only).

## Stretch goal: edge-cloud collaborative routing

Current research (2026) is exploring routing: run the cheap/fast path on-device
by default, and fall back to a larger cloud model only when the on-device
policy's confidence is low. `router.py` can be a simple version of this:
threshold on the flow-matching / action-head uncertainty, log how often it would
have escalated. This is optional (Milestone M6/M7 territory) — don't attempt it
before the core pipeline (quantize → export → benchmark → eval) works
end-to-end.

## What NOT to build

- Do not reimplement SmolVLA or any policy architecture — reuse the pretrained
  checkpoints on the HF Hub.
- Do not build a new simulation environment — reuse PushT/LIBERO already wired
  into `lerobot-eval`.
- Do not build a custom training loop for the baseline policy — only `distill.py`
  needs new training code, and it should reuse as much of the existing
  `lerobot-train` infrastructure (data loading, logging, checkpointing) as
  possible.
