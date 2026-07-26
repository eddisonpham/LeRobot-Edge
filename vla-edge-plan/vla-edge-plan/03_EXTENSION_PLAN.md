# Extension Plan — Plugging Into LeRobot With Zero Changes To Its Source

Goal: make `02_TARGET_ARCHITECTURE.md` work using only LeRobot's *public*
API and plugin system, discovered in `01_API_INGESTION.md`. If any step below
seems to require editing a file inside the installed `lerobot` package, stop —
that's a signal the design needs rethinking, not a green light to patch a
site-packages file.

## 1. Registration mechanism

Whatever `01_API_INGESTION.md` Step 2.2 found (entry_points, decorator, or
explicit call), implement it in `lerobot_edge/__init__.py` so that simply
`import lerobot_edge` (or installing it, if LeRobot auto-discovers via
entry_points) makes every `CompressedPolicy` variant available to
`--policy.type=...` the same way a built-in LeRobot policy would be.

```python
# illustrative — match the real mechanism found during ingestion
# option A: entry_points-based auto-discovery (no explicit import needed)
#   -> declare in lerobot_edge's pyproject.toml under the group LeRobot scans for
# option B: explicit registration call
from lerobot.policies.factory import register_policy  # name TBD from real API
from lerobot_edge.base import CompressedPolicy

register_policy("edge_quant_int8", CompressedPolicy.for_backend("quant_int8"))
```

## 2. Config exposure

Add the `deploy_backend`-style config surface using whatever mechanism M0
found for third-party config extension. If LeRobot's config system doesn't
have a clean third-party extension point for *new* CLI flags (only for new
policy types), that's fine — you can also expose your own separate CLI
commands (e.g. `lerobot-edge-quantize`, `lerobot-edge-benchmark`) that operate
on a checkpoint path and produce a new checkpoint that plain `lerobot-eval
--policy.type=edge_quant_int8 --policy.path=<output of your command>` then
consumes. Prefer whichever path M0 shows is actually well-supported — don't
force the single-flag design if it fights the real API.

## 3. Eval/record scripts — literally zero changes

Because `CompressedPolicy` satisfies the same interface as a native LeRobot
policy and registers through the public plugin path, `lerobot-eval` and
`lerobot-record` — installed from a completely unmodified `pip install
lerobot` — should require no changes at all. If you find yourself needing to
patch anything in the installed `lerobot` package to make this work, the
`CompressedPolicy` wrapper is incomplete — fix the wrapper, not LeRobot.

## 4. Dependency hygiene

- Pin `lerobot>=0.6.0,<0.7` (or whatever range you actually test against) in
  `lerobot_edge`'s `pyproject.toml` — don't depend on "latest," since a
  fast-moving library can change its plugin API.
- New dependencies specific to this project (`onnxruntime`, `bitsandbytes`,
  `tensorrt`, etc.) belong in `lerobot_edge`'s own optional extras
  (`pip install "lerobot-edge[onnx]"`, `[tensorrt]`, etc.), not forced on
  every user.

## 5. Testing strategy

- Unit tests per module (`tests/test_quantize.py`, `tests/test_export_onnx.py`,
  etc.) in your own `tests/` directory, using LeRobot only as an installed
  dependency (import `lerobot`, don't need its source tree present).
- One integration/smoke test: install both packages fresh (simulating a real
  user), load `smolvla_base` from the Hub, quantize via `lerobot_edge`, export
  to ONNX, run 2 episodes of PushT through plain `lerobot-eval
  --policy.type=edge_...`, assert it completes. This test is your strongest
  proof that the "plugin, not fork" architecture actually works — run it in
  CI on every push.
- CI (GitHub Actions, in your own repo): `pip install lerobot lerobot-edge[dev]`
  from a clean environment on every run — this is what proves to a reviewer
  that your package really does bolt onto stock LeRobot with no hidden patch
  step.

## 6. If you ever want to upstream part of this

If a piece (e.g. the benchmark harness or the plugin registration helper)
seems genuinely useful to LeRobot itself, that's a separate, optional future
step: open a real PR against `huggingface/lerobot` following their
`CONTRIBUTING.md` and `AI_POLICY.md`. That would be a small, real, attributable
contribution to the upstream project — a nice complement to `lerobot_edge`,
but not required for `lerobot_edge` itself to be a complete, standalone
portfolio project.
