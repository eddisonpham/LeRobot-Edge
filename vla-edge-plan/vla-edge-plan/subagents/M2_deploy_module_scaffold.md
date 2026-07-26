# M2 — Scaffold the `lerobot_edge` Package and Prove the Plugin Hook

## Objective
Create the new package structure in your own repo and prove the plugin
registration mechanism works end-to-end, with a no-op backend — before any
real compression logic exists.

## Inputs
- `docs/agent-notes/api-map.md` (M0), including the working toy registration
  example.
- `../02_TARGET_ARCHITECTURE.md`, `../03_EXTENSION_PLAN.md`.

## Steps
1. Create `lerobot_edge/` in your repo with `__init__.py`, `base.py`, and stub
   files for `quantize.py`, `export_onnx.py`, `export_tensorrt.py`,
   `distill.py`, `benchmark.py`, `report.py` (empty/TODO bodies are fine).
2. Implement `base.py`: the `CompressedPolicy` wrapper and a trivial
   `identity` backend that just wraps a native policy unchanged — this proves
   the plumbing works before any real compression exists.
3. Implement the real registration mechanism found in M0 (`03_EXTENSION_PLAN.md`
   §1) so that `edge_identity` becomes a valid `--policy.type` value the
   moment `lerobot_edge` is installed alongside `lerobot` — with zero edits to
   the installed `lerobot` package.
4. Add `pyproject.toml` for `lerobot_edge` with `lerobot>=0.6.0,<0.7` as a
   dependency, and an empty/near-empty `[deploy]` extras group for later
   milestones' dependencies.
5. Re-run the M1 baseline eval command, this time with
   `--policy.type=edge_identity`, and confirm results match the M1 baseline
   (within normal sim nondeterminism).

## Acceptance criteria
- `lerobot-eval ... --policy.type=edge_identity` produces results
  statistically indistinguishable from the native `--policy.type=smolvla`
  baseline — proving the plugin adds no behavior change on its own.
- This works from a **fresh install**: `pip install lerobot lerobot-edge`
  (your package installed in editable/dev mode from your repo) in a clean
  virtualenv, no manual patching steps, no source checkout of `lerobot`
  itself required.
- A unit test exists for the registration mechanism.

## Handoff to M3
A working, tested no-op plugin — M3 onward just implements real backends
behind this same interface and registration path.
