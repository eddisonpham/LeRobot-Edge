# M0 — Ingest LeRobot's Public API and Plugin System

## Objective
Build an accurate, written map of LeRobot's public API and plugin/registration
system before any code is written. No LeRobot source gets modified at any
point in this project — this milestone is about understanding it as a
dependency, not editing it.

## Inputs
- `../01_API_INGESTION.md` (the ingestion checklist).
- A fresh repo for your own project (`git init`, or create it on GitHub first
  and clone it) — this is where all your work and all your commits will live.

## Steps
1. `pip install "lerobot>=0.6.0"` inside your new repo's virtual environment.
2. Optionally clone `huggingface/lerobot` **read-only, elsewhere on disk**,
   purely to search its source for answers below — this clone is not your
   project and nothing gets committed from it.
3. Read, from the docs site and/or the read-only reference clone: `README.md`,
   `AGENTS.md`, `CLAUDE.md`, `docs/source/_toctree.yml`.
4. Read `lerobot.policies.pretrained.PreTrainedPolicy` — the exact method
   signatures a policy must implement.
5. Read `lerobot.policies.factory.make_policy` and the "Policy Factory and
   Plugin System" doc page — find the *exact* mechanism for registering a new,
   third-party policy type (entry_points, decorator, explicit call).
6. Confirm whether `lerobot-eval`/`lerobot-record` can discover a third-party-
   registered `--policy.type` value, and how.
7. Confirm SmolVLA's documented memory/compute requirements from LeRobot's own
   "Compute Hardware Guide".
8. Identify the lightest-weight sim benchmark(s) available through
   `lerobot-eval` for a laptop-speed dev loop.
9. Confirm the config system in use and how a third-party package exposes new
   CLI-visible config fields.

## Acceptance criteria
- `docs/agent-notes/api-map.md` exists **in your own new repo** and answers
  every question in `01_API_INGESTION.md` Step 2, with real class/function
  names and a working toy example of registering a dummy policy type from
  outside LeRobot's source tree.
- You can state, in one sentence each: (a) the policy interface, (b) the exact
  plugin registration mechanism, (c) the config extension mechanism.
- `pyproject.toml` in your new repo pins the tested `lerobot` version range.

## Handoff to M1
`docs/agent-notes/api-map.md`, plus a working toy registration example proving
the plugin mechanism actually works end-to-end before any real compression
logic is built on top of it.
