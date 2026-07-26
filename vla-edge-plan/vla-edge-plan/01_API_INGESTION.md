# API & Plugin System Ingestion — Read Before Writing Any Code

**Purpose:** understand LeRobot well enough to build *against* it as a library
user and plugin author — not to modify its source. You are reading it the way
you'd read any dependency's docs before writing code against it.

## Step 1 — Install and orient

1. `pip install "lerobot>=0.6.0"` inside your new repo's environment.
2. Read the public docs (huggingface.co/docs/lerobot) and the GitHub README —
   you can clone `huggingface/lerobot` **read-only, locally, just to search its
   source** for the questions below. Do not commit to that clone, do not treat
   it as your project — it's reference material, like reading a library's
   source on your disk instead of only its docs site.
3. Also read, from that reference clone: `AGENTS.md`, `CLAUDE.md`,
   `docs/source/_toctree.yml` — these tell you what documentation pages exist,
   including the ones you need most: "Policy Interface and Base Classes",
   "Policy Factory and Plugin System", "Bring Your Own Hardware" /
   "bring_your_own_policies", "EnvHub", and "Compute Hardware Guide".

## Step 2 — Answer these specific questions before moving to 02

1. What is the exact interface a policy must implement to work with
   `lerobot-eval`/`lerobot-record`? (Confirmed importable as
   `lerobot.policies.pretrained.PreTrainedPolicy` — read its actual method
   signatures: `select_action`, `forward`, `reset`, device handling.)
2. **How does the plugin/registration system actually work?** Read the
   "Policy Factory and Plugin System" doc page and the real source of
   `lerobot.policies.factory.make_policy`. Is registration done via:
   entry_points in `pyproject.toml`, a decorator (`@register_policy(...)`),
   or an explicit function call at import time? This is the single most
   important answer in this whole document — it determines how
   `lerobot_edge` plugs in without touching LeRobot's source.
3. Where, concretely, does `lerobot-eval` load a checkpoint into a policy
   object, and does it accept an arbitrary registered `--policy.type` value
   from a third-party package, or only from LeRobot's own built-in registry?
   If the latter, find the extension point that makes third-party types
   discoverable (this is what the plugin system doc page should answer).
4. Confirm SmolVLA's actual documented memory/compute requirements from
   LeRobot's own "Compute Hardware Guide" page.
5. What sim benchmarks does `lerobot-eval` support out of the box (LIBERO,
   MetaWorld per the README) and which is lightest-weight for a laptop,
   minutes-not-hours dev loop.
6. Confirm the config system in use (likely `draccus` dataclasses) and how a
   third-party package would add new CLI-exposed config fields (e.g.
   `--policy.deploy_backend=...`) without editing LeRobot's own config
   classes — this usually means your own config dataclass that LeRobot's
   parser can still resolve, or a documented extension mechanism for exactly
   this.

## Output artifact

Write findings to `docs/agent-notes/api-map.md` in **your new repo**:

```markdown
# LeRobot API Map (as of lerobot==<version installed>)

## Policy interface (class, methods, signatures)
...

## Plugin/registration mechanism (exact API, with a working toy example)
...

## Eval/record checkpoint-loading path
...

## Config system and how third-party fields get exposed
...

## Sim benchmarks available for laptop-speed runs
...

## Open questions / risks
...
```

Pin the exact `lerobot` version you tested against in this file and in
`pyproject.toml` (`lerobot>=0.6.0,<0.7`) — third-party packages built on a
fast-moving library should always pin a tested range, not "latest."

If, after reading the plugin system docs, external registration turns out to
be more limited than expected (e.g. only a few extension points are actually
public), do not work around it by monkeypatching LeRobot's internals — treat
that as a scoping constraint for `02_TARGET_ARCHITECTURE.md` and design within
what the public API actually supports. A smaller, honestly-scoped integration
is better than an internals hack you'd have to explain away in an interview.
