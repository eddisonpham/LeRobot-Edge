# Making It Legitimately Yours — Attribution and the Resume Story

## The short version

Because `lerobot_edge` is a standalone repo that depends on `lerobot` via pip
rather than a fork, the "is this really mine" question mostly answers itself:
every file and every commit in the repo is yours, because it is. What's left
is doing honest attribution of the dependency, and telling the story well.

## Attribution — what's actually required and what's just good practice

- **Required (Apache-2.0):** if you copy any actual code from LeRobot (rather
  than just calling its public API as a dependency, which is the whole point
  of this design), you must retain its copyright/license notice on that code.
  Aim to copy nothing — the plugin architecture in `03_EXTENSION_PLAN.md` is
  designed specifically so you don't need to.
- **Good practice, not strictly required, but do it anyway:** credit LeRobot
  clearly in your README, e.g.:
  > `lerobot_edge` is a policy-compression and edge-deployment plugin for
  > [🤗 LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0). It
  > installs alongside a stock `pip install lerobot` and adds no changes to
  > LeRobot itself — see `ARCHITECTURE.md` for how the plugin registers.
- Cite the LeRobot paper/repo the same way any dependency gets cited in a
  technical README (a line in an "Acknowledgments" or "Built On" section is
  enough — you don't need a legal disclaimer, just don't imply you built
  LeRobot itself).

## Resume bullet templates (fill in real numbers once you have them)

- "Built and released `lerobot_edge`, an open-source plugin package extending
  Hugging Face's LeRobot with policy quantization, ONNX export, and teacher-
  student distillation, reducing VLA inference latency by X% and memory
  footprint by Y% with Z% task-success retention on PushT/LIBERO benchmarks."
- "Designed a plugin architecture that registers custom compressed policies
  through LeRobot's public extension API, allowing `lerobot-eval` to
  benchmark quantized/distilled/ONNX-exported VLA variants with zero changes
  to the upstream library."
- "Built a reproducible efficiency-vs-accuracy benchmark harness for
  vision-language-action policies, producing Pareto-frontier reports across
  FP32/INT8/distilled variants on laptop CPU and cloud GPU targets."

Notice the second bullet is arguably the strongest one on this whole list —
"integrated with a major library's public extension API with zero upstream
changes" is a more specific, more verifiable, and more senior-sounding claim
than "modified an open-source repo," and it happens to also be exactly what
you did.

## Interview narrative — what actually gets asked

Be ready to explain, not recite:
- **Why a plugin package instead of a fork.** This is a great question to get
  asked, because the honest answer ("so the contribution is unambiguous and
  the integration survives LeRobot's own updates") is a genuinely good
  engineering answer, not a rationalization.
- The real latency/success-rate tradeoff you found, and why (know your own
  numbers cold — this is the heart of the project).
- What broke during quantization or ONNX export and how you debugged it —
  usually the most interesting part of the conversation.
- How the plugin registration actually works under the hood (entry_points vs.
  explicit registration, whichever `01_API_INGESTION.md` found) — this shows
  you understood LeRobot's architecture, not just its CLI.
