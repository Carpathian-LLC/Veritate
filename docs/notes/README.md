# Notes

User-driven brainstorms, vision docs, dialogue-captured explanations, and
reference material that informs **how** to develop but is not a strict
contract.

## What goes here

- **Explainers** for human readers (project narrative, "how we did X").
- **Vision docs** capturing the user's intent, future direction, design space.
- **Dialogue-folded ideas** — when a chat conversation produces a substantive
  idea worth keeping, it lands here.
- **Reference notes** that aren't engine/training contracts — historical
  context, "why this approach", recorded reasoning.

## What does NOT go here

- Hard contracts: `GLASS_MODEL_ROE.md`, `MODEL_NAMING.md`, protocol specs.
  Those live at `docs/` top level.
- Sprint plans: `PLAN.md`. Top level.
- Empirical results: `docs/results/`.
- Long-form architectural plans: `docs/plans/`.
- Static technical reference: `docs/reference/`.

## Rule of thumb

If a document says "you must" or "you must not", it's a contract — top level.
If it says "here's what we figured out" or "here's where we're going" or
"here's how to think about X", it's a note — `docs/notes/`.

## Index

- `HOW_WE_DID_IT.md` — explainer for the PyTorch ↔ C bridge, QAT pipeline,
  diff harness, lm_head fix. Written for sharing the project's story with
  external readers.
- `MOONSHOT_LOWPOWER.md` — vision doc for frontier-class reasoning on a
  low-power machine. Centers Mixture of Recursions as the multi-mind
  composer. Cites toy-scale evidence in `experiments/30_bitnet_mor_prototype/`.
