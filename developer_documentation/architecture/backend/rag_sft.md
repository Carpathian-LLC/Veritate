# rag_sft

## What it is

Context-grounded (RAG) supervised fine-tune: continue-trains a saved checkpoint on the grounded corpus so the model copies facts out of its context window. Writes a new model dir and leaves the source untouched. Lives at [veritate_mri/training/rag_sft.py](../../../veritate_mri/training/rag_sft.py).

## How it works

- `load_source(name)` reads the latest checkpoint through `readers.checkpoints` and rebuilds the model with `veritate_core.load.load_from_state_dict(..., strict_canonical=False)`, so any variant reloads with its own args.
- Device comes from `veritate_core.plugin.hardware.pick_device()`; batches come from `native_trainer.make_loader`, the same seeded memmap loader the native trainer uses (one loader owner, rule 20).
- Corpus stem resolves through `save.resolve_corpus`, which covers both shared (`trainers/corpus/`) and bundled stems.
- Full-sequence LM loss, AdamW, gradient clip at `GRAD_CLIP`. Every `EVAL_EVERY` steps it appends a train row and, when a val bin exists, a val row. Every `CKPT_EVERY` steps and at the last step it saves.
- Save discipline (rule 21): checkpoints through `save.save()` (which runs the full dump suite), CSV rows through `save.append_train_row()`. The saved args carry `corpus`, `description`, and `grounded_sft_from`.
- Entry point: spawned as a subprocess by [rag_routes.py](routes.md) with `--source`, `--name`, `--corpus`, `--steps`. Stdout is `.rag_run.log`, tailed by `GET /rag/status`.

## Dependencies

- [save.py](save.md) for checkpoints and CSV rows.
- [native_trainer.py](native_trainer.md) for `make_loader`.
- [rag corpus builder](../../corpus/rag_corpus.md) for the bins.

## Pitfalls

- The sequence length is clamped to `min(--seq, model.seq)`; a source model with a short context silently trains on its own window, not the requested one.
- The source checkpoint's args are copied forward. Passing a `--name` that already exists appends to that model's CSV and checkpoint dir.
- No resume: every run starts at step 1 of the new model.
