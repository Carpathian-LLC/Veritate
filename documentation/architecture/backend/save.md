# save (checkpoint + CSV contract)

## What it is

The unified save module at [veritate_mri/training/save.py](../../../veritate_mri/training/save.py). Every trainer calls into it for two things: per-step CSV append and per-checkpoint full dump.

## The hooks / save guarantee

Every training path persists through `save.save()`. There is no training path that bypasses the dump suite. `save()` always writes, in one call:

1. The PyTorch checkpoint `checkpoints/step_<N>.pt`.
2. `config.json` (bootstrapped from `args` if absent).
3. The full dump suite into `hooks/step_<N>/`.

The per-step `train.csv` row is the separate `append_train_row()` call.

Callers that go through this single path:

- The shared plugin loop [trainers/common/vanilla_trainer.py:536](../../../trainers/common/vanilla_trainer.py#L536): `save.save(veritate_model, name, step, optimizer=opt, args=ckpt_args)`.
- The per-trainer loops `trainers/<id>/trainer.py` (e.g. [veritate_200m/trainer.py:497](../../../trainers/veritate_200m/trainer.py#L497)). Trainers reach `save` only through `veritate_core.plugin.save`, which re-exports this module ([veritate_core/plugin/__init__.py:24](../../../veritate_core/plugin/__init__.py#L24)).
- The grounded SFT [experiments/v2/rag/sft_grounded.py:66](../../../experiments/v2/rag/sft_grounded.py#L66): `vsave.save(model, name, step, optimizer=opt, args=ckpt_args)`.

Any new training entry point must call `save.save()` to keep this guarantee.

## The CSV contract

Strict header at [save.py:38](../../../veritate_mri/training/save.py#L38):

```
step,split,loss,lr,grad_norm,tok_per_s,wall_s,seed
```

`append_train_row(name, step, split, loss, lr, grad_norm, tok_per_s, wall_s, seed)` appends to `models/<name>/train.csv`. Formatting per field:

- `loss`, `grad_norm` — `f"{v:.6f}"`
- `lr` — `f"{v:.6e}"`
- `tok_per_s` — `f"{v:.2f}"`
- `wall_s` — `f"{v:.3f}"`
- `step`, `seed` — int
- `split` — string, "train" or "val"

This CSV is the source of truth for training curves. Every reader (`train_csv.py`), every dashboard chart (Training, Learning), and the heartbeat fallback detector consumes it.

## Checkpoint save

Signature ([save.py:386](../../../veritate_mri/training/save.py#L386)):

```
save(model, name, step, *, optimizer=None, args=None, prompt=None, dump_set=None)
```

- `args` supplies the description (ROE rule 6) and bootstraps `config.json` when it is absent.
- `prompt` overrides the canonical probe prompt; defaults to `PROBE_PROMPT`.
- `dump_set` is an optional iterable of dump names to skip; default runs all.

Returns the absolute path of the `.pt` written. It writes:

1. `models/<name>/config.json` (or bootstraps it from `args`).
2. `models/<name>/checkpoints/step_<N>.pt` via atomic rename (`.tmp` then `os.replace`).
3. The full dump suite into `models/<name>/hooks/step_<N>/`:
   - `probe.json`, `lens.npz`, `classroom.json`, `grades.json`, `math.json`, `grammar.json`, `reasoning.json`, `concepts.json`, `surprise.json`, `quant_kl.json`, `writing_health.json`, `reading_comprehension.json`, `generation.json`.

The dump files are produced by [checkpoint_probe.py](checkpoint_probe.md) and renamed per `RENAME_MAP_TEMPLATE` at [save.py:51](../../../veritate_mri/training/save.py#L51). Each dump runs under its own try/except, so one failed probe logs and continues without aborting the checkpoint. The `generation` dump is skipped (with a logged reason) when the model has no resolvable corpus stem.

## Dependencies

- [readers/paths.py](../../../veritate_mri/readers/paths.py) — for `model_dir`, `checkpoints_dir`, `train_csv` paths.
- [training/checkpoint_probe.py](../../../veritate_mri/training/checkpoint_probe.py) — generates the dump artifacts.
- [readers/models.py](../../../veritate_mri/readers/models.py) — validates model names before write.

## Pitfalls

- Trainers must not write `.pt` files or `train.csv` directly. Every write goes through this module so the format and the dump suite stay consistent.
- Adding a new CSV column breaks `train_csv.py` and every consumer. Coordinate the schema change across reader + every dashboard chart + the heartbeat fallback in the same commit.
- The dump suite runs synchronously at checkpoint time and can take seconds to minutes on large models. Don't add per-step calls to anything in the dump pipeline.
- `_validate_name` (via `models.is_valid_name`) gates every write. Name format: `<slug>_<size>` or legacy `<corpus>_<size>_<precision>_<version>`.
