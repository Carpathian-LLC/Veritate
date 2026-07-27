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

- `loss`, `grad_norm`: `f"{v:.6f}"`
- `lr`: `f"{v:.6e}"`
- `tok_per_s`: `f"{v:.2f}"`
- `wall_s`: `f"{v:.3f}"`
- `step`, `seed`: int
- `split`: string, "train" or "val"

This CSV is the source of truth for training curves. Every reader (`train_csv.py`), every dashboard chart (Training, Learning), and the heartbeat fallback detector consumes it.

## Checkpoint save

Signature ([save.py:386](../../../veritate_mri/training/save.py#L386)):

```
save(model, name, step, *, optimizer=None, args=None, prompt=None, dump_set=None)
```

- `args` supplies the description (ROE rule 6) and bootstraps `config.json` when it is absent.
- `prompt` overrides the probe/generation seed. When omitted, `dump_probe` runs over a deterministically sampled seed collection (`sample_probe_prompts()`); single-prompt dumps (`surprise`, `quant_kl`, `generation`) use `PROBE_PROMPT`. An explicit `prompt` pins `dump_probe` to that single seed too.
- `dump_set` is an optional iterable of dump names to skip; default runs all.

Returns the absolute path of the `.pt` written. It writes:

1. `models/<name>/config.json` (or bootstraps it from `args`).
2. `models/<name>/checkpoints/step_<N>.pt` via atomic rename (`.tmp` then `os.replace`).
3. The full dump suite into `models/<name>/hooks/step_<N>/`:
   - `probe.json`, `lens.npz`, `classroom.json`, `grades.json`, `math.json`, `grammar.json`, `reasoning.json`, `concepts.json`, `surprise.json`, `quant_kl.json`, `writing_health.json`, `reading_comprehension.json`, `generation.json`.

The dump files are produced by [checkpoint_probe.py](checkpoint_probe.md) and renamed per `RENAME_MAP_TEMPLATE` at [save.py:51](../../../veritate_mri/training/save.py#L51). Each dump runs under its own try/except, so one failed probe logs and continues without aborting the checkpoint. The `generation` dump is skipped (with a logged reason) when the model has no resolvable corpus stem.

### Corpus-stem resolution

The `generation` and `writing_health` dumps need a prepped corpus bin. `save()` resolves it from `training_args.corpus` ([save.py:504](../../../veritate_mri/training/save.py#L504)):

- Plain stem `"chat_v1"`: used as-is.
- Multicorpus mix `"stem1:w1,stem2:w2,..."`: the highest-weight stem wins (first on ties).
- Single `"prefix:stem"` (no numeric tail): the part after the last `:`.

The stem resolves to `trainers/corpus/<stem>_train.bin`; if that file is missing, `generation` is skipped with a logged error.

### Model-type gate

The language probes in `LANGUAGE_DUMPS` (`probe`, `grades`, `reading_comprehension`, `math`, `grammar`, `reasoning`, `concepts`, `writing_health`, `generation`) are meaningless for a non-text model, so `save()` adds them to `skip` when the run's `model_type` is in `NON_LANGUAGE_TYPES` (`statistical`, `other`). `language` and `code` both consume text, so both keep the full set. The architecture probes (`classroom`, `surprise`, `quant_kl`) and the checkpoint itself always run. `model_type` comes from the `VERITATE_MODEL_TYPE` env that `trainer_runner` sets from the dashboard's selector; it cannot ride in through the parsed args because trainers `parse_known_args()` and drop the unknown `--model_type` flag. `_sync_model_meta()` (mirroring `_sync_qat_flag`) promotes `model_type` from that env into `config.json`'s `training_args` on every save, so the `/run/<name>/eval_deep` route, the dashboard panels, and resumed runs read the same value the checkpoint already carries. Without it the key never lands in `config.json` and a stale `language` default stands. Resumed runs (no env) read `model_type` from the existing config.

## Dependencies

- [readers/paths.py](../../../veritate_mri/readers/paths.py): for `model_dir`, `checkpoints_dir`, `train_csv` paths.
- [training/checkpoint_probe.py](../../../veritate_mri/training/checkpoint_probe.py): generates the dump artifacts.
- [readers/models.py](../../../veritate_mri/readers/models.py): validates model names before write.

## Pitfalls

- Trainers must not write `.pt` files or `train.csv` directly. Every write goes through this module so the format and the dump suite stay consistent.
- Adding a new CSV column breaks `train_csv.py` and every consumer. Coordinate the schema change across reader + every dashboard chart + the heartbeat fallback in the same commit.
- The dump suite runs synchronously at checkpoint time and can take seconds to minutes on large models. Don't add per-step calls to anything in the dump pipeline.
- A skipped dump is not a failed dump: skip reasons (missing corpus stem, model-type gate) go to the server log via `logmod.error`, never as a `DUMP FAILED:` line in the run log. A bad stem resolution silently skips `generation` for an entire run with nothing to grep in the run log. Verifying a run's dumps means checking the artifact SET in `hooks/step_<N>/` against the canonical filenames, not just grepping `DUMP FAILED`.
- `_validate_name` (via `models.is_valid_name`) gates every write. Accepted format is `NAME_RE` in [readers/models.py](../../../veritate_mri/readers/models.py): lowercase alphanumerics and underscores, starting and ending on an alphanumeric.
