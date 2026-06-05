# save (checkpoint + CSV contract)

## What it is

The unified save module at [veritate_mri/training/save.py](../../../veritate_mri/training/save.py). Every trainer calls into it for two things: per-step CSV append and per-checkpoint full dump.

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

This CSV is the source of truth for training curves. Every reader (`train_csv.py`), every dashboard chart (Training, Coral Lab, Learning), and the heartbeat fallback detector consumes it.

## Checkpoint save

`save(model, name, description, step, ...)` writes:

1. `models/<name>/config.json` (or updates it).
2. `models/<name>/checkpoints/step_<N>.pt` via atomic rename (`.tmp` then `os.replace`).
3. The full dump suite into `models/<name>/hooks/step_<N>/`:
   - `probe.json`, `lens.npz`, `classroom.json`, `grades.json`, `math.json`, `grammar.json`, `reasoning.json`, `concepts.json`, `surprise.json`, `quant_kl.json`, `writing_health.json`, `reading_comprehension.json`, `generation.json`.

The dump files are produced by [checkpoint_probe.py](checkpoint_probe.py.md) and renamed per `RENAME_MAP_TEMPLATE` at [save.py:51](../../../veritate_mri/training/save.py#L51).

## Dependencies

- [readers/paths.py](../../../veritate_mri/readers/paths.py) — for `model_dir`, `checkpoints_dir`, `train_csv` paths.
- [training/checkpoint_probe.py](../../../veritate_mri/training/checkpoint_probe.py) — generates the dump artifacts.
- [readers/models.py](../../../veritate_mri/readers/models.py) — validates model names before write.

## Pitfalls

- Trainers must not write `.pt` files or `train.csv` directly. Every write goes through this module so the format and the dump suite stay consistent.
- Adding a new CSV column breaks `train_csv.py` and every consumer. Coordinate the schema change across reader + every dashboard chart + the heartbeat fallback in the same commit.
- The dump suite runs synchronously at checkpoint time and can take seconds to minutes on large models. Don't add per-step calls to anything in the dump pipeline.
- `model.exists(name)` is the gate for whether a save is allowed. Name format: `<slug>_<size>` or legacy `<corpus>_<size>_<precision>_<version>`.
