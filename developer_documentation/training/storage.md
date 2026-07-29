# Model storage layout

Layout of `models/<name>/` directories. Gitignored. Every trainer writes here through [veritate_mri/training/save.py](../../veritate_mri/training/save.py); readers under [veritate_mri/readers/](../../veritate_mri/readers/) consume it.

## Save guarantee

Every training path persists through `save.save()`, which always writes the full dump suite (`hooks/step_<N>/`) plus `config.json` and the checkpoint, and `append_train_row()` for `train.csv`. No training path bypasses the hooks dump. The shared plugin loop ([trainers/common/vanilla_trainer.py:536](../../trainers/common/vanilla_trainer.py#L536)), the per-trainer loops (`trainers/<id>/trainer.py`, reaching `save` via `veritate_core.plugin.save`), and the grounded SFT ([experiments/v2/rag/sft_grounded.py:66](../../experiments/v2/rag/sft_grounded.py#L66)) all call it. See [save.md](../architecture/backend/save.md).

## Layout

```
models/<name>/
├── config.json                       # shape, training_args, n_params, plugin id, wrote_at
├── train.csv                         # canonical training log per save.py:38
├── checkpoints/
│   ├── step_<N>.pt                   # PyTorch checkpoint dict: {model, step, args, ...}
│   ├── step_<N>.pt.tmp               # atomic-write tempfile (transient)
│   └── ...
├── hooks/
│   └── step_<N>/                     # per-checkpoint dump suite
│       ├── probe.json                # top-k FFN neurons + logit lens
│       ├── lens.npz                  # per-layer logits + residual norms
│       ├── classroom.json            # per-grade reading perplexity
│       ├── grades.json               # pass/fail at grade bands
│       ├── math.json                 # capability eval
│       ├── grammar.json              # capability eval
│       ├── reasoning.json            # capability eval
│       ├── concepts.json             # 50-concept surprise probe
│       ├── surprise.json             # held-out surprise
│       ├── quant_kl.json             # KL between fp32 and quantized predictions
│       ├── writing_health.json       # higher-tier eval
│       ├── reading_comprehension.json
│       └── generation.json           # sample generations at this step
└── veritate.bin                      # exported engine artifact (when generated)
```

## File responsibilities

| Path                                  | Writer                                                       | Reader                                                              |
| ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------- |
| `config.json`                         | Trainer's `write_config()`                                   | [readers/config.py](../../veritate_mri/readers/config.py)           |
| `train.csv`                           | [save.append_train_row()](../architecture/backend/save.md)   | [readers/train_csv.py](../../veritate_mri/readers/train_csv.py)     |
| `checkpoints/step_<N>.pt`             | `torch.save()` via atomic rename                             | [readers/checkpoints.py](../../veritate_mri/readers/checkpoints.py) |
| `hooks/step_<N>/*`                    | [checkpoint_probe.py](../architecture/backend/checkpoint_probe.md) | [readers/hooks.py](../../veritate_mri/readers/hooks.py)         |
| `veritate.bin`                        | [training/export.py](../../veritate_mri/training/export.py)  | C engine, [readers/bin.py](../../veritate_mri/readers/bin.py)       |

## Name format

Two valid forms accepted by `models.is_valid_name`:

- **User-friendly:** `<slug>_<size>`, e.g., `chatty_otter_85m`.
- **Four-part:** `<corpus>_<size>_<precision>_<version>`, e.g., `wikitext_25m_v1`.

Validation lives at [veritate_mri/readers/models.py](../../veritate_mri/readers/models.py).

## Atomic writes

`.pt` checkpoints are written via `torch.save(... + ".tmp")` then `os.replace(tmp, final)`. A killed trainer leaves a stale `.tmp` but never a partial canonical file.

## Checkpoint retention

Nothing prunes checkpoints automatically. A run writes one `step_<N>.pt` every `ckpt_every` steps and keeps every one: the 200M pretrain at `ckpt_every` 500 held **158 checkpoints / 239 GB by step 79,000 of 164,388**, on track to need ~500 GB by the end. [veritate_mri/training/retention.py](../../veritate_mri/training/retention.py) owns the cleanup, exposed as `POST /models/checkpoints/prune` and the **prune checkpoints** button beside *fork to new model* on the Training tab.

- A step survives if it is on the ladder (`step % keep_every == 0`), among the newest `keep_last`, or younger than `MIN_AGE_S` (1200 s). The age guard exists because a live trainer renames `step_<N>.pt.tmp` into place; never race it.
- **`hooks/step_<N>/` is never touched.** The dump suite is the research artifact and costs ~8 MB a step against 1.5 GB for the weights, so retention keeps the full analysis history and drops only the ability to *resume* from a passed step. `plan()` returns `hooks_kept` so the UI can say so.
- `plan()` is pure and `prune()` re-plans internally, so a stale plan cannot be replayed against a directory that has moved on. The route defaults `dry_run` to **true**: a malformed request plans, it does not delete. The dashboard's confirm button stays disabled until a preview has rendered and re-disables when a knob changes.
- `keep_last` must be >= 1. Pruning a model to zero checkpoints destroys it and is refused.
- Defaults (`keep_every` 5000, `keep_last` 4) are the initial values of the dashboard controls, not policy: both are set per prune (rule 11).

## Pitfalls

- Don't write directly to `models/<name>/`. Always go through `save.py` so the dump suite stays consistent.
- `models/` is gitignored: clean clones start empty.
- Stale `.tmp` files accumulate if trainers are killed mid-checkpoint. Safe to delete any `*.tmp` when no training is running.
- **`config.json` bootstraps once, then only named keys are re-synced.** `_ensure_config` writes the file only when absent, so a fork plus resume would otherwise leave the fork source's `training_args` in place forever. `_sync_run_args` re-records the keys in `RUN_ARG_KEYS` from the live run on every save, so a resumed model describes how IT trained. The set covers the corpus mix and loss mask, the lr and schedule, batch/seq/chunking, total_steps, precision and optimizer, the **optimization shape** (weight_decay, betas, grad_clip, label_smoothing, wsd decay), the **seed**, the **cadence** (ckpt_every, log_every, eval_every, eval_iters), and the **memory regime** (use_act_ckpt, use_8bit_adam). The cadence keys were added after a measured gap: a resumed SFT checkpointed every 250 steps while its config still reported the fork source's 1500, so the recorded rollback granularity was 6x coarser than the run's. `model_type` is deliberately NOT in that set: rule 24a's "hand-edit `training_args.model_type` to correct a mislaunched run" workflow depends on the save path not overwriting it, and `_sync_model_meta` owns that key. `description` and `name` are hand-owned for the same reason. Anything outside `RUN_ARG_KEYS` is left alone.
