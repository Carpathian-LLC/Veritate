# Model storage layout

Layout of `models/<name>/` directories. Gitignored. Every trainer writes here through [veritate_mri/training/save.py](../../veritate_mri/training/save.py); readers under [veritate_mri/readers/](../../veritate_mri/readers/) consume it.

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
├── veritate.bin                      # exported engine artifact (when generated)
└── coral_meta.json                   # only for coral-merged models
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

- **User-friendly:** `<slug>_<size>` — e.g., `chatty_otter_85m`.
- **Legacy:** `<corpus>_<size>_<precision>_<version>` — e.g., `tinystories_25m_v1`.

Validation lives at [veritate_mri/readers/models.py](../../veritate_mri/readers/models.py).

## Atomic writes

`.pt` checkpoints are written via `torch.save(... + ".tmp")` then `os.replace(tmp, final)`. A killed trainer leaves a stale `.tmp` but never a partial canonical file.

## Pitfalls

- Don't write directly to `models/<name>/`. Always go through `save.py` so the dump suite stays consistent.
- `models/` is gitignored — clean clones start empty.
- Stale `.tmp` files accumulate if trainers are killed mid-checkpoint. Safe to delete any `*.tmp` when no training is running.
