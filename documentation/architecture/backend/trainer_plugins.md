# Trainer plugins

## What they are

Self-contained trainer bundles under [trainers/](../../../trainers/). `trainers/` is a synced checkout from an upstream canonical repo; the upstream is the source of truth (see [trainers/.sync_state.json](../../../trainers/.sync_state.json)). Each trainer is a directory containing a `manifest.json` declaring sizes + args + flow, and a `trainer.py` that is a standalone Python script. The dashboard discovers them via [readers/trainers.py](../../../veritate_mri/readers/trainers.py); launching one spawns a subprocess via [trainer_runner.md](trainer_runner.md). The native trainer is listed first; plugins follow ordered by headline model size, parsed from the id token (`10m`, `1b3`, `50b`), with declared max params as the fallback for non-scale trainers.

## Layout

```
trainers/<plugin_id>/
├── manifest.json     # declares sizes, defaults, flow
└── trainer.py        # standalone trainer script
```

`manifest.json` example:

```json
{
  "name": "Distill Teacher",
  "kind": "trainer",
  "flow": ["scratch"],
  "defaults": {
    "size": "30m",
    "batch": 8,
    "seq": 256,
    "skip_gen": false
  }
}
```

`flow` is the list of valid starting points: `scratch` (new model), `continue` (resume from checkpoint), or both. `sizes` maps a size label to a model shape (`layers`, `hidden`, `ffn`, `heads`, `params`).

## trainer.py pattern

The only platform surface a trainer reaches is `veritate_core.plugin` (preflight rule 39); it never imports `veritate_mri` or `veritate_engine` directly.

```python
from veritate_core.plugin import save, paths, model as _model_mod, qat, multicorpus

def main():
    args = parse_args()                       # built from manifest defaults
    model = _model_mod.build(...)
    for step in range(args.total_steps):
        loss = train_step(model)
        save.append_train_row(args.name, step, "train", loss, lr, gnorm, tps, wall, seed)
        if step % args.ckpt_every == 0:
            save.save(model, name, step, optimizer=opt, args=ckpt_args)
```

`save.save(model, name, step, *, optimizer, args, ...)` is the only checkpoint path (see [save.md](save.md)). It writes the checkpoint, `config.json`, and the full hooks dump suite on every call; per-step rows go through `append_train_row`. Most trainers share the loop in [trainers/common/vanilla_trainer.py](../../../trainers/common/vanilla_trainer.py); the per-trainer `trainer.py` sets the shape and recipe and calls in.

## Current plugins

The canonical set (preflight rule 34b). New trainers are not added without explicit permission; new capabilities extend existing trainers or `veritate_core/plugin/`.

| Plugin          | Purpose                                                          |
| --------------- | --------------------------------------------------------------- |
| `veritate_10m`  | Smallest byte-level base                                         |
| `veritate_80m`  | Base trainer at 80M                                              |
| `veritate_200m` | Default base trainer, two-phase QAT, exports to v9 INT8          |
| `veritate_400m` | Base trainer at 400M                                             |
| `veritate_800m` | Base trainer at 800M                                             |
| `veritate_1b`   | Base trainer at 1B                                               |
| `veritate_1b3`  | Base trainer at 1.3B                                             |
| `veritate_3b`   | Base trainer at 3B                                               |
| `veritate_13b`  | Base trainer at 13B                                              |
| `veritate_50b`  | Largest base trainer                                             |
| `common/`       | Shared trainer code (`vanilla_trainer.py`) and corpus builders   |
| `corpus/`       | Built corpus `.bin` files                                        |

## Launching

Three paths:

1. **Dashboard Training tab** → calls `/trainers/<id>/start` → `trainer_runner.start(plugin_id, args)`.
2. **Direct CLI** → `python trainers/<id>/trainer.py --arg val ...`. Bypasses `trainer_runner` and thus the heartbeat primary-detection path; the [heartbeat fallback](heartbeat.md) covers this case.
3. **Programmatic** → import `training.trainer_runner` and call `start()`.

## Sync

Per-file three-state sync against the upstream repo, implemented in [training/sync/](../../../veritate_mri/training/sync/) (`sync_common.py` engine, `trainers_sync.py` for trainers). `.sync_state.json` records the SHA written at the last sync. `GET /trainers/git/files` classifies every file as `current`, `missing`, `update_available`, `modified`, `conflict`, or `orphan` (tracked locally but dropped upstream). `POST /trainers/git/sync` applies per-file actions: `install`, `update`, `force`, `adopt`, `delete`, `skip`. The dashboard's per-file details panel exposes the action buttons; `force` and `delete` route through a confirm dialog. `delete` is valid only for orphans: it removes the local file and drops the tracking entry so it stops resurfacing.

## Dependencies

- [veritate_core/plugin/](../../../veritate_core/plugin/): the only platform surface trainers import (`save`, `paths`, `model`, `qat`, `multicorpus`).
- [training/save.py](save.md): CSV + checkpoint + dump contract (re-exported as `veritate_core.plugin.save`).
- [readers/trainers.py](../../../veritate_mri/readers/trainers.py): plugin discovery.

## Pitfalls

- A trainer's `manifest.json` is the schema. The dashboard generates form fields from `defaults`; missing keys mean missing fields.
- `trainers/` is a synced checkout. Local-only edits get overwritten on the next `/trainers/git/sync`; mirror changes upstream (preflight rule 34a).
- Plugins are subprocesses, so their stdout is the only feedback channel. Use `print(..., flush=True)` for log visibility.
- Single-instance training enforced by `trainer_runner`. Direct CLI launches bypass that lock.
