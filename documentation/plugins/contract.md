# Trainer plugin contract

The platform surface that trainers are allowed to import. Trainers may only reach `veritate_core.plugin` (or the legacy `veritate.plugin` alias via the [shim](../architecture/backend/veritate_shim.md)). Direct imports from `veritate_mri/` or `veritate_engine/` are forbidden by preflight rule 39.

## Plugin bundle layout

```
trainers/<plugin_id>/
├── manifest.json
└── plugin.py
```

Optional per-trainer files (corpus builders, helpers used by only this trainer) live in the same directory. Shared helpers live in `trainers/common/` (preflight rule 36).

## manifest.json schema

```json
{
  "name": "Human-readable name",
  "description": "One-sentence purpose.",
  "kind": "trainer",
  "flow": ["scratch", "continue"],
  "defaults": {
    "size": "30m",
    "batch": 8,
    "seq": 256,
    "total_steps": 6000
  }
}
```

- `name` — display name in the dashboard.
- `kind` — `"trainer"` (currently the only kind).
- `flow` — list of valid entry modes. `scratch` = new model; `continue` = resume from a checkpoint.
- `defaults` — every argparse arg the plugin accepts, with its default value. The dashboard generates form fields from this; missing keys mean missing form fields.

## plugin.py contract

The plugin is a standalone Python script. It must:

1. **Set up `sys.path`** so the package imports resolve when the script is launched as a subprocess:

   ```python
   HERE = os.path.dirname(os.path.abspath(__file__))
   REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
   sys.path.insert(0, REPO)
   sys.path.insert(0, os.path.join(REPO, "veritate_mri"))
   ```

2. **Import only from the allowed surface:**

   ```python
   from veritate_core.model import Veritate          # or legacy: from veritate.model import Veritate
   from veritate_core import qat as vqat             # or legacy: from veritate import qat as vqat
   from training import save                         # or legacy: from veritate.plugin import save
   from readers import paths                         # or legacy: from veritate.plugin import paths
   ```

   Direct imports from `veritate_mri.routes`, `veritate_mri.runtime`, or `veritate_engine` are forbidden.

3. **Build argparse from manifest defaults** (the convention; not strictly enforced).

4. **Write the canonical CSV** via [`save.append_train_row(name, step, split, loss, lr, grad_norm, tok_per_s, wall_s, seed)`](../architecture/backend/save.md) at every log interval.

5. **Save checkpoints** via `save.save(model, name, description, step, ...)` so the dump suite runs ([checkpoint_probe](../architecture/backend/checkpoint_probe.md)).

6. **Stream progress to stdout** with `print(..., flush=True)`. The dashboard tails stdout into the in-memory log ring.

## Plugin surface (`veritate_core.plugin`)

| Module                                                                                | Provides                                                       |
| ------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| [hardware](../../veritate_core/plugin/hardware.py)                                    | `physical_cores()`, device detection                           |
| [multicorpus](../../veritate_core/plugin/multicorpus.py)                              | `make_mixed_loader(stems_or_spec, ...)` for `"a+b"` or `"a:0.5,b:0.5"` |
| [oom_recovery](../../veritate_core/plugin/oom_recovery.py)                            | Wrap a training step to catch and recover from CUDA OOM        |

## Launch paths

- **Dashboard Training tab** → calls `/trainers/<id>/start` → `trainer_runner.start(plugin_id, args)`.
- **Direct CLI** → `python trainers/<id>/plugin.py --arg val ...`. Bypasses the runner.
- **Programmatic** → `from training import trainer_runner; trainer_runner.start(id, args)`.

The dashboard's Training tab is the canonical path. Direct CLI is valid for experiments but bypasses the heartbeat primary-detection path; the [heartbeat fallback](../architecture/backend/heartbeat.md) (train.csv mtime scan) catches direct-CLI runs.

## Forbidden

- No `import` from `veritate_mri.routes`, `veritate_mri.inference`, `veritate_engine`, or any other internal path.
- No writing to `models/<name>/` outside the `save.py` API.
- No background threads outside the main training loop.
- No new files outside the plugin bundle and `models/<name>/`.

## Pitfalls

- The `from veritate.X import` aliases work today via the [shim](../architecture/backend/veritate_shim.md). New code should still prefer the canonical `from veritate_core.X import` and `from training import save` paths.
- Subprocess stdout is line-buffered. Use `flush=True` or unbuffered output (`python -u`) so logs appear in the dashboard in real time.
- Single-instance enforced by `trainer_runner.start()`. Direct CLI launches can be concurrent — be deliberate.
