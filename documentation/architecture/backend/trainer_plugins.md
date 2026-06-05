# Trainer plugins

## What they are

Self-contained trainer bundles under [trainers/](../../../trainers/). Each is a directory containing a `manifest.json` declaring CLI args + flow, and a `plugin.py` that is a standalone Python script. The dashboard discovers them via [readers/trainers.py](../../../veritate_mri/readers/trainers.py); launching one spawns a subprocess via [trainer_runner.md](trainer_runner.md).

## Layout

```
trainers/<plugin_id>/
├── manifest.json     # declares args, defaults, flow
└── plugin.py         # standalone trainer script
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

`flow` is the list of valid starting points: `scratch` (new model), `continue` (resume from checkpoint), or both.

## plugin.py pattern

```python
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "veritate_mri"))

from veritate.model import Veritate            # via shim -> veritate_core.model
from veritate.plugin import save, paths        # via shim -> training, readers

def parse_args():
    # build argparse from manifest defaults
    ...

def main():
    args = parse_args()
    model = Veritate(...)
    for step in range(args.steps):
        loss = train_step(model)
        save.append_train_row(args.name, step, "train", loss, lr, gnorm, tps, wall, seed)
        if step % args.ckpt_every == 0:
            save.save(model, args.name, args.description, step, ...)
```

Plugins write CSV per the [save.md](save.md) contract and produce checkpoints under `models/<name>/`.

## Current plugins

| Plugin               | Purpose                                                                |
| -------------------- | ---------------------------------------------------------------------- |
| `distill_teacher`    | Sequence-level distillation from Ollama (llama3.1:8b etc.)             |
| `multimind_m1`       | Schema-state mixture                                                   |
| `multimind_m3`       | Holographic Hebbian memory adapter                                     |
| `multimind_mega`     | Scale-up of m3 for larger models                                       |
| `example_plugin`     | Template for new trainers                                              |
| `common/`            | Shared utilities (e.g., pg19 corpus builder)                           |
| `corpus/`            | Built corpus binaries                                                  |

## Launching

Three paths:

1. **Dashboard Training tab** → calls `/trainers/<id>/start` → `trainer_runner.start(plugin_id, args)`.
2. **Direct CLI** → `python trainers/<id>/plugin.py --arg val ...`. Works because plugins set up `sys.path` themselves. Bypasses `trainer_runner` and thus the heartbeat primary-detection path; the [heartbeat fallback](heartbeat.md) covers this case.
3. **Programmatic** → import `training.trainer_runner` and call `start()`.

## Dependencies

- [veritate/](../../../veritate/) shim — enables the `from veritate.X import` imports.
- [veritate_core/model.py](../../../veritate_core/model.py) — model class.
- [training/save.py](save.md) — CSV + checkpoint contract.
- [readers/trainers.py](../../../veritate_mri/readers/trainers.py) — plugin discovery.

## Pitfalls

- A new plugin's `manifest.json` is the schema. The dashboard generates form fields from `defaults`; missing keys mean missing fields.
- Plugins are subprocesses, so their stdout is the only feedback channel. Use `print(..., flush=True)` for log visibility.
- Single-instance training enforced by `trainer_runner`. Direct CLI launches bypass that lock — be deliberate about concurrent runs.
