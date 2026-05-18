---
title: "Recovering models that disappeared after an engine update"
date: 2026-05-09
tags: [troubleshooting, models, format, engine]
summary: What to do when the dashboard stops listing your trained models after an engine rework or a major build update. The PyTorch checkpoints in `models/` are the source of truth; the engine `.bin` is a derived artifact you can always re-export.
---

If you updated Veritate and your trained models are no longer visible in the dashboard, the model files themselves are almost always still on disk. What you are seeing is one of three things:

1. The engine `.bin` file format was bumped and the old `.bin` no longer loads.
2. The `models/` folder was wiped (manual delete, fresh install in a different directory, or a re-clone of the wrong repo).
3. The dashboard is pointing at a different `models/` location than the one your checkpoints live in.

This page walks each case end to end.

## first: confirm what is actually missing

Open a terminal at the repo root and look at what is in `models/`:

```
ls models/
ls models/<model_name>/
```

Each model folder typically contains:

- `*.pt` PyTorch checkpoint(s). **This is the source of truth.** Training writes these.
- `*.bin` engine model(s). **This is derived.** Generated from the `.pt` by the export step.
- `manifest.json` metadata for the dashboard.
- `hooks/` user-defined probe hooks (optional).

Any model with a `.pt` is recoverable. A model with only a `.bin` and no `.pt` cannot be retrained back to the same weights without the original training run.

## case 1: `.pt` is present but the model does not appear in the dashboard

This is the engine `.bin` format mismatch case. The Veritate C engine bumps its on-disk format whenever the layout changes (for example, build 5 retired v10 ternary `.bin` and introduced v11 with `quant_mode` + `n_experts` + `router_topk` header fields). Older `.bin` files fail to load and the dashboard hides the model.

The fix is to re-export the engine `.bin` from your `.pt` checkpoint. From a Python shell at the repo root:

```python
from veritate_mri import export

# ternary / MoE models:
export.export_checkpoint_ternary("<model_name>", <step>)

# INT8, dense (non-MoE) models:
export.export_checkpoint("<model_name>", <step>)
```

`<step>` is the training step number from the `.pt` filename. If you only have one checkpoint, point at that step.

The dashboard's topbar banner lists every model whose `.bin` is stale relative to the engine version, so you do not have to guess which ones need re-export.

## case 2: the `models/` folder is empty or missing

`models/` is a self-contained git repo, separate from the parent Veritate repo, and `models/` itself is gitignored in the parent so the in-app updater cannot touch it. If the folder is gone, it was removed outside of Veritate (manual delete, OS-level cleanup, or you cloned the parent repo into a fresh directory and never copied `models/` over).

To restore the upstream barebones models repo (configs and manifests, not trained weights):

1. In the dashboard, open **Settings → Sync → Models**.
2. Click **update**. With no `models/` folder present, the sync will clone the remote into place.

This brings back the upstream model definitions only. **Trained weights you produced locally are not on the remote** and cannot be recovered from upstream. If you have a backup of your old `models/` folder, copy your model directory back into `models/` and the dashboard will pick it up on the next refresh.

If your `.pt` files are gone too, the only path forward is to retrain.

## case 3: dashboard is pointing at the wrong `models/`

If you have multiple Veritate installations (or you cloned the repo into a new path), the dashboard reads from the `models/` directory next to whichever copy of `veritate_mri/` is currently running. Confirm by checking the path the server reports in its startup log, or by running:

```python
from veritate_mri.readers import paths
print(paths.MODELS_ROOT)
```

If the path is not where your trained models live, either move the models into that path, or relaunch Veritate from the directory that has them.

## prevention

- **Never delete `data/`, `models/`, or `trainers/` when reinstalling.** All three are gitignored in the parent repo and survive the in-app updater's `reset --hard`. They will not survive a manual `rm -rf` or re-clone into a fresh directory.
- **Keep `.pt` checkpoints.** They are the only thing that lets you recover from any engine format bump. The `.bin` is disposable; the `.pt` is not.
- **After any engine version bump,** watch the topbar banner. The dashboard tells you exactly which models need a re-export and will keep telling you until you run it.
