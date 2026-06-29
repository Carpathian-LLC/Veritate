# inference_brain (PyTorch)

## What it is

The PyTorch inference path used by the Generation tab. Lives under [veritate_mri/inference/](../../../veritate_mri/inference/). Co-exists with the C engine — selectable per request.

## How it works

- [pytorch.py](../../../veritate_mri/inference/backends/pytorch.py) — loads a `.pt` checkpoint, holds the model in memory with a lock for thread safety. Reloaded on model switch or idle-unload.
- [decode/](../../../veritate_mri/inference/decode/) — decoding strategies: greedy, nucleus sampling, temperature schedule. Per-token telemetry emission for the dashboard.
- [agent/](../../../veritate_mri/inference/agent/) — multi-step reasoning (chain-of-thought, tool use).
- [addons/](../../../veritate_mri/inference/addons/) — pluggable behaviors (embeddings, classification heads).

Held under `app.config["BRAIN"]`. Loaded eagerly at startup if `pytorch_load_mode="always"`; lazily on first `/generate` if `on_demand`. The idle watcher unloads it after `pytorch_idle_unload_secs` of inactivity in `on_demand` mode.

## Dependencies

- [veritate_core/model.py](../../../veritate_core/model.py) — model class.
- [veritate_core/load.py](../../../veritate_core/load.py) — checkpoint loading.
- [routes/backends_routes.py](../../../veritate_mri/routes/backends_routes.py) — `/generate`, `/meta`, neuron lookups.
- Settings: `pytorch_load_mode`, `pytorch_idle_unload_secs`, `device_preference`.

## Pitfalls

- `BRAIN` is a single-instance global. Two simultaneous `/generate` calls serialize on the brain lock.
- Switching models reloads — first request after a switch is slow.
- The brain holds GPU memory while loaded; long idle periods on a shared GPU machine should use `on_demand` to release memory automatically.
