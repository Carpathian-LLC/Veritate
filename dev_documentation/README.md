# Developer documentation

Quick-reference notes on how the platform works internally. Folder layout mirrors the dashboard's tab structure plus the always-visible HUD.

- `hud/` — system metrics overlay (CPU, RAM, GPU, temperatures)
- `generation/` — inference backends (PyTorch brain + C engine)
- `learning/` — probes, atlas, classroom, neuron memory
- `training/` — trainers, dispatch, sync, builders
- `logs/` — log routing and tail
- `settings/` — settings store, hooks, lifecycle
- `wiki/` — embedded wiki
- `platform/` — hardware tiers, OS compatibility, dependency matrix

Each doc is intentionally short. Read the code for the long version; this folder is for orientation. New docs are added as systems are touched.
