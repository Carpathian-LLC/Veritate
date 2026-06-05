# Prune panel

## What it is

Standalone neuron-pruning panel inside the Models tab. Analyzes dead FFN units in a checkpoint and can generate a width-pruned trainer plugin.

## How it works

File: [veritate_mri/web/prune.js](../../../veritate_mri/web/prune.js) + [prune.css](../../../veritate_mri/web/prune.css). IIFE module per the standalone-module pattern.

- Lists vanilla checkpoints via `/pytorch-models`.
- POSTs to `/pruning/report` for a dead-neuron analysis and a target-size estimate.
- POSTs to `/pruning/generate_plugin` to write a new trainer plugin with the pruned shape baked in.

Mounts into pre-defined IDs in the Models tab (e.g., `#pruneModel`, `#pruneMsg`, `#pruneGenerate`).

## Dependencies

- `/pruning/report` and `/pruning/generate_plugin` from [pruning_routes.py](../../../veritate_mri/routes/pruning_routes.py).
- The Models tab markup for the host IDs.
- [standalone_modules.md](standalone_modules.md) for the IIFE pattern.

## Pitfalls

- Generated plugins are written into `trainers/<id>/` as standalone bundles. The user is responsible for launching them via the Training tab; the panel doesn't auto-start training.
- The dead-neuron threshold defaults to a percentile-based cutoff. Aggressive pruning can damage quality; the panel shows an estimated quality drop alongside the size savings.
