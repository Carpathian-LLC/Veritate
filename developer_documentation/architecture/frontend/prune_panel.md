# Prune panel

## What it is

Standalone neuron-pruning panel inside the Models tab. Analyzes dead FFN units in a checkpoint and can generate a width-pruned trainer plugin.

## How it works

File: [veritate_mri/web/prune.js](../../../veritate_mri/web/prune.js) + [prune.css](../../../veritate_mri/web/prune.css). IIFE module per the standalone-module pattern.

- Lists vanilla checkpoints via `/pytorch-models`.
- GETs `/pruning/report` for a dead-neuron analysis, per-layer keep fractions, and a target-size estimate.
- POSTs to `/pruning/generate_plugin` to write a new trainer plugin with the pruned shape baked in.

Self-builds its panel and inserts it at the top of the Models (learning) tab; it creates its own controls (`#pruneModel`, `#pruneStep`, `#pruneAnalyze`, `#pruneGenerate`, `#pruneMsg`) rather than binding to pre-existing markup.

## Dependencies

- `/pruning/report` and `/pruning/generate_plugin` from [pruning_routes.py](../../../veritate_mri/routes/pruning_routes.py).
- The `.tab-body[data-tab="learning"]` container it inserts into.
- [standalone_modules.md](standalone_modules.md) for the IIFE pattern.

## Pitfalls

- Generated plugins are written into `trainers/<id>/` as standalone bundles and must be launched manually from the Training tab; the panel does not auto-start training.
- The report displays dead-neuron percent, per-layer keep fractions, and size before/after. A well-trained dense base has few dead units, so the recommended plan often prunes little; forcing a deeper cut trades quality for size.
