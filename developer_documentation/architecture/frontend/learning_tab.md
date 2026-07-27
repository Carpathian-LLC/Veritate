# Learning tab

## What it is

Walks through every checkpoint of a training run and shows how the model evolved: FFN activations, top neurons, saturation, quantization KL, confidence evolution, reading level by grade.

## How it works

Markup at [index.html:504-900](../../../veritate_mri/web/index.html#L504). Tab labeled "Models" in the UI.

- A timeline slider picks a checkpoint step. Each step has a hooks directory at `models/<name>/hooks/step_<N>/` with per-checkpoint artifacts.
- `ensureLearningLoaded()` ([index.js around the activateTab branch](../../../veritate_mri/web/index.js#L2102)) fetches `GET /timelines` for the run list, then `GET /timeline/<name>/timeline.json` for that run's available steps and their artifact metadata.
- On step change, the relevant artifacts (probe.json, lens.npz, classroom.json, concepts.json, grades.json) are fetched and rendered.
- Canvas drawers prefixed `L` (e.g., `cFfnL`, `cTopL`, `cTelL`) render the side-by-side checkpoint view, separate from training-tab canvases.

`renderTier2ForLearning` extends the base render with tier-2 panels (math, grammar, reasoning capability evals when available).

### Model-shape scaling

Every model dimension the dashboard renders comes from one module-global, `modelShape` (`layers`, `hidden`, `ffn`, `ffn_buckets`, `vocab`, `seq`) in [index.js](../../../veritate_mri/web/index.js). `setModelShape(meta)` fills it from `/meta` and from the generate-stream meta frame, keeping any field the newer payload omits. A field left at `0` means "not known yet": callers omit the control, caption, or plot instead of substituting a literal.

- `setLayerCount(n)` sets `modelShape.layers`, rebuilds both region legends (`#ffnRegionLegend`, `#ffnRegionLegendL`) via `_buildFfnLegend`, and re-applies the input caps. `renderLearning` calls it when the scrubbed checkpoint's layer count differs, mirroring the Generation tab.
- `applyShapeControls()` sets `max` on `#ablLayer`, `#ablNeuron`, `#topk`, and `#genRepWindow`, and removes the attribute when the dimension is unknown. The HTML carries no `max` for these.
- `applyShapeText()` fills `[data-shape="<field>"]` spans in the explainer prose, falling back to the span's `data-unknown` wording, and hides `[data-shape-when="<field>"]` clauses whose number is not known. It runs after `injectDetails()` so cloned `<template>` copy is covered.
- `_regionBounds(n)` is the only depth split (thirds: sensory / association / output) and `regionOf(layer, total)` is the only classifier; `REGIONS` carries the per-band name, CSS classes, and colors. Nothing re-derives the split.
- `runVocab(config)` resolves a training run's vocabulary from its own `config.json` (`shape.vocab`, then `vocab`), then the loaded model. Panels that scale perplexity against random output require it and say so when it is missing.

`ffn_buckets` is only present on the generate-stream meta frame, not on `/meta`, so the FFN bucket-count clause stays hidden until the first generation.

## Dependencies

- Backend [checkpoint_probe.py](../../../veritate_mri/training/checkpoint_probe.py) produces the hooks/ artifacts at each checkpoint.
- Reader [hooks.py](../../../veritate_mri/readers/hooks.py) lists and loads them.
- [canvas_rendering.md](canvas_rendering.md) for the chart helpers.
- [../backend/checkpoint_probe.md](../backend/checkpoint_probe.md), what's in each artifact.

## Pitfalls

- Checkpoint dumps are heavy. Scrubbing fast can stack fetch requests; the UI drops in-flight requests when a newer step is selected.
- `lens.npz` is a binary numpy archive; parsing is done client-side. Large vocab (256-byte) is fine; large hidden dims slow rendering.
- The "L" canvas suffix is easy to confuse with the training tab's "T" suffix. Stay consistent when adding panels.
