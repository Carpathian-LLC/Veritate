# Learning tab

## What it is

Walks through every checkpoint of a training run and shows how the model evolved: FFN activations, top neurons, saturation, quantization KL, confidence evolution, reading level by grade.

## How it works

Markup at [index.html:504–900](../../../veritate_mri/web/index.html#L504). Tab labeled "Models" in the UI.

- A timeline slider picks a checkpoint step. Each step has a hooks directory at `models/<name>/hooks/step_<N>/` with per-checkpoint artifacts.
- `ensureLearningLoaded()` ([index.js around the activateTab branch](../../../veritate_mri/web/index.js#L2102)) fetches `/timelines/<name>` to discover available steps + their artifact metadata.
- On step change, the relevant artifacts (probe.json, lens.npz, classroom.json, concepts.json, grades.json) are fetched and rendered.
- Canvas drawers prefixed `L` (e.g., `cFfnL`, `cTopL`, `cTelL`) render the side-by-side checkpoint view, separate from training-tab canvases.

`renderTier2ForLearning` extends the base render with tier-2 panels (math, grammar, reasoning capability evals when available).

## Dependencies

- Backend [checkpoint_probe.py](../../../veritate_mri/training/checkpoint_probe.py) produces the hooks/ artifacts at each checkpoint.
- Reader [hooks.py](../../../veritate_mri/readers/hooks.py) lists and loads them.
- [canvas_rendering.md](canvas_rendering.md) for the chart helpers.
- [../backend/checkpoint_probe.md](../backend/checkpoint_probe.md) — what's in each artifact.

## Pitfalls

- Checkpoint dumps are heavy. Scrubbing fast can stack fetch requests; the UI drops in-flight requests when a newer step is selected.
- `lens.npz` is a binary numpy archive; parsing is done client-side. Large vocab (256-byte) is fine; large hidden dims slow rendering.
- The "L" canvas suffix is easy to confuse with the training tab's "T" suffix. Stay consistent when adding panels.
