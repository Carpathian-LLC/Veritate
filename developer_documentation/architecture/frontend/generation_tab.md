# Generation tab

## What it is

Chat-style interaction with the currently-loaded model. The user types a prompt; the response streams back token-by-token alongside per-token telemetry (FFN heatmaps, logit lens, top neurons).

## How it works

Markup at [index.html:73–500](../../../veritate_mri/web/index.html#L73). Default-active tab on page load.

- Prompt input + mode row (autocomplete vs chat vs agent vs reasoning).
- Response area renders streamed bytes from `/generate`.
- Per-token frame visualizations: FFN heatmap, logit lens table, top neurons, lens-logit, decode chain.

The submit handler POSTs to `/generate` and reads an SSE-style stream. Each event delivers a frame: `{byte, prob, lens, ffn_top, neuron_top, ...}`. Frames are appended to an in-memory `frames` array; the active frame is whatever index the user is scrubbing.

Scrubbing the timeline re-renders all telemetry panels for the selected frame. The current frame is also annotated on the decode chart.

## Dependencies

- `/generate` route at [backends_routes.py](../../../veritate_mri/routes/backends_routes.py).
- PyTorch inference brain at [veritate_mri/inference/backends/pytorch.py](../../../veritate_mri/inference/backends/pytorch.py) — see [../backend/inference_brain.md](../backend/inference_brain.md).
- `/meta` for the current model's layers and hidden size (drives canvas dimensions).
- Canvas helpers in [canvas_rendering.md](canvas_rendering.md).

## Pitfalls

- Frames buffer can grow large for long generations; the existing UI caps it but check before extending.
- Mode switching changes which decode strategy the backend uses. The mode value is part of the `/generate` payload.
- Chat history persists in localStorage ([state_persistence.md](state_persistence.md)) but frames do not — switching tabs and back drops the visualization.
