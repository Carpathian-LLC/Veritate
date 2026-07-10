# Generation tab

## What it is

Chat-style interaction with the currently-loaded model. The user types a prompt; the response streams back token-by-token alongside per-token telemetry (FFN heatmaps, logit lens, top neurons).

## How it works

Markup at [index.html:73–500](../../../veritate_mri/web/index.html#L73). Default-active tab on page load.

- Prompt input + mode row (autocomplete vs chat vs agent vs reasoning).
- Response area renders streamed bytes from `/generate`.
- Per-token frame visualizations: FFN heatmap, logit lens table, top neurons, lens-logit, decode chain.

The submit handler POSTs to `/generate` and reads an SSE-style stream. Each event delivers a frame: `{byte, prob, lens, ffn_top, neuron_top, ...}`. Frames are appended to an in-memory `frames` array; the active frame is whatever index the user is scrubbing.

### Chat mode framing

Chat mode wraps the prompt in the platform chat markers before sending: `wrapChat` emits `<|system|>\n{persona}\n<|user|>\n{prompt}\n<|assistant|>\n` — the full trained template from `build_chat_corpus.py::render` (system + user + assistant turns), the same byte contract as `/hybrid/chat` and all current chat corpora (chat_v1/v2/v3). The system turn gives the model an identity to answer with; without it, factual and open prompts (e.g. "what's your name?") degenerate into repetition. `persona` comes from the Advanced panel's `genPersona` field (default "You are Veritate, a helpful assistant.", persisted in localStorage); autocomplete mode sends the raw prompt, no framing. These are literal bytes, not special tokens (vocab=256). The backend stops generation via `_chat_stop_seq` ([backends_routes.py](../../../veritate_mri/routes/backends_routes.py)) at the FIRST of a set of turn markers, because a reply must never contain any turn marker: platform-marker prompts stop at `<|end|>`, `<|user|>`, or `<|assistant|>` (the multi-turn SFT model often starts a next `<|user|>` turn without first emitting `<|end|>` — stopping only on `<|end|>` streamed the whole self-conversation); legacy ChatML at `<|im_end|>` or `<|im_start|>`; no stop for plain (autocomplete) prompts. `_stop_on_bytes` watches a rolling tail sized to the longest marker and emits a synthetic `stop` event whose `reason` is the marker that fired. The final chat bubble is trimmed by `stripChatResponse`; the live streaming view is trimmed by `stripChatStream`, which also drops a trailing partial marker (e.g. a forming `<|us`) so self-talk never flashes as bytes arrive.

### Chat repetition control

Chat mode also sends repetition-control params (`rep_window`, `rep_penalty`, `no_repeat_ngram`) read from the Advanced panel (`genRepWindow` / `genRepPenalty` / `genRepNgram`, defaults 256 / 0.5 / 16, persisted). Autocomplete mode omits them, so the server default (off) applies and byte-level autocomplete repetition is untouched. Mechanism and defaults live in [../backend/inference_brain.md](../backend/inference_brain.md).

Scrubbing the timeline re-renders all telemetry panels for the selected frame. The current frame is also annotated on the decode chart.

### Hallucination detection (confidence coloring + grounding + the Hallucination panel)

An output-tools row under `#response` drives it. `#confColorToggle` turns on per-span confidence coloring of the generated text: `renderResponseInto` (given the aligned `frames` and a base offset) groups the visible characters into spans at the `#confColorLevel` granularity (word / sentence / paragraph), colors each span by its mean per-byte `confidence` via `confColor` (the existing red-to-green ramp), and sets a `title` with the numeric score. Coloring defaults off and needs the rich (`token`) stream; fast-mode bytes push no frames, so coloring is skipped. In chat mode `stripChatStream` shifts byte offsets, so `_charByteMap` maps each visible code point back to its last encoded byte to read the right frame.

`#detectHalluc` (and auto-run on `done` when `#hallucAuto` is checked) POSTs to `/hallucination/analyze` with the current model, backend, and prompt/RAG settings, then: (1) `renderGroundedResponse` re-colors the text from the report's own span offsets and underlines each word by grounding (`.g-yes` green / `.g-no` red / `.g-partial` amber); (2) `drawHallucination` fills `#hallucinationPanel` with the verdict chip, the risk / grounded-fraction / context-divergence bars, the uncertain flag, a confidence legend, and provenance (grounded source chunks, plus nearest training passages labeled as similarity, not proof). Backend contract and detection logic live in [../backend/hallucination_detector.md](../backend/hallucination_detector.md).

## Dependencies

- `/generate` route at [backends_routes.py](../../../veritate_mri/routes/backends_routes.py).
- PyTorch inference brain at [veritate_mri/inference/backends/pytorch.py](../../../veritate_mri/inference/backends/pytorch.py) — see [../backend/inference_brain.md](../backend/inference_brain.md).
- `/meta` for the current model's layers and hidden size (drives canvas dimensions).
- Canvas helpers in [canvas_rendering.md](canvas_rendering.md).

## Pitfalls

- Frames buffer can grow large for long generations; the existing UI caps it but check before extending.
- Mode switching changes which decode strategy the backend uses. The mode value is part of the `/generate` payload.
- Chat history persists in localStorage ([state_persistence.md](state_persistence.md)) but frames do not — switching tabs and back drops the visualization.
