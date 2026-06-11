# Frontend components

The dashboard is vanilla JavaScript with no framework. Served by Flask from [veritate_mri/web/](../../../veritate_mri/web/). Dashboard entry points: `index.html`, `index.js`, `index.css`.

Page routing: `/` and `/chat` both serve the chat front door (`hybrid.html`), `/app` serves the dashboard (`index.html`). The dashboard and management APIs sit behind the optional password gate; the chat page stays public. See [auth.md](../backend/auth.md).

## Tabs

- [generation_tab.md](generation_tab.md) — chat with a loaded model
- [learning_tab.md](learning_tab.md) — inspect checkpoints across training
- [training_tab.md](training_tab.md) — monitor live + past training runs
- [wiki_tab.md](wiki_tab.md) — markdown project log
- [logs_tab.md](logs_tab.md) — engine status, system metrics, log ring
- [settings_tab.md](settings_tab.md) — dashboard preferences and trainer config

## Cross-cutting

- [tab_system.md](tab_system.md) — how tabs work; the `valid` allowlist; hashchange routing
- [data_flow.md](data_flow.md) — Flask routes the frontend polls + SSE feeds
- [standalone_modules.md](standalone_modules.md) — the IIFE pattern (prune, tutorial)
- [canvas_rendering.md](canvas_rendering.md) — fitCanvas, drawSeries, palette
- [state_persistence.md](state_persistence.md) — what's in localStorage, what comes from server

## Panels and overlays

- [hud.md](hud.md) — always-on system metrics overlay (CPU, MEM, GPU°)
- [prune_panel.md](prune_panel.md) — neuron pruning UI inside the Models tab
- [rag_train_panel.md](rag_train_panel.md): RAG corpus build + SFT panel in the Training tab
- [tutorial.md](tutorial.md) — onboarding walkthrough overlay

## Adding a new tab

1. Add a `<div class="tab" data-tab="X">` and a `<div class="tab-body" data-tab="X">` in [index.html](../../../veritate_mri/web/index.html).
2. Add `"X"` to the `valid` array in [index.js:2098](../../../veritate_mri/web/index.js#L2098).
3. If the tab needs init/cleanup, add an `else if (name === "X")` branch in `activateTab()` ([index.js:2102+](../../../veritate_mri/web/index.js#L2102)).
4. Add a file here describing the tab.
