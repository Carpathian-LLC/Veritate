# Warm models panel

## What it is

A Settings-tab panel that picks which exported models stay permanently loaded as resident Veritate Engine (C) subprocesses (the warm pool), so switching to them in chat is instant. Backend: [../backend/warm_models.md](../backend/warm_models.md).

## How it works

Markup in [index.html](../../../veritate_mri/web/index.html): a **Warm models** `settings-row` directly after the **Engine** row, holding an **Always-loaded models** card with a list container `#warmModelsList` and a summary line `#warmModelsSummary`.

`_renderWarmModels(memAvail)` in [index.js](../../../veritate_mri/web/index.js) fetches `GET /backends` and renders one checkbox per model in `d.c.warm` (models with a servable `.bin`), each showing its `.bin` size and, when live, a green `RESIDENT` badge and, for the selected engine, an `ACTIVE` badge. It sums the checked models' sizes and shows `Selected: <sum> of <available> available`, turning warm-colored when the sum exceeds available RAM (soft warning, no hard block). Available RAM is the `sys_mem_available` field of the `/sys_metrics` snapshot the Settings tab already polls.

The whole list is data-driven — no model names are hardcoded; it scales from `/backends`. Rendering is gated by a signature (`_warmSig`) so the 1s Settings poll only rebuilds when the servable set, residency, pins, or available RAM changed, which avoids checkbox thrash.

Toggling a checkbox (`_onWarmToggle`) collects the checked names and `POST`s `{warm_models: [...]}` via `_saveSettings`; the backend `warm_apply` spawns/drops subprocesses to match, and the next poll repaints the residency badges.

## Dependencies

- `GET /backends` — the `c.warm` status array (`_backends_status_payload` / `_warm_status`).
- `GET /sys_metrics` — `sys_mem_available` for the RAM summary.
- `POST /settings` — persists `warm_models` and applies the pool change.

## Pitfalls

- `RESIDENT` reflects a live subprocess right now; it may lag a spawn by up to the 1s poll interval after a toggle or a fresh server start.
- The RAM figure is available (free) memory, not installed; a high baseline load lowers it. The warning is advisory — the box still lets the operator over-subscribe.
- Rebuilding the engine closes the pool; badges clear until the models re-warm on the next server restart.
