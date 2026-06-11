# Canvas rendering

## What it is

All charts in the dashboard render to HTML5 canvas elements. Two shared helpers — `fitCanvas` and `drawSeries`/per-chart drawers — handle DPR scaling, axis layout, and color tokens.

## fitCanvas

Defined at [index.js:144–160](../../../veritate_mri/web/index.js#L144). Syncs a canvas's backing buffer to its CSS size times the device pixel ratio so lines render crisp on Retina and external displays.

- Caches the last DPR on `c.__fitDpr`. Only reallocates the buffer when CSS size or DPR changes (avoids GPU upload churn during hot redraws like slider scrub).
- Returns early when the canvas is detached (`offsetParent === null`) — important because hidden tabs would otherwise reflow on every poll.
- Calls `ctx.setTransform(dpr, 0, 0, dpr, 0, 0)` so all subsequent drawing uses CSS pixels.

## Drawing functions

Each chart has a named drawer. Examples:

- `drawFfn(canvas, ctx, ffnFull)` at [index.js:277–304](../../../veritate_mri/web/index.js#L277) — FFN layer × bucket heatmap.
- `drawTelemetry(canvas, ctx, frames, idx)` at [index.js:603–661](../../../veritate_mri/web/index.js#L603) — time-series traces (loss, lr, throughput).
- `drawQuantKl(canvas, ctx, checkpoints, idx)` at [index.js:382–471](../../../veritate_mri/web/index.js#L382) — quantization KL with checkpoint markers.

All drawers follow the same shape: fitCanvas, clear background, compute bounds, draw geometry, draw axes and labels.

## Color palette

`PALETTE` constant at [index.js:19–31](../../../veritate_mri/web/index.js#L19) defines semantic color roles:

- `cool` (blue), `warm` (orange), `hot` (red), `purple` — categorical
- `dataPos`, `highlight`, `accent`, `good` — semantic
- `dim`, `text`, `line` — structural

CSS custom properties in [index.css](../../../veritate_mri/web/index.css) mirror the same names so HTML and canvas share the same color story.

`regionRamp(t, layer)` at [index.js:173–192](../../../veritate_mri/web/index.js#L173) maps layer index to one of three regions (sensory/association/output) and returns an RGB string based on the normalized value `t`. Used in lens-style visualizations where layer-region matters.

## Dependencies

- DPR-aware canvases need `c.style.width` and `c.style.height` set by CSS layout. If the canvas has no CSS size, `fitCanvas` returns zero-sized.
- Drawers consume parsed data (frames, csv rows, palette tokens). Path parsing or fetch is done outside the drawer.

## Pitfalls

- `fitCanvas` returns early on hidden canvases; calling a drawer before the tab is active produces nothing visible. The pattern is to call drawers inside `requestAnimationFrame` from the tab's activation branch.
- Don't draw cumulative state. Drawers should be idempotent given the same inputs so they can be called repeatedly during scrub.
- Lines drawn at 1px without `lineWidth = 1.5` and `setLineDash` get fuzzy on non-integer DPR (e.g., 2.25 on some 4K displays). Use the existing examples as templates.
