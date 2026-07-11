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

## Redraw on visibility / layout change

`canvas` is `width: 100%` with no CSS height, so its displayed height is aspect-ratio-derived and settles only after layout. A chart fit while its tab or panel is hidden (or before width settles) renders distorted until a reflow re-fits it. One mechanism fixes this: a single `ResizeObserver` (`_canvasResizeObserver`) observes every `<canvas>` and, on any box change, coalesces to one `requestAnimationFrame` and calls `redrawAllVisible()`, which re-fits and redraws the active tab's charts. The observer fires on the 0 -> N transition when a tab or panel becomes visible, on panel expand/collapse, and on window resize (the `resize` listener calls the same `redrawAllVisible`). `fitCanvas` preserves the display aspect ratio, so it never changes the CSS box and never re-triggers the observer (no feedback loop). The expand button only toggles the `.expanded` class; the observer does the re-fit, so there is no manual `fitCanvas` + `dispatchEvent("resize")` in the click handler.

## Dependencies

- DPR-aware canvases need `c.style.width` and `c.style.height` set by CSS layout. If the canvas has no CSS size, `fitCanvas` returns zero-sized.
- Drawers consume parsed data (frames, csv rows, palette tokens). Path parsing or fetch is done outside the drawer.

## Pitfalls

- `fitCanvas` returns early on hidden canvases (`offsetParent === null`); a drawer called while the tab is hidden produces nothing. The `_canvasResizeObserver` covers this: the redraw fires when the canvas becomes visible. New dynamically-created canvases are not auto-observed (the observe pass runs once at init over the static DOM).
- Don't draw cumulative state. Drawers should be idempotent given the same inputs so they can be called repeatedly during scrub.
- Lines drawn at 1px without `lineWidth = 1.5` and `setLineDash` get fuzzy on non-integer DPR (e.g., 2.25 on some 4K displays). Use the existing examples as templates.
