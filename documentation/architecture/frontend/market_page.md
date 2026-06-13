# market.html: Market LLM dashboard (experimental)

Vanilla-JS + canvas comparison ground served at `/market`. No build step, no chart library,
no framework: it only calls `/market/*` on the canonical server. The `/market` route is
served unconditionally; the experimental settings toggle only reveals the Market LLM link in
the main dashboard nav (`#navMarket`, hidden by default). The page is reachable regardless of
the toggle. 100 percent ASCII copy (no emojis, no emdashes).

The page is driven by the **Veritate byte-level model** (the on-mission engine), not the GBDT
baseline. The whole point is to test whether a model trained on JUST market data (no news, no
labels) can predict price. It folds in the former standalone `/predict` benchmark, so the
predict page and its routes no longer exist.

Visual language matches the canonical dashboard exactly: monospace type, the `--bg/--panel/
--line/--accent` token set, 5px panels, `--accent` primary buttons, `--data-pos` green and
`--hot` red for semantic data. No emojis, no emdashes.

## Controls and feedback
A sticky controls row: market (crypto/stocks), symbol (a real `<select>` populated per source
so a crypto symbol can never be requested under stocks), model, trade-size dollars, **Run**,
**Live**. The controls are reactive: changing market, symbol, model, or dollars re-renders
(instantly from cache when that request has been seen before). The Run button shows a spinner
and a "Running" state while in flight, and a status bar under the controls reports what is
happening (busy / ok / error, color-coded). Every chart box has a shimmer skeleton while
loading and an empty-state message when there is no real data, so a chart never renders blank
or fabricated. Performance figures (hit rate, precision, P/L, magnitude) are colored green when
the model beats its baseline and red when it is below; descriptive stats stay neutral.

## Caching
The hindcast and benchmark endpoints run the byte model on the server, so their JSON is cached
in `localStorage` keyed by request URL (`cf(url, force)`; `mktllm:` prefix, quota-safe with a
namespace purge + retry). A page reload renders from cache and never re-runs the model; the
last controls (`saveControls()`) are restored on load so the page reopens on the same symbol and
model. **Run** is the only forced refresh (`run(true)`) and updates the cache; every other
trigger uses `run(false)`. When results come from cache the status bar says so and invites a Run
to refresh. `loadModels`/`loadInstruments` stay live (cheap, and so newly trained models appear).

## Layout (top to bottom)
- **Market vs model**: the actual price as a white line, plus one colored predicted-path line
  per trained model (each model's cumulative path if it followed its own minute-by-minute
  directional calls). A legend names each model, marks the selected one, and shows its hit rate.
  Models that track the white line called direction well; ones that drift did not. **Expand** opens
  a large modal of the same chart.
- **If the model traded this for you**: a custom-dollar trading sim. Cards for starting / ending /
  profit-loss / direction-right, plus a dollar equity curve after `FEE` (10 bps per side). Expandable.
- **How good were its calls**: the scored metric card grid (precision when decisive, precision at
  high confidence, avg confidence, up/down precision, decisive rate, directional accuracy,
  magnitude correlation, magnitude error vs persistence, bars scored, final edge, move mix).
- **Model analytics**: a single chart viewer driven by a dropdown (no long bottom-stack, no
  scroll-jumping). Views: edge equity curve, cumulative directional accuracy, confidence
  calibration, predicted vs actual move, magnitude scatter, right-or-wrong strip, and **Data
  inventory** (the per-instrument table, folded in from the old "Inspect data" button). Expandable.
- **Best and worst calls**: the model's most confident right and wrong calls (ranked by
  confidence x size of the move), from `benchmark.best`/`worst`. Cheap: derived from the single
  full-series pass the benchmark already runs.
- **Multiple instruments**: the same model across the majors; click a box to load it above
  (loads in place, no forced scroll).
- **Live**: streams the live market and the model's next-minute call (every 12s). Expandable.
- **Honesty banner**: persistent plain-language statement of what the model can and cannot do.

## Data flow
`loadModels()` fills the model dropdown from `/market/veritate_models`; `loadInstruments()`
fills the instrument select per source. **Run** fans out `/market/veritate_hindcast` for every
trained model (for the overlay legend) and `/market/veritate_benchmark` for the selected model
(metrics, analytics, best/worst), guarded by a `runTok` so stale responses from fast dropdown
changes are ignored. `loadMulti()` runs `/market/veritate_hindcast` across the majors for the
grid. The **Data inventory** view lazy-fetches `/market/veritate_data_report`. **Live** toggles
a 12s `pollLive()` loop on `/market/veritate_live`, appending the latest close and redrawing the
next-minute ray.

## Bucket space
The benchmark works in return-bucket space (the codec's `RET_BINS=33`, center `RET_CENTER=16`).
`pred`/`actual` are bucket indices read from the per-run `ret_center`/`ret_bins` so the chart
drawers stay model-agnostic. Direction is `sign(bucket - center)`.

## Canvas helpers
`dpr(cv, hOver)` (device-pixel-ratio scaling via the `CH` height map, with an optional height
override so the modal can redraw the same chart large), `cwState()` (loading / ready / empty
per chart box). Drawers all take a target canvas and return true only when they drew real data
(false flips the box to its empty state): `drawOverlay` (+`predPath`), `drawDollar`, `lineChart`
(used by `drawEdge`/`drawAcc`), `drawCalib`, `drawPva`, `drawScatter`, `drawStrip`, `drawLive`.
`renderChart(kind, cv, hOver)` routes a view kind to its drawer; the analytics dropdown and the
expand modal both call it, so one chart definition serves the small view and the large modal.

## Honest-UI choices (from the research)
Direction at the minute scale is near a coin flip for any model, so the page foregrounds the
signals that are real: how well the model sizes moves (magnitude correlation, magnitude error
beating persistence) and whether its confidence is calibrated. The dollar equity curve is shown
even when it loses money. A chart only renders when its data array is present; otherwise the box
shows an empty state, never a blank or fabricated chart. Data inventory reports approximate bar
counts (from file size), labeled as approximate.
