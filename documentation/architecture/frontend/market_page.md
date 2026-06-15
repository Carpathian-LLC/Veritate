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
A sticky controls row grouped by purpose with a small uppercase caption (`.cgroup .cap`) and a
thin `.csep` divider between groups: **Data** (market, resolution, symbol), **Models** (compare
models), **Scoring** (window), then Run / Live. Grouping is presentation-only. Trade-size,
confidence sizing, and aggressiveness are NOT in this row: they are the Trading controls inside the
"If the model traded this for you" panel.

Controls: **market** (crypto | stocks; per-second / 1-second was dropped, no free 1s feed),
**resolution** (tooltip: bar size each step represents, per-minute to hourly for crypto, daily for
stocks, 1s unavailable), **symbol** (a typeahead text input bound to a `<datalist>`: type to search
the instrument set instead of scrolling a flat select; the list is server-ordered majors-first then
alphabetical), **compare models** (a multi-select checkbox picker), **window** (history length;
tooltip: how much history is scored, e.g. "1 month" scores the last month of calls). Trade-size
dollars, a **confidence-sizing** checkbox, and **aggressiveness** (a confidence-threshold slider)
live in the trading panel below.

### Primary (analytics) model = first checked
There is a single model picker. The old standalone "Analytics model" `<select>` was removed; the
**primary** (analytics) model is the FIRST checked model in the compare picker. `primaryModel()`
returns `SELECTED[0]||''`; everything that drives a single-model view (benchmark, live, the metric
cards, best/worst, the multi-instrument grid, the overlay shading + balance line) routes through it.
Directly below the picker a small `#primaryLbl` line reads `Primary (analytics) model: <name>`
(`updatePrimaryLabel()`, refreshed on `onPick`/`loadModels`) so the user always sees which model
drives the single-model charts. Charts that show only one thing use the primary; "Market vs model"
and the trading-equity panel render every checked model. The primary persists implicitly as
`SELECTED[0]` (the checked set is cached under `selectedModels`); `saveControls()` no longer stores a
separate `model` key.
A `spanBadge` chip shows the computed `bars x bar-size =
total span` for the current resolution/window so the evaluated duration is visible before running.
At fine resolutions a long window exceeds the bar-count clamp (`nBars()` caps at `NMAX=20000`),
so the badge reads `(capped)` and the run scores the most recent 20000 bars; coarser
resolutions reach the full window (e.g. 1 year at hourly = 8760 bars, uncapped).

### Source selector (crypto vs stocks)
`SRC` (one object per source) owns the per-source control sets; `srcCfg()` returns the active
one. **Crypto** streams 1m bars and live: resolution = per minute / 5m / 15m / hourly (all four are
populated by `loadControlsForSource()` from `SRC.crypto.res` whenever market = crypto; daily is
stocks-only), windows now span 15m / 1h / **4h** / 6h / 1d / 3d / 1w / 2w / 1mo / 3mo / 1yr so a
short "trade for a few hours / a day" run is one click, default symbol `BTCUSDT`, live enabled.
**Stocks** are daily bars only (no intraday, no live stream): resolution = Daily (`1d` is the single
option), windows are daily spans (**1w / 2w** / 1mo / 3mo / 6mo / 1yr / 2yr / 5yr expressed in `DAY`
units), default symbol `AAPL`, live disabled. The bar-count clamp `nBars()` still bounds every run to
`[NMIN=300, NMAX=20000]` bars: a window shorter than 300 bars at the chosen resolution evaluates the
most recent 300 bars and the `spanBadge` shows the real evaluated span, so the short windows are most
useful at coarser resolutions (e.g. 1 day at 1m = 1440 bars). `loadControlsForSource(keepRes,keepWin)` repopulates the resolution and window selects
from `srcCfg()` and is called on init and on every source change (it keeps the prior res/window
only if the new source still offers it, else falls back to that source's defaults). Switching
source also re-runs `loadInstruments()` (now defaulting the symbol via `srcCfg().defSym`),
`applyLive()`, and `updateSpanBadge()`, then `run(false)`. The byte model is scale-free and
instrument-agnostic, so the same crypto-trained checkpoints score stock daily bars with no
retrain; stocks are purely a data-plumbing + UI change. All control state (including `source`)
persists via `saveControls()` and restores on load.

The backend already threads `source` end to end: `/market/instruments`, `/market/veritate_hindcast`,
`/market/veritate_benchmark`, `/market/veritate_data_report`, and `/market/veritate_live` all take
a `source` arg and pass it to `data.list_instruments(source)` / `data.load_tail(..., source=...)`
(`market/data.py`). Stock CSVs live at `external_data/stocks/` (503 daily-OHLCV tickers, schema
`date,open,high,low,adjclose,volume`); the schema-flexible `load_1m()` reads the `date` string
column and the OHLCV subset directly.
The compare picker is the overlay set; the primary (first-checked) model is the one model that drives
the metrics, trading sim, best/worst, and live panels. There is no `model_type` filtering: the
picker lists every model with a checkpoint and the checked set decides what runs. The controls
are reactive: changing market, resolution, symbol, the checked set, window, dollars, the confidence
sizing toggle, or the aggressiveness slider re-renders (instantly from cache when that request has
been seen before). The Run button shows a spinner
and a "Running" state while in flight, and a status bar under the controls reports what is
happening (busy / ok / error, color-coded). Every chart box has a shimmer skeleton while
loading and an empty-state message when there is no real data, so a chart never renders blank
or fabricated. Performance figures (hit rate, precision, P/L, magnitude) are colored green when
the model beats its baseline and red when it is below; descriptive stats stay neutral.

## Caching
The hindcast and benchmark endpoints run the byte model on the server, so their JSON is cached
in `localStorage` keyed by request URL (`cf(url, force)`; `mktllm:` prefix, quota-safe with a
namespace purge + retry). A page reload renders from cache and never re-runs the model; the
last controls (`saveControls()`: source, resolution, window, symbol, dollars, aggressiveness
threshold, confidence-sizing toggle) are restored on load. The checked compare set is persisted
separately under `selectedModels` (`onPick` writes it; `loadModels` restores and filters to models
that still exist) so the overlay selection (and hence the primary = `SELECTED[0]`) survives a reload.
**Run** is the only forced refresh (`run(true)`) and updates the cache; every other
trigger uses `run(false)`. When results come from cache the status bar says so and invites a Run
to refresh. `loadModels`/`loadInstruments` stay live (cheap, and so newly trained models appear).

## Layout (top to bottom)
- **If the model traded this for you** (combined comparison panel): one panel holds two co-located
  views of the same calls, each under a `.subhead` sub-section with its own **Expand** button:
  "Market vs model" (the prediction overlay) then "Trading the same calls" (the dollar sim). They were
  separate panels; they are now grouped so the all-models prediction comparison and the all-models
  trading comparison read as one area.
  - **Market vs model** (sub-section): the actual price as a white line, plus one colored predicted-path
    line per **checked** model (each model's cumulative path if it followed its own bar-by-bar
    directional calls). A legend names each model, marks the **primary** one, and shows its hit rate.
    Models that track the white line called direction well; ones that drift did not. The same chart
    **also reflects the aggressiveness slider**: `drawTrades()` (called inside `drawOverlay`) shades
    the bars the **primary** model would trade at the current threshold (`TRADE_UP` green long /
    `TRADE_DN` red short fill spanning each traded bar), dots each new entry (on a side change), and
    draws the resulting **balance line** on its own right-side $ axis (`padR=64`) with min/max labels
    and an `EQ_REF` break-even baseline. The balance line is **trend-colored** by `drawEquityTrend()`:
    each segment is drawn green (`EQ_UP`) where the account is rising and red (`EQ_DN`) where it is
    falling, segment-by-segment by slope, so up vs down is obvious at a glance (replaces the former
    single gold `EQ_COLOR` line). The trade set + equity come from `simulate(prim, STATE.dollars,
    STATE.conf)` off the cached hindcast, so dragging the slider reshades the bars and redraws the
    balance line **live with no model re-run**. An always-visible `.chartkey` strip sits under the
    chart (white = actual price, green/red shade = primary long/short, balance line green rising / red
    falling). **Expand** opens a large modal of the same chart.
  - **Trading the same calls** (sub-section): a custom-dollar trading sim. A plain-language readout line
    (`pnlReadout`) states `Started $X -> ended $Y (+/-Z%) after 20 bps round-trip fees, N trades`, then
    cards for starting / ending / profit-loss / trades-taken, plus a dollar equity chart after `FEE`
    (`FEE=0.0010`, 10 bps per side). **Multi-model equity.** The dollar chart draws ONE equity curve per
    checked model (`equitySeries()`: runs `simulate()` over each cached hindcast in `STATE.results` at the
    current dollars + threshold, tags the primary model, sorts it last so it draws on top). `drawDollar`
    takes the series array, shares one $ axis across all curves; the **primary** line is trend-colored via
    `drawEquityTrend()` (green up / red down) and the other models stay dimmed to `EQ_DIM` in their overlay
    colors (`r.color` from `MCOLORS`). A `#dollarLegend` shows the green/red primary-trend key plus a
    color -> model swatch for each non-primary model. All client-side off cached hindcasts; no server
    field added.
  - **Aggressiveness** slider: a directional confidence threshold (`conf`, 0..0.9, default
    `DEF_CONF=0.20`): the client sim takes a position in a bar only when that bar's confidence
    `|p_up-0.5|*2 >= thr`, else it flattens to cash. Lower threshold = more trades = more fees
    (aggressive); higher = fewer, higher-conviction trades. **Trade cadence follows resolution**: each
    bar is one trade decision, so picking per-minute vs hourly changes how often it trades; the slider
    tooltip and `dollarNote` copy state this.
  - **Confidence-weighted sizing** (`#sizeConf` checkbox, "Bet bigger when more confident"): when ON,
    `simulate()` sizes each trade by that bar's confidence, scaling exposure from `SIZE_FLOOR` (weak
    calls) up to `SIZE_FULL` (surest calls) of the Trade size; when OFF, every trade uses the full Trade
    size (flat, as before). `side[]` now holds a signed exposure *fraction* in `[-1, 1]`; the fee is
    charged on the change in absolute exposure (`FEE * |want - pos|`), so a full flip pays the round-trip
    (2 x FEE) and a partial size-up pays proportionally. `STATE.sizing` (cached under `sizing`) is read
    by `simulate()` via a default param so every caller (`resim`, `equitySeries`, `drawTrades`,
    `loadMulti`) picks it up; toggling it re-sims + redraws live off cache (`$('sizeConf').onchange`).
    Copy is honest: on a near coin-flip 1m signal, bigger bets on louder calls mostly amplify the fee
    bleed, not the profit. The sim stays pure client JS over the cached hindcast `g.p_up`/`g.price`, so
    the slider and the toggle both re-sim instantly with no server re-run. Threshold + toggle persist via
    `saveControls()`. The note copy and trades-taken card keep the honest point visible: 1m direction is
    ~a coin flip (dir-acc ~0.51), so more trades pays more fees and loses faster.
  - **Slider plumbing.** The slider is a labeled `Trades often <-> Only its best bets` continuum
    (`.aggr` CSS: a `--warm -> --cool` gradient track, `--accent` thumb, end labels). It is now
    **full-width** (`flex:1 1 100%`, track `width:100%`) so it spans the trading panel on its own row
    below trade-size, and **granular** (`step="0.02"`, was 0.05). Readouts under it: a `#confLbl` chip
    + `#confStat` (trade-count and net-after-fee P/L) plus a `#confDesc` plain-language line that
    updates **live on every input tick** via `confDescribe(thr,sim)` (e.g. "Trading on almost every
    call: 36 trades, net +$17" vs "Acting only on its surest calls: 4 trades, net -$3"). Band
    thresholds are named constants `CONF_LO`/`CONF_HI`. All
    threshold-driven rendering is in **`resim()`** (overlay incl. trades+equity via `drawOverlay`, the
    dollar curve, the four sim cards, the readouts) which reads `STATE.primary`/`STATE.conf`/
    `STATE.dollars` only -- no fetch, no model re-run. `run()` calls `resim()` once after a fetch; the
    slider's `oninput` calls `resim()` directly on every drag tick for a live reshape; `onchange`
    persists via `saveControls()` and refreshes the multi-instrument grid P/L off cache. `simulate()`
    runs cheaply twice per drag (once for the cards, once inside `drawTrades`); recomputing a pure fn
    on a <=20k array beats threading the sim through `drawOverlay` (called by resize + modal without a
    sim in scope).
- **How good were its calls**: the scored metric card grid, led by a one-line "what this shows" note
  (a scorecard for the analytics model; green beats a coin flip / the naive baseline, red is worse) so
  a non-expert can read it. Model-quality cards (precision when
  decisive, precision at high confidence, avg confidence, up/down precision, decisive rate,
  directional accuracy, magnitude correlation, magnitude error vs persistence, bars scored, final
  edge, move mix) followed by trader-facing cards from `benchmark.trading` (win rate, profit
  factor, expectancy, Sharpe per trade, max drawdown, total return; gross / before fees) plus a
  **Net after fees** card from `trading.net_return` (total return minus `ROUND_TRIP_FEE` = 20 bps per
  trade, computed server-side in `veritate.py::_trade_metrics`). Rendered only when `trading` is present.
- **Model analytics**: a single chart viewer driven by a dropdown (no long bottom-stack, no
  scroll-jumping), led by a one-line "what this is" note (close-up views of the same scored calls,
  one at a time; the `viewNote` line under the chart says in plain words what the chosen view plots
  and what a good shape looks like). Views: edge equity curve, cumulative directional accuracy, confidence
  calibration, predicted vs actual move, magnitude scatter, right-or-wrong strip, and **Data
  inventory** (the per-instrument table, folded in from the old "Inspect data" button). Expandable.
- **Best and worst calls**: the model's most confident right and wrong calls (ranked by
  confidence x size of the move), from `benchmark.best`/`worst`. Cheap: derived from the single
  full-series pass the benchmark already runs.
- **Multiple instruments**: the analytics model across the majors at the current resolution; click
  a box to load it above (loads in place, no forced scroll). The major set is per-source
  (`MULTI_SET`): crypto majors for crypto, large-cap tickers for stocks.
- **Live**: streams the live market and the analytics model's next-bar call (every 12s). Live
  ignores the window/resolution controls (it always streams the latest 1m feed). Expandable.
  **Crypto only.** For stocks there is no live intraday feed, so `applyLive()` hides the Live
  button + chart and shows the panel with a one-line plain-language note (not an error); switching
  to stocks while live is running stops the poll.
- **Honesty banner**: persistent plain-language statement of what the model can and cannot do.

## Data flow
`loadModels()` fills the single compare picker from `/market/veritate_models` (no slice, no type
filter) and sets the `#primaryLbl` line; `loadInstruments()` fills the symbol typeahead's `<datalist>`
options from `/market/instruments` (local cached symbols unioned with the fetchable crypto
majors, ordered majors-first then alphabetical by `data.list_instruments`, so a fresh install
is populated and the dropdown leads with high-volume pairs). The symbol input uppercases and
trims a typed value on change before running. Picking a
symbol with no cached data triggers a one-time on-demand Binance backfill in the data layer
(`market/fetch.py`), shown as the normal "Scoring..." spinner; it is cached after. Every
request carries `base` (resolution) and `n` (`nBars()` = window-seconds / resolution-seconds,
clamped to [300, 20000]). **Run** fans out `/market/veritate_hindcast` for each **checked** model
(the overlay) and `/market/veritate_benchmark` for the primary (first-checked) model (metrics, trading,
analytics, best/worst), guarded by a `runTok` so stale responses from fast control changes are
ignored. With nothing checked the page shows empty states (`clearOutputs()`). `loadMulti()` runs
`/market/veritate_hindcast` across the majors for the grid. The **Data inventory** view
lazy-fetches `/market/veritate_data_report`. **Live** toggles a 12s `pollLive()` loop on
`/market/veritate_live`, appending the latest close and redrawing the next-bar ray.

## Bucket space
The benchmark works in return-bucket space (the codec's `RET_BINS=33`, center `RET_CENTER=16`).
`pred`/`actual` are bucket indices read from the per-run `ret_center`/`ret_bins` so the chart
drawers stay model-agnostic. Direction is `sign(bucket - center)`.

## Canvas helpers
`dpr(cv, hOver)` (device-pixel-ratio scaling via the `CH` height map, with an optional height
override so the modal can redraw the same chart large), `cwState()` (loading / ready / empty
per chart box). Drawers all take a target canvas and return true only when they drew real data
(false flips the box to its empty state): `drawOverlay` (+`predPath`, +`drawTrades`), `drawDollar`
(takes a series array of `{eq,color,analytics}` from `equitySeries()`, one line per checked model;
the primary line is trend-colored, the rest dimmed), `drawEquityTrend(x,eq,X,Y,wd)` (shared helper:
draws a balance polyline segment-by-segment, green `EQ_UP` rising / red `EQ_DN` falling, used by both
`drawTrades` and `drawDollar` for the primary line), `lineChart` (used by `drawEdge`/`drawAcc`),
`drawCalib`, `drawPva`, `drawScatter`, `drawStrip`, `drawLive`.
`drawTrades(x,prim,X,Y,w,h,padL,padR)` is the slider's chart effect: it reuses the overlay's price
`X`/`Y` mapping to shade traded bars + dot entries (on side change), and adds a second right-side $
axis for the trend-colored balance line. `drawOverlay` reserves a `padR=64` right gutter for that axis.
`renderChart(kind, cv, hOver)` routes a view kind to its drawer; the analytics dropdown and the
expand modal both call it, so one chart definition serves the small view and the large modal.

## Honest-UI choices (from the research)
Direction at short time scales is near a coin flip for any model, so the page foregrounds the
signals that are real: how well the model sizes moves (magnitude correlation, magnitude error
beating persistence) and whether its confidence is calibrated. The dollar equity curve is shown
even when it loses money. A chart only renders when its data array is present; otherwise the box
shows an empty state, never a blank or fabricated chart. Data inventory reports approximate bar
counts (from file size), labeled as approximate.
