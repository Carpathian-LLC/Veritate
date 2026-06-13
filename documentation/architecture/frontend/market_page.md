# market.html: Market LLM dashboard (experimental)

Self-contained, vanilla-JS + canvas decision-support page served at `/market`. No build
step, no external chart library, no framework: it only calls `/market/*` (and `/settings`,
`/sys_metrics`) on the canonical server. Gated behind the experimental settings toggle (a
⚡ Market LLM link appears in the main dashboard nav when `experimental` is on).

## Layout
- **HUD + back link**: a `← Veritate` brand link (to `/app`) sits at the head, so the
  standalone page returns to the main platform. The platform telemetry HUD strip is
  reproduced at the top of the page (CPU / MEM / temps / per-GPU bars); `initHud()` reads
  `/settings` and shows it only when `hud_enabled`, polling `/sys_metrics` every second,
  honoring `hud_detailed` and `temperature_unit`. The HUD markup/CSS/render mirror the
  dashboard's so the experimental page looks and feels like the platform.
- **Header controls**: source (crypto/stocks), instrument (datalist), model (`<base>_h<h>`),
  **Forecast →** (static, from local data tail) and **● LIVE** (polls Binance.US every 12s).
- **Where it thinks price is going**: candlestick of recent bars + a forward probability
  **cone** (50 / 80 / 95% fans) widened by the volatility forecast, with a faint center
  line for the directional tilt. The "now" divider separates history from forecast.
- **The call**: plain-language decision cells — Lean (P up), Confidence, Expected move (±bps),
  Regime badge, Edge-after-fees (EV bps), Suggested size (¼-Kelly), last price, model acc.
  The note states honestly whether the directional EV is positive (rare) or negative.
- **Expected move size (volatility)**: predicted-vs-actual realized vol from the backtest
  replay — the panel that shows the model's *real* skill (vol R²).
- **Calibration**: reliability dots (predicted prob vs observed frequency) with a diagonal;
  below the line = overconfident.
- **Would the calls have made money?**: cost-aware equity curve (bps), usually flat/negative
  at short horizons after fees — shown honestly.
- **Honesty banner**: a persistent plain-language statement of what the model can and can't do.

## Data flow
`loadStatus()` populates the model dropdown from `/market/status`; `loadInstruments()` fills
the datalist. **Forecast** → `/market/forecast` (+`/market/backtest` for the lower panels).
**LIVE** → `setLive()` toggles a 12s `pollLive()` loop hitting `/market/live`, appending the
forming bar and redrawing the cone in real time; a status strip under the header shows the
live price, lean, expected move, regime, and last-update time.

## Canvas helpers
`drawCone` (candles + nested cone fans + tilt + now-divider), `drawVol` (pred vs actual
lines), `drawCal` (reliability scatter on a unit square), `drawEquity` (cost-aware equity,
green/red by sign). All use device-pixel-ratio scaling via `dpr()`.

## Honest-UI choices (from the research)
Never shows a bare point forecast; the cone foregrounds uncertainty. Calibration is a
first-class panel so "65% up" can be checked against reality. The equity curve is shown
even when it loses money. Plain-language labels throughout (no ML jargon on the face of it).
