# market_routes.py — experimental market dashboard API

Backend for the `/market` Market LLM page. Registered in `app.py`
(`market_routes.register(app)`); the page itself is served at `/market`
(`market_page()` → `web/market.html`). Isolated from canonical pipelines: it only
imports the `market/` package (lazily, so a missing sklearn never breaks startup) and
reads `external_data/` + `models/market/`.

All handlers wrap their body in `_safe("market", fn)` for the JSON-error contract.

## Endpoints

- **GET `/market/status`** → `{models, summary, instruments:{crypto,stocks}, cone_levels}`.
  `models` = available `<base>_h<h>` bundles; `summary` = `models/market/summary.json`
  (per-model val metrics). Drives the dashboard's model dropdown + honesty report.

- **GET `/market/instruments?source=crypto`** → `{instruments:[...]}` from
  `external_data/<source>/*.csv`.

- **GET `/market/forecast?symbol&base&horizon`** → forecast from the **local** data tail
  (`data.load_tail`, reads only the file's end). Returns: `last_close`, `vol_fwd[_bps]`,
  `p_up`, `confidence`, `regime`, `candles` (last 120), `cone` (per-level price bands over
  H steps), `decision` (EV + ¼-Kelly), and the model's val `metrics`.

- **GET `/market/live?symbol&base&horizon`** → same shape as `/forecast` but from **live**
  Binance.US REST klines (`market.live`), forecasting from the last CLOSED bar, plus the
  in-progress `forming` bar. 502 if the symbol isn't on Binance.US.

- **GET `/market/backtest?symbol&base&horizon&n`** → per-instrument honest replay
  (`market.backtest.replay`): `summary` (vol R², dir acc vs base, net equity, Sharpe),
  `vol_series` (pred vs actual), `reliability` (calibration bins), `equity` (cost-aware,
  bps), `price_series`.

## Constants
`BASE_SEC` (timeframe→seconds), `CONE_LEVELS=(0.5,0.8,0.95)`, `DEFAULT_COST_BPS=10`
(×2 = 20 bps round-trip), `CANDLES_OUT=120`. `_kelly()` computes symmetric-payoff
¼-Kelly + EV after cost.

## Notes
- The cone is built in `models.MarketModel.cone`: width from the forecast volatility
  (the strong signal), a slight drift from `p_up`, scaled by the split-conformal `cone_k`
  so the nominal band ≈ its stated coverage.
- No API key is needed anywhere — Binance.US market data is public. api.binance.com is
  geo-blocked (451) in the US; only api.binance.us is called.
