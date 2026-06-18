# paper trading page

`extensions/canonical/paper_trade/page/index.html` is a self-contained page (own
HTML/CSS/JS, no build step, namespaced `pt-*`) for paper-trading the Market LLM's
forecasts with no capital. It is a **page-only canonical extension** (manifest + page, no
`register.py`, no `server/`): it gets forecasts + policy results from the Market LLM
extension over HTTP and holds its paper ledger in the browser. Mounted at
`/ext/paper_trade`; nav label "Paper Trading"; experimental.

## layout

Top bar = **what to trade** (mode, model, market, resolution, symbol, bars) + Run. A
dedicated **Trading rules** panel holds the policy controls (strategy, starting capital,
fee, confidence gate, move gate, sizing, max size); each rule carries a `?` badge
(`.pt-help[data-tip]`) that shows a plain-language explanation on hover, pure CSS, no JS.
Results read top-down: **Account** (dollar P&L), **Price and trades**, **Trades**, and
the **Live** panel in Live mode.

P&L is shown in **dollars** against a starting capital the user sets: the server returns
`equity`/`pnl_bps` in bps (capital-agnostic), the page applies capital
(`dollars = capital * (1 + bps/1e4)`) client-side, so changing capital re-renders the
account instantly without a refetch.

## what it is

**The forecast is fetched once; the rules run in the browser.** The model scoring each bar
is the only expensive step and the only thing the rules do not change, so Run fetches
`GET /market/paper_signal` (the raw per-bar forecast, cached) and the page runs the policy
client-side. Tweaking any rule, dragging the aggressiveness slider, or optimizing reshapes
the result instantly with no re-score. Only a **data change** (model/symbol/resolution/bars)
re-fetches; `markStale` prompts "press Run" when one changes.

- `simulate(sig, rules)` (the in-browser policy) **mirrors `server/policy.py`**
  (`backtest` + `_metrics` + `trades`): vol-harvest premium = trailing mean `|ret|` over
  `PREM_WINDOW=96`, the same gates/sizing/fee. `policy.py` stays the canonical scorer and
  the live trader; keep the two in sync (no automated check, no JS runtime here).
- **Aggressiveness slider** (`pt-aggr`) drives the move gate (`AGGR_HI=2.0` at "only the
  best setups" to `AGGR_LO=0.6` at "trade often"), two-way bound to the move-gate input;
  recomputes on input.
- **Optimizer** (`pt-optimize`) grid-searches move gate x conf gate over the loaded window,
  picks the best by the chosen objective (profit / Sharpe / per-trade), ignores settings
  with `< MIN_TRADES` (20), sets the rules, and labels the result **in-sample**.

Three modes: **Historical** (recompute + render), **Replay** (animate the cached sim bar by
bar at `pt-speed`), **Live** (poll `GET /market/paper_decide` every 15s; ledger in
`localStorage` `pt:led:<model|source|symbol|strategy>`, each new closed bar resolves the
prior open decision against the realized move). Live rule changes apply on the next poll.

## how it works

- Charts are hand-rolled canvas (`drawEquity`, `drawPrice`, `gridY`, `poly`) with a `dpr`
  helper, same approach as the Market LLM page; no chart library.
- `simulate` returns `equity`/`pnl_bps`/`max_dd` in **bps**; the page converts to dollars
  (`dollars = capital * (1 + bps/1e4)`), so changing capital re-renders instantly with no
  refetch. Live equity is the running sum of resolved `pnl_bps`, also in dollars.
- The signal is cached in `localStorage` (`pt:sig:<data-url>`, keyed by data only, not
  rules) so re-running the same data is instant; the quota fallback purges `pt:sig:`.
  Controls persist (`pt:ctl`); the live ledger persists per key and survives reloads.

## dependencies

- The Market LLM extension (`extensions/canonical/market/`) must be installed: this page
  calls `/market/veritate_models`, `/market/instruments`, `/market/paper_signal`,
  `/market/paper_decide`. If those 404 the page shows a "needs the Market LLM extension"
  notice and disables Run.
- Registered via `extensions/catalog.json` + `manifest.json`; activates on next server
  start (`extensions/registry.py`).

## pitfalls

- **Paper only.** No real orders, no keys, no capital. Real-money trading is a separate
  external service that calls the same API; do not add order execution here.
- **Live cadence vs bar size.** Live uses the 1m live feed and resolves on bar close;
  picking a coarse resolution in Live mode still resolves per 1m close. Historical/Replay
  honor the Resolution control via `base`.
- **Restart to activate.** Editing the page while the server runs has no effect until the
  next start (extension lifecycle).
