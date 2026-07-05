# trading page (/ext/trading)

`extensions/canonical/trading/page/index.html` is the Trading extension's single
self-contained page (own HTML/CSS/JS, no build step, namespaced `t-*`). Five tabs behind
one nav bar (`#t-nav`, active tab persisted in `localStorage` `trading:tab`):
Overview | Strategies | Market Intel | Research | Settings. Simulated capital everywhere;
the top badge and per-strategy section lines say so on every money number.

## tabs

- **Overview.** A "What is this?" card (three plain sentences); the system-status strip
  (`pollSystem` -> `GET /ext/trading/system`: one row per runnable with a green/grey dot,
  started-when, per-item Stop -> `POST /ext/trading/system/stop`, and a confirm-gated
  "Stop everything" -> `POST .../system/stop_all`); active-strategy portfolio cards
  (`pollOverview` aggregates `/ext/trading/paper/{account,xsmom/accounts,eqmom/accounts,
  ml7/accounts,exp/accounts}`: equity, P&L, vs-buy-and-hold, `drawSpark` sparkline); the three latest
  pump alerts (`/ext/trading/intel/pumps?n=3`, risk badges); and the static "Can I trust
  these numbers?" table (one honest validation line per strategy).
- **Strategies.** Four cards: news sentiment, weekly crypto momentum (xsmom), the ML
  weekly book (ml7), monthly equity momentum (eqmom). Each carries a "How it was
  validated" warn panel with the real research numbers, a two-sentence what-it-does note,
  Start/Stop controls with the strategy's knobs, and performance (equity vs benchmark
  chart via `drawNTEquity` / `drawBookOverlay`, positions, recent trades; xsmom/eqmom/ml7
  share the parameterized `BOOKS` / `pollBook` render path; ml7's status line surfaces the
  frozen-model state from `GET /ext/trading/paper/ml7/status`). Section header lines carry
  live equity / vs-hold summaries (`#t-sum-news`, `#t-sum-xs`, `#t-sum-ml7`, `#t-sum-eqm`).
  The news card's ADVANCED sub-section
  (`#t-adv-toggle`, hidden by default) holds the six insight charts (`drawInsights`), the
  N-arm experiment engine, and the "News it's reading" feed. Backend contracts:
  [../backend/paper_trade_sentiment.md](../backend/paper_trade_sentiment.md),
  [../backend/paper_trade_xsmom.md](../backend/paper_trade_xsmom.md),
  [../backend/paper_trade_eqmom.md](../backend/paper_trade_eqmom.md).
- **Market Intel.** The surveillance radar: watch controls (brief model, cadence,
  start/stop watch, scan now), channel-health chips, the trending board (anomaly-scored,
  local-model "why it's moving" cells), the pump radar cards behind the permanent
  exit-liquidity warning, and the event log. Calls `/ext/trading/intel/*`
  ([../backend/market_intel.md](../backend/market_intel.md)).
- **Research.** "Test the chart model on past data": the byte-model backtest cockpit.
  Run fetches `GET /ext/trading/market/paper_signal` once (cached in `localStorage`
  `trading:sig:<data-url>`); the trading rules then run in the browser (`simulate` mirrors
  `server/policy.py`: same gates/sizing/fee, `PREM_WINDOW=96`; keep the two in sync), so
  rule tweaks, the aggressiveness slider (`AGGR_HI=2.0`..`AGGR_LO=0.6` -> move gate), and
  the in-sample optimizer reshape instantly with no re-score. Three modes: Historical,
  Replay (animates the cached sim), Live (polls `GET /ext/trading/market/paper_decide`
  every 15s, ledger in `localStorage`). P&L renders in dollars against a user-set capital
  (`dollars = capital * (1 + bps/1e4)`), so capital changes re-render without a refetch.
  A card links to the model-analytics page `/ext/trading/models`
  ([trading_models_page.md](trading_models_page.md)).
- **Settings.** Server-side defaults (`GET/POST /ext/trading/settings`): default LLM
  model (Ollama list via `POST /teacher/models`), news/intel scan intervals, the news-fee
  assumption, and the channel manager (type/value/enabled/health table + add row +
  delete; channel edits POST immediately). Every field carries a plain-language caption
  or `?` tooltip.

## how it works

- Charts are hand-rolled canvas (`drawEquity`, `drawPrice`, `drawNTEquity`, `gridY`,
  `poly`, `dpr`); no chart library, no CDN.
- Polling is unconditional (strategies keep trading when their tab is hidden); switching
  tabs re-runs the relevant pollers so canvases redraw at real widths.
- Panel collapse (`setupPanels`) applies to the Strategies + Research tabs only; primary
  panels start expanded, state persists per panel title.
- Controls persist in `localStorage` (`trading:ctl`, `trading:nt`, `trading:expsel2`).

## pitfalls

- **Paper only.** No real orders, no keys, no capital. Real-money trading is a separate
  external service that calls the same API; do not add order execution here.
- **Live cadence vs bar size.** Research Live mode resolves per 1m close whatever the
  Resolution control says; Historical/Replay honor it via `base`.
- **Restart to activate.** Page edits take effect on the next server start (extension
  lifecycle); the registry serves the file from disk but routes register at boot.
- The in-browser `simulate` and `server/policy.py` have no automated equivalence check;
  a policy change must touch both in the same commit.
