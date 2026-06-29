# paper_trade sentiment layer + /teacher/complete

The Paper Trading extension is now a standalone extension with its own server
(`extensions/canonical/paper_trade/server/`): a free news scraper, an LLM sentiment
scorer that calls a user-added model, and the routes that serve them. It reuses the
Market LLM extension's `/market/*` API for the chart-model forecast; this server owns the
sentiment side only.

## /teacher/complete (platform endpoint)

`POST /teacher/complete` (`veritate_mri/routes/teacher_routes.py`) is a one-shot call to a
user-added model. Body: `{prompt, system?, provider?, model?, base_url?, api_key?,
max_tokens?, temperature?}`; defaults to the configured teacher (settings + `_stored_key`).
Returns `{ok, text, provider, model}` or `{ok:false, error}` (502 on provider failure).
Thin wrapper over `teacher.complete(provider, model, [{role:user,content:prompt}], **opts)`.
This is the programmatic surface an extension uses to score text with the user's model
(Ollama, or any configured provider) without importing platform internals.

## scraper.py

Free crypto headlines + sentiment context, stdlib + certifi only (no key, no feedparser):
- `scrape(limit, focus=None)` — recent headlines across `NEWS_FEEDS` (CoinTelegraph, Decrypt,
  CryptoSlate, Bitcoin Magazine RSS), deduped by title, newest first. When `focus` is a ticker
  (e.g. `SOL`), it ALSO pulls a token-specific Google News search RSS (`gnews()` via
  `TOKEN_NAMES`) and ranks headlines mentioning that coin to the top — so choosing a coin in the
  UI automatically focuses the news the model reads. One unreachable feed is skipped, never fatal.
- `fear_greed()` — current alternative.me fear-greed index `{value 0..100, label}` or None.
- `fetch_rss`, `_ts` (RFC-822 pubDate -> unix, fallback to now).

## sentiment.py

Scores headlines via `/teacher/complete` (HTTP, no internal imports) and aggregates:
- `score_items(items, provider, model, url, cache=None)` — scores each headline (`SENT_SYSTEM`
  asks for compact JSON `{asset, sentiment -1..1, confidence 0..1}` and to score the *surprise* /
  not-already-priced-in impact, not raw tone), attaches the parsed fields; unscorable items are
  dropped. Pass a `cache` dict (keyed by title) to memoize scores so a re-scan only pays model
  latency for NEW headlines — this is what makes a slower, stronger reader (`qwen2.5:72b`, far
  better at judging "already priced in") affordable in steady state. The measured +0.48 GDELT
  probe used `qwen2.5:7b-instruct`; that is the floor, the 72b is the upgrade path. Defaults to
  the configured teacher.
- `_parse` — extracts the `{...}` object from model text (tolerates surrounding prose),
  clamps ranges.
- `aggregate(scored, half_life_s, now)` — per-asset time-decayed signal `{asset:{score,n}}`,
  `score in [-1,1]`, weight = `confidence * 0.5**(age/half_life)` (fresh confident calls
  dominate). Default half-life 6h.

## news_trader.py (the autonomous "script")

Standalone forward-running paper trader (CLI, like `recorder.py`). Each tick: `scraper.scrape`
-> `sentiment.score_items` + `aggregate` -> `targets()` (per-asset long exposure; only coins
in the `TRADABLE` universe above `--gate`, so the LLM tagging COINBASE/ALIBABA never becomes a
position) -> `rebalance()` a SIMULATED JSON ledger marked to live Binance.US prices, round-trip
spread `--fee_bps` -> persist (per-asset sentiment + price each tick, for the insight charts).
**Trading is event-driven, not on a clock:** `--interval` only sets how often it SCANS news
(default 300s); `rebalance` trades a position only when its target exposure shifts more than
`--band` (the fee-aware deadband, default 0.12 of equity). So it trades heavily during breaking
news and not at all when quiet, and never churns small drift into fees (the cost-aware filter
that makes the signal tradable). FAKE money only (a JSON account file, no
broker, no keys). Sentiment supplies DIRECTION (the chart model is a coin flip there); the
optional `--use_chart` gate vetoes entries the byte model sees no tradable move in (via
`/market/paper_decide`). `--once` runs a single tick and prints the result; default loops
every `--interval` seconds, accumulating a forward (out-of-sample) track record — the only
honest validation. Ledger at `extensions/installed/paper_trade/data/account.json`.

Run: `python extensions/canonical/paper_trade/server/news_trader.py --model qwen2.5:7b-instruct`
(needs the dashboard up for `/teacher/complete` scoring; `--use_chart` also needs the Market
LLM extension).

## routes (register.py, under /ext/paper_trade)

- `GET /ext/paper_trade/sentiment?n&model&provider` — scrape -> score -> aggregate; returns
  `{ok, fear_greed, signal, scored, n}`. Scoring n headlines costs ~n x model latency. The
  page's "News it's reading" panel calls this (shows source, headline, asset, sentiment).
- `GET /ext/paper_trade/feed?n` — scrape only (fast), `{ok, fear_greed, items}`.
- `GET /ext/paper_trade/account` — the live paper ledger: `{ok, running, equity, cash,
  start_cash, positions, signal, curve, bench, series, recent}`. `bench` is the BTC buy-hold
  benchmark aligned to `curve` (start_cash marked to BTC from tick 0) — the page overlays it as
  a dashed line so the only honest question ("are we beating just holding BTC?") is always on
  screen. Each tick records `btc` price for this. Polled every 30s.
- `POST /ext/paper_trade/trader/start` — start the news trader as a managed in-process
  thread (`news_trader.start_thread`); body `{model, provider, gate, interval, fee_bps,
  source, use_chart, focus}`, defaults to `sentiment.DEFAULT_MODEL`. `POST .../trader/stop` stops
  it; `GET .../trader/status` -> `{running, model, interval, focus, ...}`. The page's Start/End
  session buttons + status poll drive these. This is the "main" run (label `main`,
  `account.json`).

### multi-run + A/B (BTC vs DOGE)

`news_trader` supports concurrent named runs: `_RUNS` is a registry keyed by label, and
`ledger_for(label)` maps each to its own ledger (`main` -> `account.json`, else
`account_<label>.json`), so runs never share a ledger. `start_thread(..., label=)`,
`stop_thread(label=)`, `status(label=)`, `status_all()` operate per-label; `load_ledger(path)` /
`save_ledger(led, path)` / `tick(..., ledger_path=)` take an explicit path. The CLI gained
`--label`. Ledgers persist on disk, so a run resumes its equity history across restarts (stateful).

**Live config update (no restart):** each run's tick reads its config from a shared mutable
`_RUNS[label]["cfg"]` dict, so `update_run(label, model=/gate=/band=/interval=/focus=...)` changes
take effect on the next tick with the ledger untouched. Routes: `POST .../trader/update` (main),
`POST .../ab/update` (both arms). The page applies model/sensitivity/scan-rate/token edits live
while a run is active instead of forcing stop/start.

**Holdings:** `_holdings(led, eq, last_prices)` marks each position to the last tick price ->
`{asset, qty, price, value, weight}`; `/account` and `/ab/accounts` return `holdings` +
`cash_weight` (and the arms return `recent` trades) so the page shows how the capital is allocated
and spent, not just a curve.

The legacy 2-arm A/B (`/ab/*`, fixed BTC vs DOGE) is superseded by the generic N-arm experiment.

### markets: crypto + stocks

The trader is market-aware (`tick(..., market=)`, default `crypto`). `price(symbol, market)` dispatches:
crypto -> Binance.US spot (`BTCUSDT`), stocks -> Yahoo Finance quote (`AAPL`). `universe(market)` is a
research-grounded **barbell**: `TRADABLE` = liquid crypto majors + a small meme tier (crypto breadth
is mostly illusory); `STOCK_UNIVERSE` = high-retail-attention / news-covered US names + SPY/QQQ.
`sym_for(asset, market)` builds the trade symbol. `market_open(market)` gates trading: crypto is
24/7, stocks trade only in the US regular session (09:30-16:00 ET, Mon-Fri) because the free Yahoo
quote freezes after hours, so a tick out of session marks positions but places no trades (otherwise
the arm pays spread on a frozen price = pure fee loss). `scraper.scrape(..., market=)` swaps the feed set
(crypto RSS vs `STOCK_FEEDS` general financial) and the Google-News qualifier (`cryptocurrency` vs
`stock`); the sentiment prompt is unified for crypto + equities. Each tick records `bench_px` (price of
the arm's benchmark asset: the focused ticker, or BTC for broad crypto / SPY for broad stocks) so the
view benchmarks any arm against its own buy-and-hold.

### generic N-arm experiment (`/exp/*`)

Each arm also carries a **strategy config**: `aggr` (Conservative/Balanced/Aggressive -> the `AGGR`
map sets gate/band/max_size: choosy+small+low-turnover vs acts-on-weak-signal+full-size) and `mode`
(`follow` = momentum, long on positive sentiment; `fade` = contrarian, long on NEGATIVE sentiment,
since the research says news sentiment mean-reverts). A global **risk-off gate** (`risk_off`, default
on) is the one documented drawdown-reducer: when broad `MARKET` sentiment <= -0.35, Follow arms cut
target exposure to 25% (toward cash); Fade arms ignore it. These are the levers the tool exists to
search over.

`POST /ext/paper_trade/exp/start` body `{model, arms:[{market, focus, aggr, mode}], interval, risk_off, reset}`
launches up to `EXP_MAX_ARMS` parallel arms — each a `{market, focus}` slice (focus=None => broad)
on its own ledger, all scored by ONE shared model. Arm specs are normalized to stable labels
(`_arm_spec`) and persisted to `experiment.json`, so the experiment + its ledgers resume across
restarts (the cron re-starts with `reset:false`). `.../exp/stop`, `.../exp/status`,
`.../exp/update` (live-edit model/gate/band/interval on all arms), and `.../exp/accounts` (per arm:
equity, pnl, **vs own buy-and-hold** bench, holdings, recent trades, trades/ticks counts, series for
client-side hit-rate) drive the **Experiment panel**: one-click launch, a sorted leaderboard
(leading arm tinted; hit-rate shown only at >=10 resolved pairs, else "thin"), an overlaid equity
chart (all arms, shared Y, stable colors), and per-arm tabs (equity-vs-its-hold, cash-vs-invested
bar, holdings, recent trades). The standalone `news_trader.py` CLI (`--market`, `--label`, `--focus`)
is for headless use; don't run a CLI and a managed thread on the same ledger (they race).

### honest status (grounded in the trading research, 2026-06-19)

The literature is blunt: a **long-only, fee-paying news-sentiment strategy has no documented
net-of-cost edge.** The alpha lives in the short leg (unusable here); costs erase ~96% of the rest;
the signal mean-reverts (buy the spike, eat the reversal); crypto adds spread + pump contamination.
Realistic expectation: arms **track or slightly underperform** their own buy-and-hold. This is a
research instrument for measuring signal honestly, not a money-maker — the UI says so on the panel.

## page cockpit (page/index.html)

Two clearly separated sections: **LIVE TRADING (forward)** — the news bot: sentiment-model
picker (from `POST /teacher/models` provider=ollama), a **Trade-token selector** (`#pt-nt-token`:
Auto or one coin — single-token mode focuses both the news pull and the positions on that coin),
scan interval, trade sensitivity (band), gate, Start/End session buttons, the live ledger panel
(equity curve with the dashed BTC buy-hold benchmark), and the "News it's reading" feed —
and **BACKTEST (past data)** — the chart-model Historical/Replay tooling, which is **hidden
entirely while a live session runs** (`pollTraderStatus` toggles `#pt-bt-banner` + `#pt-bt` on
the running-state transition only, so a manual show/hide sticks).

The **"Is the news actually predicting?"** panel is a 6-chart grid (`drawInsights(series, focus)`),
the live costed version of the historical +0.48 probe: (1) sentiment-vs-next-move scatter
(green=direction right), (2) sentiment over time per coin, (3) sentiment-vs-price overlay for the
focus coin (lead/lag), (4) directional hit-rate bars by coin vs 50%, (5) rolling accuracy (is the
edge holding), (6) follow-signal-vs-hold cumulative return (before fees). Live controls + status
are wired to the `/trader/*` + `/account` + `/sentiment` routes; the token choice rides on
`start` (`focus`), `sentiment` (`token`), and is echoed back by `/account` (`focus`).

## dependencies + honest status

- Needs a configured teacher model (Ollama works locally). Calls `/teacher/complete` on the
  same host; the scraper hits public RSS + alternative.me.
- **Validation status (see `overnight_run_log.md`):** the scrape->score->aggregate path is
  built and verified live. But the only FREE historical sentiment series (fear-greed) has
  ~zero predictive correlation with forward returns (corr ≈ 0.02), so there is no free
  historical backtest for the LLM-news thesis. The published edge is event-level and fast;
  the only honest validation is FORWARD paper trading over weeks. Treat this as an
  instrument to RUN that forward experiment, not a validated money-maker.

## pitfalls

- Scoring is synchronous per headline; cap `n` on web requests. A down teacher host makes
  `/teacher/complete` return `ok:false` and items are dropped (signal thins, never crashes).
- `news_trader.py` is the execution loop (sentiment -> targets -> paper ledger). Backtests of
  LLM sentiment are look-ahead-contaminated (the model was trained on the past) and the free
  historical proxy (fear-greed) is a null predictor — so trust the FORWARD paper record, not a
  backtest. A GDELT historical headline-scoring test (`/tmp/gdelt_sent_test.py`) is the closest
  historical probe; see `overnight_run_log.md` for its result.
