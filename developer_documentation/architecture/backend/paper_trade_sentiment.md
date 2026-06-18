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
  source, use_chart}`, defaults to `sentiment.DEFAULT_MODEL`. `POST .../trader/stop` stops
  it; `GET .../trader/status` -> `{running, model, interval, ...}`. The page's Start/End
  session buttons + status poll drive these. The thread is the single owner — the standalone
  `news_trader.py` CLI is for headless use; don't run both on one ledger (they race).

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
