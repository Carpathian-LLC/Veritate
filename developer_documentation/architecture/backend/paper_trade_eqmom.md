# paper_trade: equity momentum forward trader

## what it is

Monthly 12-1 US equity momentum, long-only, paper money. Forward validator for the backtest in
`developer_documentation/market/trading_model_plan.md` section 13, which passed its bar (+5.6pp/yr
CAGR over a survivorship-matched random-decile null) but cannot be trusted historically because
free data drops delisted names. The universe (~496 large caps) was frozen 2026-07-03 in
`extensions/installed/trading/data/paper/equity_universe.json`; the forward record from that date is
the claim's real test.

## how it works

- `eqmom_trader.py::tick` (extensions/canonical/trading/server/eqmom_trader.py:96) — hourly:
  mark held names + SPY (benchmark on every `_exp_view`-compatible history row). On a new calendar
  month, only during the US regular session (`news_trader.market_open`): fetch two years of Yahoo
  daily adjcloses per universe name (`daily_adjcloses` :62), rank 12-1 momentum (return t-252 to
  t-21 trading days, `momentum_12_1` :75), rebalance to the equal-weight top decile (~50 names,
  `targets` :83), 3bp per side. Requires >= 100 rankable names or holds and retries next tick.
- Ledger helpers reused from `news_trader`; marks fall back to `last_px` on quote failures.
- Routes (`register.py`): `POST /ext/trading/paper/eqmom/start|stop`, `GET .../status|accounts`.
- Auto-resume: `start_thread` stamps `auto: true` in the ledger, `stop_thread` clears it, and
  `register(app)` calls `resume()` at boot (same contract in `xsmom_trader`), so a dashboard
  restart never gaps the forward record.
- UI: the page's strategy dropdown (`eqmom` entry) renders the shared book panel (board, overlay
  equity vs $10k, per-arm tab, positions, recent trades).
- Standalone: `python extensions/canonical/trading/server/eqmom_trader.py` (`--once` single
  tick). Same ledger as the routes; run one or the other.

## dependencies

`news_trader` (ledger, stock quotes, market-hours gate), Yahoo Finance chart API (free, no key),
the frozen universe JSON.

## pitfalls

- A monthly fetch cycle is ~500 Yahoo history calls (paced by `FETCH_PAUSE`); transient failures
  just drop names from that ranking, and under 100 rankable names the month is retried next tick.
- No dividends are credited on held cash or positions between rebalances beyond what adjclose
  carries into the ranking; equity marks use live trade prices.
- First launch mid-month trades on that month's stamp; the first partial month is not a clean
  monthly sample.
- The backtest's +5.6pp/yr is an upper bound (backfilled winners inflate the momentum decile);
  never quote it as expected forward return.

## tests

`extensions/canonical/trading/tests/test_eqmom_trader.py` — 12-1 window arithmetic, short-
history exclusion, decile targets, monthly cadence, market-hours hold, resume-flag clearing.
Network-free.
