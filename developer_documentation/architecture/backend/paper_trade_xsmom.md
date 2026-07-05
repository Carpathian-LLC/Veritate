# paper_trade: xs momentum forward trader

## what it is

Weekly cross-sectional momentum long/short paper trader inside the Trading extension. Forward
out-of-sample validator for the one strategy that stayed net-positive across every backtest year at
real fees (`developer_documentation/market/trading_model_plan.md` sections 7 and 9). Fake money:
JSON ledgers, no broker, no keys.

Two arms on separate ledgers:

- `base` (`account_xsmom_base.json`) — always invested.
- `gated` (`account_xsmom_gated.json`) — flat when BTC 30d annualized vol >= its trailing 365d
  median. The gate cleared the backtest bar (test Sharpe 0.93, all years positive, maxDD 10%) but
  was test-adjudicated (placebo gates clear ~21% of the time), so both arms run and the forward
  record decides.

## how it works

- `xsmom_trader.py::tick` (extensions/canonical/trading/server/xsmom_trader.py:150) — hourly:
  mark held positions to live OKX prices, append a history row (`_exp_view`-compatible: `equity`,
  `bench_px`=BTC, `acts`). On a new ISO week: rank trailing 7d returns over the 40-major universe,
  rebalance to long top-8 / short bottom-8 (`targets` :119, `rebalance_to` :131), 3.5bp per side on
  traded notional. Gate evaluated only at rebalance (`vol_gate_invested` :107), matching backtest
  cadence.
- Prices: OKX public REST (`daily_closes` :84 pages `history-candles` for the 395-day vol lookback;
  closed candles only, `confirm=="1"` — no lookahead). Unfetchable/delisted names never trade.
- Ledger helpers reused from `news_trader` (`load_ledger`/`save_ledger`/`equity`/`ledger_for`);
  positions are signed (short opens add cash).
- Routes (`register.py`): `POST /ext/trading/paper/xsmom/start` `{arms?: ["base","gated"]}`,
  `POST .../xsmom/stop`, `GET .../xsmom/status`, `GET .../xsmom/accounts` (per-arm `_exp_view`
  payload + `invested`). Threads via `start_thread`/`stop_thread` mirroring `news_trader`.
- Standalone: `python extensions/canonical/trading/server/xsmom_trader.py --arm both`
  (`--once` for a single tick). Same ledgers as the routes; run one or the other, not both at once.
- UI: the page's strategy dropdown (`xsmom` entry) shows the arms board (state/return/vs-BTC/
  trades), overlay equity chart, per-arm tabs with the signed book and recent trades, and
  Start/Stop wired to the routes.

## dependencies

`news_trader` (ledger + quote constants), `certifi`, OKX public API (US-reachable, no auth).

## pitfalls

- No funding credit and no cash yield are modeled; both are conservative for this book (backtest
  realized funding was +1.85%/yr in its favor).
- The week stamp is written only after a successful rebalance; a feed failure retries next tick.
- A mid-week first launch trades immediately on that week's stamp — the first partial week is not
  a clean weekly sample.
- Marks fall back to the last known price (`last_px`) when a quote fetch fails, so equity history
  does not crater on transient feed errors.
- Run the dashboard threads OR the standalone CLI, never both (same ledgers).
- `_get`, `mark_price`, and `UNIVERSE` are consumed by `ml7_trader.py`
  ([trading_ml7.md](trading_ml7.md)); renaming them must update both.

## tests

`extensions/canonical/trading/tests/test_xsmom_trader.py` — ranking, gross, gate both regimes,
signed rebalance math, exit-untargeted, weekly cadence, history shape. Network-free.
