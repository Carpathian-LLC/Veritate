# market execution.py + autotrader.py: autonomous paper-trading plumbing

The execution layer turns a policy decision into broker orders and runs the full
autonomous loop. It is the **test rig** for taking a strategy live-in-simulation; it does
not create an edge. Lives in the market extension's `server/` beside the model serving it
consumes.

## what it is

- `execution.py` — an Alpaca **paper** REST adapter plus the pure decision-to-order
  translation. Alpaca crypto is **spot, long-only**, so a policy decision maps to a target
  long exposure (0..1); short and straddle are not expressible and map to flat. The honest
  venue constraint, encoded.
- `autotrader.py` — the standalone CLI loop: each tick pulls the latest closed bar
  (`live.fetch`), forecasts (`veritate.predict_next` + `trailing_premium`), decides
  (`policy.decide`), and rebalances the position (`execution.rebalance`).

## how it works

- `to_alpaca_symbol(sym)` (`execution.py`): `BTCUSDT -> BTC/USD` (strips USDT/USDC/USD,
  quotes in USD).
- `target_qty_for(decision, equity, price)`: long decision -> `size * equity / price`;
  everything else -> 0.
- `plan_order(current, target, min_qty)`: buy/sell to close the gap, or `None` under
  `min_qty` (dust filter, `MIN_NOTIONAL` USD).
- `Broker` (`execution.py`): keys from `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` env or
  constructor. Paper base URL by default; live needs `paper=False` + keys. One network seam
  `Broker._http` (mocked in tests). `dry_run` returns a simulated ack without sending.
  `account`/`equity`/`position_qty`/`submit_order` (market order, `gtc`).
- `rebalance(broker, symbol, decision, price)`: reads equity + current position, computes
  the target, submits at most one market order.
- `autotrader.tick(symbol, bundle, broker, overrides)`: one cycle, returns a log record.
  `loop(...)` repeats every `interval`, bails after `MAX_ERRORS` consecutive failures.
  Closed-bar rule: forecasts off `df.iloc[:-1]`, never the forming candle.

Run: `python extensions/canonical/market/server/autotrader.py --model <name> --symbol BTCUSDT`
(paper by default; `--dry_run` to log-only; `--live` is the only path to real money and
needs real keys).

## dependencies

- `veritate` (forecast), `policy` (decision), `live` (feed) — all in the same `server/`.
- A free **Alpaca paper account**: set `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` to run
  live-in-sim. Without keys the loop still loads but `Broker` raises a clear error on any
  network call (use `--dry_run` to exercise it keyless).
- Tests `tests/test_execution.py` (8) + `tests/test_autotrader.py` (2): broker network and
  the model are mocked; no live calls, no checkpoint.

## pitfalls

- **Spot long-only.** Alpaca crypto cannot short or trade options, so vol-harvest
  (straddle) and directional shorts collapse to flat here. The carry strategy needs a perps
  venue, not Alpaca.
- **Paper is not proof.** A good paper run tests the plumbing, not an edge. Direction is a
  coin flip; do not promote to `--live` on the strength of a paper week.
- **Safety defaults.** Paper + (optional) dry_run are the defaults; `--live` is explicit and
  guarded by key presence. Keep it that way.
