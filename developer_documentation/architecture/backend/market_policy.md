# market policy.py: trading-policy layer

`extensions/canonical/market/server/policy.py` converts the byte model's per-bar
forecast into trade decisions and scores them with fees. It is **model-agnostic**: it
operates on a signal series (`price`, `p_up`, `conf`, `exp_move`, `vol`, `ret_next`),
not the model. `veritate.signal_series` (`server/veritate.py`) produces that series;
`policy` scores it. The Paper Trading extension is the consumer, over the
`/market/paper_*` routes (see [market_routes.md](market_routes.md)).

## what it is

The model's validated edge is **magnitude/volatility**, not direction: expected-|z| vs
realized-|z| correlates ~0.25 to 0.30 (rising with horizon), while direction is ~coin
-flip. So the default mode is vol-harvesting; a directional mode is kept for comparison.

- **`vol_harvest`** (default): direction-agnostic. Buy volatility (straddle proxy) when
  the forecast move clears the prevailing premium. Per-bar payoff
  `|ret_next| - premium - fee`, sized. Premium is trailing mean `|ret|` over
  `premium_window`, or an explicit per-bar `premium` array (e.g. DVOL implied vol).
- **`directional`**: trade the `p_up` lean only when `exp_move >= move_gate * fee` and
  `conf >= conf_gate`. Payoff `sign(p_up-0.5) * ret_next - fee`, sized. Confirms
  direction stays unprofitable after fees.

## how it works

- `backtest(price, p_up, conf, exp_move, vol, ret_next, **overrides)`
  (`policy.py:65`): vectorized scorer. Returns the `_metrics` set plus the aligned
  per-bar arrays `equity` (cumulative net pnl, **bps**), `gate`, `lean`, `size`,
  `pnl_bps`. `ret_next[i]` is the realized log return from bar i to i+1 (the outcome of
  acting at bar i).
- `decide(p_up, conf, exp_move, vol, premium=None, **overrides)` (`policy.py:129`):
  single-bar live decision the downstream trader / Paper Trading Live mode calls.
  Returns `{act, side, size, reason}` (`side` is `straddle` for vol_harvest).
- `trades(sig, res, limit=200)` (`policy.py`): the most-recent gated bars as per-trade
  rows (`t`, `price`, `side`, `lean`, `size`, `pnl_bps`) for the trades table.
- `_size` (`policy.py:52`): `fixed` (1.0), `confidence` (clip `conf`), or `vol_target`
  (`vol_target / vol`), capped at `max_size`.

`DEFAULTS` (`policy.py:31`) holds `mode`, `fee` (round-trip fraction, `0.0005` = 5 bps),
`conf_gate`, `move_gate`, `premium_window` (96), `premium`, `sizing`, `max_size`,
`vol_target`, `stop`.

## dependencies

- `numpy` only. No model, no I/O, no network. `veritate.signal_series` /
  `veritate.predict_next` + `veritate.trailing_premium` supply the inputs.
- Consumed by `register.py` (`/market/paper_backtest`, `/market/paper_decide`).

## pitfalls

- **Units.** `mean_bps`, `equity`, `pnl_bps`, and `max_dd` are all **bps**; `net` and the
  internal payoff math are return-unit fractions. `fee` overrides are **fractions**
  (`fee_bps/1e4`), not bps.
- **Implied premium.** `vol_harvest` defaults premium to trailing `|ret|`; pass a
  `premium` array to price against implied vol (DVOL). DVOL is BTC/ETH-only research data
  on the T9 drive, not in the extension data dir, so the shipped page uses the trailing
  default.
- **Direction is dead after fees.** `directional` mode exists to demonstrate this, not to
  trade. Tests in `tests/test_policy.py`.
