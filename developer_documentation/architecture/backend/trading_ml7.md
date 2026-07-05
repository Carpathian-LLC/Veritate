# trading: ml7 forward trader (ML weekly book, 7-day ranks)

## what it is

Live forward paper arm for the audited h7 ML strategy: the first strategy family to survive
the full adversarial audit (`developer_documentation/market/trading_model_plan.md` section 11;
`SMOKE_RESULTS/daily_ml_panel_smoke.py` construction, `daily_ml_audit_smoke.py` hardening,
verdict CONFIRMED). A frozen 3-seed HistGradientBoostingClassifier ensemble scores next-7d
direction for the 40 crypto majors on a 28-feature daily panel; the book is dollar-neutral
long/short in 7 daily-staggered tranches. Fake money: JSON ledger
(`account_ml7_main.json`), no broker, no keys.

Validated expectation is the audit's lag1 arm (~12.9%/yr, Sharpe ~1.04, maxDD ~12%, 5/5 years
positive on seed means), NOT the same-close baseline (~15%/yr): live entries land at the tick
after the day closes, which is the lag1 timing by construction.

## how it works

- **Feature parity is the contract.** `ml7_features.py` is the ONE builder, extracted verbatim
  from the smoke: `daily_from_minute_csv` (:51, 1m CSV -> UTC-day bars + `nmin` gap count),
  `compute_features` (:73, the 28 FEATS + `y_sign_7`), `build_panel` (:128, cross-sectional
  `xs_rank_7/30`, all-FEATS-notna filter). Training and live scoring both consume it;
  `tests/test_ml7_features.py` asserts byte-equal output vs the smoke's own code on a fixture.
  Editing any formula requires a retrain.
- **Training** (`ml7_trader.py::train` :318, CLI `--train` only, never at import/boot): resample
  the full `crypto_of` 1m CSVs to daily, bridge from the CSV end (2026-05-31) to the last closed
  UTC day with OKX `1Dutc` bars + OKX funding history, fit
  `HistGradientBoostingClassifier(max_iter=200, random_state=s)` for seeds {0,1,2} on all labeled
  rows, `joblib.dump` the list to `installed/trading/data/paper/ml7_model.joblib` + manifest
  (`ml7_model_manifest.json`: train_end, data_end, rows, features, seeds, sklearn version).
  Also seeds the rolling live caches. First run: ~85.8k train rows, fit 87s, total 163s.
  **Retrain quarterly** via `--train` (the audit trained per-year; a quarterly refresh is well
  inside the validated staleness) and after any sklearn major upgrade (joblib pickles are not
  cross-version stable).
- **Live caches**: `installed/trading/data/paper/ml7_history/<PAIR>.csv` (daily OHLCV+nmin,
  last 400 days) + `ml7_history/funding/<PAIR>.csv` (daily funding sums). Extended each trading
  tick from OKX closed `1Dutc` candles (`confirm=="1"` only) and
  `/public/funding-rate-history` (8h rates summed per UTC day, the smoke's convention; missing
  funding degrades to the smoke's 0-fill). CSV loads use `float_precision="round_trip"`: the
  default pandas parser is 1 LSB off, which would break parity.
- **Tick** (`tick` :276, hourly): on a new closed UTC day, `_signal` (:191) extends caches,
  rebuilds the panel, scores mean `predict_proba` across the ensemble (`load_model` :172,
  lazy, graceful "missing - run --train" via `model_state`); `tranche_targets` (:224) takes the
  top/bottom quintile of scored names at +-w, w = (1/7)/2/k of equity (k = int(n*0.2), 8 at full
  40); `rebalance_tranche` (:238) trades ONLY slot `epoch_day % 7`, fee 3.5bp per side on traded
  delta, the other six tranches ride (the audited turnover model). Day stamp `led["day"]` gates
  one trade per day; feed/model failure leaves the stamp unset so the next tick retries. Every
  tick marks to OKX and appends an `_exp_view`-compatible history row (`bench_px`=BTC,
  `last_px` fallback, HISTORY_CAP 5000).
- **Funding is neither credited nor charged** in the ledger. Audit measured it at ~-1%/yr on
  this book (funding_off arm passed on its own); omission is honest in either direction.
- Routes (`register.py`): `POST /ext/trading/paper/ml7/start|stop`, `GET .../ml7/status`
  (+`model` state), `GET .../ml7/accounts`; in `/ext/trading/system` runnables (`ml7_main`) and
  `stop_all`. Auto-resume stamp (`auto` in the ledger) identical to xsmom/eqmom; nothing starts
  at boot without it.
- Standalone: `python extensions/canonical/trading/server/ml7_trader.py` (`--train`, `--once`).
  Same ledger as the managed thread; run one or the other, not both at once.

## dependencies

- `ml7_features` (parity module), `news_trader` (ledger helpers, QUOTE), `xsmom_trader`
  (`_get`, `mark_price`, UNIVERSE), `data` (`source_dir`, `FUNDING_DIR`).
- `scikit-learn` + `joblib` (extension `requirements.txt`); pandas/numpy from the platform.

## pitfalls

- **OKX `bar=1D` is UTC+8-aligned.** ml7 uses `bar=1Dutc` (`OKX_CANDLES_UTC`); plain 1D would
  shift every bridged bar 8 hours and silently break UTC-day parity with the CSV resample.
- **The scoreable universe is currently 37/40**: FTM (CSV ends 2025-01), MKR (2025-09) and RUNE
  (2026-05) are OKX-dead; their caches go stale, they simply never score, and k adapts
  (int(37*0.2)=7 a side) exactly as the audited per-date construction did.
- **Venue seam**: history before 2026-06 is Binance (CSVs), after is OKX. Close prices agree to
  bps; `volume_exp`/`range_exp` see a one-off level seam that washes out of the 7d/30d ratio
  windows within a month. Funding bridges from Binance CSVs to OKX swap rates the same way.
- The feature builder needs ~120 warm days per pair; the 400-day rolling cache gives margin.
  Seeding only happens in `--train`: the live loop never reads the 12 GB 1m CSVs.
- Real seed variance was measured in the audit (per-seed lag1 mean Sharpe 0.97-1.15, single
  seed-years as low as -0.08): bumpy months are expected; judge the arm against the ~13%/yr
  lag1 expectation, not the baseline.
