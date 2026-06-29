# market register.py: market extension dashboard API

Market LLM is a **canonical extension** at `extensions/canonical/market/`
(self-contained: `manifest.json`, `register.py`, `page/index.html`, `server/`). The
extension registry (`extensions/registry.py`) auto-discovers it, inserts `server/` onto
`sys.path`, imports `register.py`, calls `register(app)` to add the `/market/*` API
routes, and mounts the page at `/market` from `manifest.page` (`page/index.html`). The
nav link (`nav_label: "Market LLM"`) is hidden unless the experimental toggle is on, so
`/market` and its endpoints stay reachable regardless of the toggle. Server modules import
each other by bare name (`import veritate`, `import data`, `import live`) because `server/`
is on `sys.path` at register time; the route bodies import lazily, so a missing dep never
breaks startup. The byte-model endpoints read the canonical Veritate model registry and
checkpoints **read-only** to run trained models (`veritate.list_models`/`load_model`); they
never mutate canonical training/chat/RAG state. The former `/predict` benchmark page and its
routes were folded into the `veritate_*` endpoints here, and the offline GBDT baseline routes
were removed, so the page is byte-model-only.

All handlers wrap their body in a local `_safe("market", fn)` helper in `register.py` for the
JSON-error contract (try/except, log via `runtime.logs`, return a JSON error body + 500),
which also keeps `NaN`/`Infinity` from reaching `jsonify` (invalid JSON breaks the browser
`r.json()`). The `veritate.*` functions return `None`, never `float("nan")`.

## Veritate endpoints (the byte model, on-mission engine)

- **GET `/market/veritate_models`** -> `{models:[...]}`: every byte-level Veritate model that
  has at least one checkpoint (`veritate.list_models`), regardless of `model_type`. Drives the
  multi-select compare picker; the page (not the route) decides which are overlaid.

- **GET `/market/veritate_hindcast?source&symbol&model&base&n`** -> price-space overlay
  (`veritate.hindcast`): per-bar `price`, `band` (expected move size, log-return),
  `p_up`, `mark` (1 right / -1 wrong / 0 flat), plus `hit_rate`, `n`, `step`, `base`. Same shape
  as `backtest.hindcast` so one drawer renders both. Inference is on CPU (no MPS contention
  with a live training run).

- **GET `/market/veritate_benchmark?source&symbol&model&base&n`** -> the full scored metric set
  in return-bucket space (`veritate.benchmark`): `metrics` (precision_decisive,
  high_conf_precision/_n, avg_confidence, up/down_precision, decisive_rate,
  directional_accuracy, magnitude_corr, magnitude_mae_buckets, calibration[], up/down/flat_rate,
  n), `trading` (trader-facing, gross/before-fees: win_rate, profit_factor, expectancy,
  sharpe per-trade, max_drawdown, total_return, n_trades), `baseline.magnitude_mae_buckets`
  (persistence), `equity` (cumulative edge), `actual`/`pred`/`conf`/`price`/`t` (downsampled
  chart arrays), `ret_center`/`ret_bins`, `base`, and `best`/`worst` (top-6 most confident
  right/wrong calls, ranked by confidence x move size). Metrics, `trading` and best/worst are
  computed on every scored bar in one pass; only the chart arrays are downsampled.
  Directional metrics (`directional_accuracy`, `decisive_rate`, `up_precision`,
  `down_precision`, `high_conf_precision`) use the **prob-mass up-vs-down sign**
  (`1 if P(up) >= P(down) else -1`), the exact directional call `hindcast` scores, so
  `benchmark.directional_accuracy` and `hindcast.hit_rate` agree (~0.49, coin-flip) on the
  same data. They do **not** use the argmax return bucket: for 1m returns the argmax sits on
  the flat center bin ~97% of the time, which would collapse `decisive_rate`/`directional_accuracy`
  to ~0.02 and falsely read as a broken model. `magnitude_corr`/`magnitude_mae_buckets` and the
  `equity` curve are unchanged in intent: magnitude still uses the bucket-offset distribution;
  equity uses the same prob-mass directional sign as the metrics.

- **Resolution (`base`).** `base` selects bar size: `1m, 5m, 15m, 1h` (default `1m`). All read the
  1m-native `extensions/installed/market/data/crypto/<symbol>.csv` and resample to coarser bars. The page is
  crypto-only; per-second and stocks were dropped from the UI.

- **On-demand backfill (no manual data).** A fresh install ships no data dir. When
  `data.load_tail` finds no local CSV for a crypto symbol it backfills via `server/fetch.py`:
  pages 1m klines from Binance (`api.binance.com` global, `api.binance.us` US fallback), writes
  them in `data.py`'s schema, and caches to `extensions/installed/market/data/crypto/<symbol>.csv` (later loads are
  local). When the API is unreachable it falls back to a hosted CSV URL listed in
  `server/market_data_catalog.json` (ships empty; operator fills it to enable the fallback).
  `data.list_instruments("crypto")` unions local symbols with `fetch.MAJORS` so the picker is
  populated before anything is cached.

- **GET `/market/veritate_live?source&symbol&model`** -> next-bar forecast
  (`veritate.predict_next`): `p_up`, `expected_move[_bps]`, `confidence`, `vol`, `lean`,
  `last_close`, `last_t`. Crypto uses live Binance.US klines (last CLOSED bar); stocks use
  the local daily tail.

## Paper-trading endpoints (drive the Paper Trading extension)

The trading-policy layer (`server/policy.py`, see [market_policy.md](market_policy.md)) turns
the model's per-bar forecast into trade decisions. `veritate.signal_series` produces the
policy-ready per-bar signal (the same scoring walk as `hindcast`, no downsampling); `policy`
scores it. Both endpoints take the policy overrides `mode` (`vol_harvest`|`directional`),
`fee_bps`, `conf_gate`, `move_gate`, `sizing` (`confidence`|`fixed`|`vol_target`) parsed by
`register._policy_args`. The Paper Trading extension (`extensions/canonical/paper_trade/`) is
the only caller; it consumes these over HTTP and holds its paper ledger in the browser.

- **GET `/market/paper_signal?source&symbol&model&base&n`** -> the raw per-bar forecast
  (`veritate.signal_series`): `t`, `price`, `p_up`, `conf`, `exp_move`, `vol`, `ret_next`,
  `n`, `base`, plus `symbol`/`model`/`step`. No policy applied. This is what the Paper
  Trading page fetches (and caches): it runs the policy in the browser so rule tweaks,
  the aggressiveness slider, and the optimizer reshape instantly with no re-score. Only a
  data change (model/symbol/resolution/bars) re-hits this route. The in-browser sim
  mirrors `policy.py`; keep the two in sync.

- **GET `/market/paper_backtest?source&symbol&model&base&n&mode&fee_bps&conf_gate&move_gate&sizing`**
  -> a vectorized backtest over the last `n` bars (`policy.backtest` on
  `veritate.signal_series`): the `_metrics` set (`n_trades`, `mean_bps`, `win_rate`, `sharpe`,
  `max_dd` in bps, `exposure`, `fee_bps`, `mode`), the aligned per-bar arrays `equity`
  (cumulative net pnl, bps), `gate`, `lean`, `size`, `pnl_bps`, the `t`/`price` series, and
  `trades` (most-recent gated bars: `t`, `price`, `side`, `lean`, `size`, `pnl_bps`). The
  **canonical server-side scorer** (the reference the page's in-browser sim mirrors); for
  programmatic/API backtests. The interactive page does not call it (it uses `paper_signal`
  + a client mirror so rule tweaks are instant).

- **GET `/market/paper_decide?source&symbol&model&mode&fee_bps&conf_gate&move_gate&sizing`**
  -> a single-bar live decision (`policy.decide` on `veritate.predict_next` +
  `veritate.trailing_premium`): `last_close`, `last_t`, `p_up`, `confidence`,
  `expected_move[_bps]`, `premium[_bps]`, and `decision` (`{act, side, size, reason}`). Drives
  Live mode; the page resolves each open decision against the next closed bar in the browser.

- **GET `/market/veritate_data_report?source`** -> cheap per-instrument inventory
  (`veritate.data_report`): `n`, `total_bars` (approx from file size), `gb`, and the largest
  instruments with bars/first/last. No full scan of multi-GB CSVs (head/tail reads only).

## Other endpoints

- **GET `/market/instruments?source=crypto`** -> `{ok, source, instruments:[...]}`: local raw 1m
  CSVs under `extensions/installed/market/data/crypto/` unioned with `fetch.MAJORS` (`data.list_instruments`), so the
  symbol selector is populated even on a fresh install with no cached data.

Downloadable add-on datasets (stocks, forex, the broader crypto archives, ...) are now served by
the **generic per-extension data routes** over the extension's `data_catalog.json`, not by
market-specific routes; the old `/market/extensions/{catalog,download,delete}` endpoints were
removed. See [documentation/extensions/authoring.md](../../../documentation/extensions/authoring.md).

There is no dashboard route or S3-URL setting for the byte corpus;
`server/corpus_manifest.py` is a standalone CLI that lists the data artifacts (raw OHLCV, built
byte corpuses, trained models).

## Notes
- **Per-model codec stride.** `veritate.load_model` returns the model's `bar_stride`. `save.py` no
  longer stamps `bar_stride`, so unstamped models default to the current codec stride
  (`series_codec.BAR_STRIDE`), the stride they were trained with. `hindcast` / `benchmark` /
  `predict_next` encode at that stride, so adding codec channels never breaks an older model that
  did stamp a smaller stride: it is served the exact byte format it trained on (the newer channels
  are simply not emitted for it).
- No API key is needed anywhere. Binance.US market data is public; api.binance.com is geo-blocked
  (451) in the US, so the crypto path of `veritate_live` calls only api.binance.us.
- The page is byte-model-only. The old GBDT baseline (`models.py`, `backtest.py`, `features.py`,
  `dataset.py`, `evaluate.py`, `horizon_study.py`) was deleted, so `server/` is now data +
  byte-model serving + live feed + codec/builder + capture CLIs only.
