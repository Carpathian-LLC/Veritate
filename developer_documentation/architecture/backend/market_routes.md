# market_routes.py: experimental market dashboard API

Backend for the `/market` Market LLM page. Registered in `app.py`
(`market_routes.register(app)`); the page itself is served at `/market`
(`market_page()` -> `web/market.html`). The route is served unconditionally; only the
dashboard nav link is hidden unless the experimental toggle is on, so `/market` and its
endpoints stay reachable regardless of the toggle. Routes import the `market/` package
lazily, so a missing sklearn never breaks startup. The byte-model endpoints read the
canonical Veritate model registry and checkpoints **read-only** to run trained models
(`veritate.list_models`/`load_model`); they never mutate canonical training/chat/RAG state.
The former `/predict` benchmark page and its routes were folded into the `veritate_*`
endpoints here, and the offline GBDT baseline routes were removed, so the page is byte-model-only.

All handlers wrap their body in `_safe("market", fn)` for the JSON-error contract, which
also guards against `NaN`/`Infinity` reaching `jsonify` (invalid JSON breaks the browser
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
  1m-native `external_data/crypto/<symbol>.csv` and resample to coarser bars. The page is
  crypto-only; per-second and stocks were dropped from the UI.

- **On-demand backfill (no manual data).** A fresh install ships no `external_data/`. When
  `data.load_tail` finds no local CSV for a crypto symbol it backfills via `market/fetch.py`:
  pages 1m klines from Binance (`api.binance.com` global, `api.binance.us` US fallback), writes
  them in `data.py`'s schema, and caches to `external_data/crypto/<symbol>.csv` (later loads are
  local). When the API is unreachable it falls back to a hosted CSV URL listed in
  `market/market_data_catalog.json` (ships empty; operator fills it to enable the fallback).
  `data.list_instruments("crypto")` unions local symbols with `fetch.MAJORS` so the picker is
  populated before anything is cached.

- **GET `/market/veritate_live?source&symbol&model`** -> next-bar forecast
  (`veritate.predict_next`): `p_up`, `expected_move[_bps]`, `confidence`, `lean`,
  `last_close`, `last_t`. Crypto uses live Binance.US klines (last CLOSED bar); stocks use
  the local daily tail.

- **GET `/market/veritate_data_report?source`** -> cheap per-instrument inventory
  (`veritate.data_report`): `n`, `total_bars` (approx from file size), `gb`, and the largest
  instruments with bars/first/last. No full scan of multi-GB CSVs (head/tail reads only).

## Other endpoints

- **GET `/market/instruments?source=crypto`** -> `{ok, source, instruments:[...]}`: local raw 1m
  CSVs under `external_data/crypto/` unioned with `fetch.MAJORS` (`data.list_instruments`), so the
  symbol selector is populated even on a fresh install with no cached data.
- **GET `/market/extensions/catalog`** -> `{ok, extensions:[...]}`: downloadable add-on datasets
  with live local status (`present`, `files`, `size_gb`, `downloadable`). See
  [market_extensions.md](market_extensions.md).
- **POST `/market/extensions/download`** `{source}` -> pulls a hosted dataset into
  `external_data/extension_data/<source>` (placeholder until the catalog url + S3 host land).
- **POST `/market/extensions/delete`** `{source}` -> reclaims a dataset's disk; symlinked
  (externally-parked) datasets only lose the link, never the archive.

The data-artifact paths (raw OHLCV, built byte corpuses, trained models) are listed in the root
doc `market_llm_data_manifest.md`. There is no dashboard route or S3-URL setting for the corpus;
`market/corpus_manifest.py` is a standalone CLI for the same listing.

## Notes
- **Per-model codec stride.** `veritate.load_model` returns the model's `bar_stride` (stamped at
  train time by `training/save.py`; default `LEGACY_STRIDE=5` for models saved before it existed).
  `hindcast` / `benchmark` / `predict_next` encode at that stride, so adding codec channels never
  breaks an older model on the page: it is served the exact byte format it trained on (the newer
  channels are simply not emitted for it).
- No API key is needed anywhere. Binance.US market data is public; api.binance.com is geo-blocked
  (451) in the US, so the crypto path of `veritate_live` calls only api.binance.us.
- The page is byte-model-only. The old GBDT baseline (`market/models.py`, `backtest.py`,
  `features.py`, `dataset.py`, `evaluate.py`, `horizon_study.py`) has been deleted, not just
  unrouted, so the `market/` package is now data + byte-model serving + live feed only.
