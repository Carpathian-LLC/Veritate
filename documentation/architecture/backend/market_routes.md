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

- **GET `/market/veritate_models`** -> `{models:[...]}`: byte-level Veritate models that
  have at least one checkpoint (`veritate.list_models`). Drives the model dropdown.

- **GET `/market/veritate_hindcast?source&symbol&model&n`** -> price-space overlay
  (`veritate.hindcast`): per-bar `price`, `band` (expected move size, log-return),
  `p_up`, `mark` (1 right / -1 wrong / 0 flat), plus `hit_rate`, `n`, `step`. Same shape
  as `backtest.hindcast` so one drawer renders both. Inference is on CPU (no MPS contention
  with a live training run).

- **GET `/market/veritate_benchmark?source&symbol&model&n`** -> the full scored metric set
  in return-bucket space (`veritate.benchmark`): `metrics` (precision_decisive,
  high_conf_precision/_n, avg_confidence, up/down_precision, decisive_rate,
  directional_accuracy, magnitude_corr, magnitude_mae_buckets, calibration[], up/down/flat_rate,
  n), `baseline.magnitude_mae_buckets` (persistence), `equity` (cumulative edge), `actual`/
  `pred`/`conf`/`price`/`t` (downsampled chart arrays), `ret_center`/`ret_bins`, and `best`/
  `worst` (top-6 most confident right/wrong calls, ranked by confidence x move size). Metrics
  and best/worst are computed on every scored bar in one pass; only the chart arrays are downsampled.

- **GET `/market/veritate_live?source&symbol&model`** -> next-bar forecast
  (`veritate.predict_next`): `p_up`, `expected_move[_bps]`, `confidence`, `lean`,
  `last_close`, `last_t`. Crypto uses live Binance.US klines (last CLOSED bar); stocks use
  the local daily tail.

- **GET `/market/veritate_data_report?source`** -> cheap per-instrument inventory
  (`veritate.data_report`): `n`, `total_bars` (approx from file size), `gb`, and the largest
  instruments with bars/first/last. No full scan of multi-GB CSVs (head/tail reads only).

## Other endpoints

- **GET `/market/instruments?source=crypto`** -> `{ok, source, instruments:[...]}`: the symbols
  available as raw 1m CSVs under `external_data/<source>/` (`data.list_instruments`). Drives the
  symbol selector.

The data-artifact paths (raw OHLCV, built byte corpuses, trained models) are listed in the root
doc `market_llm_data_manifest.md`. There is no dashboard route or S3-URL setting for the corpus;
`market/corpus_manifest.py` is a standalone CLI for the same listing.

## Notes
- No API key is needed anywhere. Binance.US market data is public; api.binance.com is geo-blocked
  (451) in the US, so the crypto path of `veritate_live` calls only api.binance.us.
- The page is byte-model-only. The old GBDT baseline (`market/models.py`, `backtest.py`,
  `features.py`, `dataset.py`, `evaluate.py`, `horizon_study.py`) has been deleted, not just
  unrouted, so the `market/` package is now data + byte-model serving + live feed only.
