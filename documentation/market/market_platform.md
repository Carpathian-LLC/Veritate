# Market LLM: experimental price-forecasting page

The `/market` page is driven by the **byte-level Veritate model**: a model trained on raw
price tape (no news, no labels) that forecasts the next bar's return bucket. The page tests
whether that model can read price. It is byte-model-only; an earlier GBDT (gradient-boosted
trees) baseline has been removed from the package.

Reachable at **`/market`**. The route is served unconditionally by `veritate_mri/app.py`;
only the dashboard nav link is hidden unless the experimental settings toggle is on. The
`/market` route itself is reachable regardless of the toggle.

## Scope of state it touches

- **Byte-model serving path** (`veritate.py`): reads the canonical Veritate model registry
  and checkpoints **read-only** to run trained models (`readers.checkpoints`,
  `readers.models`, `veritate_core.load.load_from_state_dict`). It never mutates canonical
  training, chat, or RAG state and never writes into their directories. `list_models()`
  surfaces canonical byte models that have at least one checkpoint.
- **Data layer** (`data.py`): reads `external_data/<source>/*.csv` (read-only, byte-tail for
  large files) and resamples with no lookahead. The package writes no model artifacts.

## Byte model (the page engine)

The byte model forecasts the next bar's return **bucket** (a z-scored, scale-free class).
`veritate.hindcast` walks an instrument, turns each predicted bucket distribution into a
price-space guess (directional lean + expected move = E|z| x trailing sigma) and scores it
against the actual next bar. `veritate.benchmark` scores the full predict-page metric set in
return-bucket space (precision-when-decisive, calibration, magnitude correlation vs
persistence, equity). `veritate.predict_next` gives the live next-bar forecast. Inference
runs on CPU (no MPS contention with a live training run) using the same `series_codec`
contract the corpus was built with.

Direction at 1m is near a coin flip; the trustworthy signals are move magnitude and
confidence calibration. The UI foregrounds those and shows the dollar equity curve even when
it loses money.

## Byte corpus

Built by `veritate_mri/tools/build_series_corpus.py` from `external_data/<source>/*.csv` into
`trainers/corpus/{crypto,stocks}_{train,val}.bin`. One corpus per asset class; instruments
are anonymous (no ticker label) so the model learns one instrument-agnostic tape dynamic.
Per-instrument time split (oldest `1-val_ratio` train, newest val). No pair or bar caps.

```
python veritate_mri/tools/build_series_corpus.py --source crypto
python veritate_mri/tools/build_series_corpus.py --source stocks
```

Current build: crypto ~1.31B tokens across 200 pairs; stocks ~0.012B tokens across ~500
tickers. Bytes are the tokens. These are experimental corpuses: they appear in the Corpus
library only when the experimental toggle is on. Large bins host on Carpathian S3 and are
pulled to local paths by hand; there is no in-dashboard S3 URL feature. All data-artifact
paths live in the root doc `market_llm_data_manifest.md`; `corpus_manifest.py` is a
standalone CLI that lists the same files.

## Why direction is hard (research context)

Before the byte model, a research sweep on ~78M bars characterized the problem: direction is
~unpredictable at 1m-1h (lag-1 return autocorrelation ~0; momentum below a coin flip; ~20 bps
round-trip costs erase the tiny edge), while volatility is strongly forecastable (abs-return
autocorrelation 0.22-0.41). That is why the page foregrounds move magnitude and confidence
calibration over raw up/down accuracy, and shows the dollar equity curve even when it loses
money. The GBDT baseline that produced these numbers has since been removed; the byte model is
the product.

Correctness guarantees: trailing-only features (no lookahead) through the shared `series_codec`
(one contract for corpus build and serving, so no train/serve skew), a per-instrument
oldest-train / newest-val time split in the corpus, and the closed-bar rule on live (the
still-forming bar is dropped). 20 bps round-trip costs are modeled in the trading sim.

## Live feed

Crypto live forecasts poll Binance.US REST (`/api/v3/klines`, no API key) and forecast from
the last closed bar; stocks use the local daily tail. api.binance.com is geo-blocked (HTTP
451) in the US, so only api.binance.us is used.
