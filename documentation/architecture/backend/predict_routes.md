# predict_routes

Backend for the standalone `/predict` page: the predictive-accuracy dashboard for
price-series corpora (stocks, crypto).

## what it is

Two endpoints that score a next-bar return forecast against a corpus's held-out
val set. The forecast is one return bucket per bar; direction and magnitude are
read straight off the bucket. Baseline predictors need no model and define the
wall a trained model must beat.

## how it works

[predict_routes.py](../../../veritate_mri/routes/predict_routes.py). Feature math
and the bucket alphabet come from [series_codec](../../../veritate_mri/tools/series_codec.py)
(shared with the corpus builder). See [series_corpus.md](../../corpus/series_corpus.md).

- `GET /predict/corpora` — lists the `stocks`/`crypto` corpora present on disk
  (instrument count, max val bars), the local trained models with checkpoints, and
  the baseline predictor names.
- `POST /predict/eval` — body `{source, predictor, n_bars}`. Loads the val bin,
  takes the longest instrument's first `n_bars` return buckets as the actuals,
  runs the predictor, returns aligned `actual` + `pred` plus metrics.

Predictors ([_predict_baseline](../../../veritate_mri/routes/predict_routes.py#L52)):
`flat` (always center / no-move), `persistence` (= prior bar), `drift` (running
mean). A predictor name that is not a baseline is rejected with 400 until the
trained-model path is wired against a real series checkpoint.

Metrics ([_metrics](../../../veritate_mri/routes/predict_routes.py#L65)): directional
accuracy (sign hit-rate over non-flat actual bars), magnitude MAE in buckets, and
up/down/flat base rates.

## dependencies

- `readers.paths.corpus_val_path` for the val bin path; the raw byte read is
  inline (same pattern as training-side corpus reads; a one-caller reader helper
  would break the lean rule).
- `readers.models` / `readers.checkpoints` for the local-model list.
- `tools.series_codec` for decode + bucket constants.

## pitfalls

- Corpora are built from gitignored local data and are NOT in `corpus_catalog.json`;
  this endpoint discovers them by disk presence, not the catalog.
- The val set is the held-out future of each instrument; eval uses the single
  longest instrument so the chart is one continuous series, not a mix.
- Directional accuracy of `flat` is 0 by definition (it never calls a direction);
  read its magnitude MAE instead.
