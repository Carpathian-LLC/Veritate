# Prediction page

Standalone page at `/predict` (separate from the `/app` dashboard and `/chat`),
served from [predict.html](../../../veritate_mri/web/predict.html). The
predictive-accuracy dashboard for price-series corpora.

## What it is

Pick a corpus (`stocks`/`crypto`), a predictor, and a bar count; run an eval; see
predicted-vs-actual next-bar return, the residual, and accuracy metrics. Isolated
on purpose: the experiment can iterate or be deleted with zero blast radius on the
working chat and training tabs.

## How it works

Self-contained page (own markup, styles, and a small canvas charter; no dependency
on the `index.js` dashboard monolith). Backed by
[predict_routes](../backend/predict_routes.md).

- On load, `loadCorpora()` fetches `GET /predict/corpora` and fills the corpus,
  predictor, and trained-model selects. Corpora not built on disk render disabled.
- `runEval()` posts `{source, predictor, n_bars}` to `POST /predict/eval` and
  renders. A selected trained model overrides the predictor field (and is rejected
  by the backend until the model path is wired).
- Charts ([drawCharts](../../../veritate_mri/web/predict.html)): canvas line of
  actual vs predicted return bucket over a flat (center) reference, and a residual
  line. Re-renders on resize.
- Metrics panel: directional accuracy, magnitude MAE, and up/down/flat base rates.

## Dependencies

- `GET /predict/corpora`, `POST /predict/eval` from [predict_routes](../backend/predict_routes.md).
- `GET /predict` page route at [app.py:80](../../../veritate_mri/app.py#L80).

## Pitfalls

- Baselines are the reference line, not a model. Direction sits near 50% by
  construction; the panel note says so. A trained series model is what makes the
  predicted line meaningful.
- The page reuses no dashboard JS; keep its charter tiny rather than importing the
  monolith into a standalone page.
