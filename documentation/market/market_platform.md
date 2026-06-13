# Market LLM: experimental price-forecasting platform

A self-contained, GBDT-based market forecaster that lives **entirely outside** the
canonical Veritate pipelines (trainers / chat / RAG). It forecasts short-horizon
crypto (and stock) price behavior, shows a forward probability cone, calibrated
direction, regime, EV/Kelly sizing, an honest backtest, and a live Binance.US feed.

Reachable at **`/market`**, gated behind the **experimental** settings toggle. It only
reads `external_data/*.csv` and `models/market/*.joblib`; it never imports or mutates
canonical training, chat, or RAG code. If it is ever spun out into its own service, the
whole platform is the `veritate_mri/market/` package plus `routes/market_routes.py` and
`web/market.html`.

## Why GBDT and not the byte-level Veritate model

A research sweep (signal/horizons, features, validation, models) plus an empirical study
on our own 78M bars converged hard:

- **Direction is ~unpredictable** at 1m–1h from price+volume. On 14 majors, lag-1 return
  autocorrelation is ≈0 (slightly negative microstructure bounce), and naive momentum
  hits 43–47% — below a coin flip. Transaction costs (~20 bps round-trip) annihilate the
  tiny edge that exists.
- **Volatility is strongly forecastable.** Absolute-return autocorrelation is 0.22–0.41;
  a trivial EWMA already explains ~23% of next-bar magnitude variance.
- For tabular, heavy-tailed, low-SNR financial features, **gradient-boosted trees beat
  deep nets** (Grinsztajn 2022; M5; FinTSB). The byte-level transformer is the wrong tool
  here. The dominant lever is features + labeling + leak-free validation, not architecture.

So the platform forecasts **volatility + a calibrated probability distribution** of the
next move (a cone), never a bare up/down point. Measured out-of-sample (purged split,
20 pooled pairs):

| horizon | vol R² (GBDT) | direction acc | base rate |
|--------:|--------------:|--------------:|----------:|
|   5 min |         0.412 |        53.5%  |    50.3%  |
|  15 min |         0.623 |        52.0%  |    50.3%  |
|   1 hour|         0.685 |        50.5%  |    50.4%  |

Volatility forecastability **rises** with horizon; the directional edge **fades to ~0**
by 1 hour. GBDT beats HAR-OLS / EWMA / persistence baselines on vol R² out-of-sample.

## Pipeline (the `market/` package)

```
external_data/<source>/<SYM>.csv  (raw 1m OHLCV, mixed ms/us epochs)
        │  data.py        normalize_time (per-row unit detect) + resample to base TF
        ▼
   features.py            32 leak-free, trailing-only features (the ONE shared
                          train+serve function — no skew). no-lookahead unit-tested.
        │
        ├─ dataset.py     forward labels (y_ret, y_vol, vol-scaled y_dir) + PURGED
        │                 time split (train labels resolve before val; embargo gap)
        ▼
   models.py              per (base,horizon) bundle: vol regressor (log-vol),
                          isotonic-CALIBRATED direction classifier, split-conformal
                          cone scale, regime thresholds → models/market/<base>_h<h>.joblib
        │
        ├─ evaluate.py    certification harness: GBDT vs HAR/EWMA/persistence + direction
        │                 vs base-rate + cost-aware backtest Sharpe (the honest read)
        ├─ backtest.py    per-instrument replay for the dashboard (vol pred-vs-actual,
        │                 reliability, cost-aware equity)
        └─ live.py        Binance.US REST klines → predict from last CLOSED bar
```

### Key correctness guarantees
- **No lookahead**: every feature at bar `t` uses only bars ≤ `t`; verified by a
  truncation test (`features._self_test`) and the codec tests.
- **Purged split**: `dataset.purged_split` drops train rows whose label window overlaps
  val, plus a horizon-sized embargo — no leakage across the boundary.
- **One feature path**: training and live serving both call `features.compute`, so there
  is no train/serve skew (the #1 production failure mode in the research).
- **Closed-bar rule (live)**: `live.predict` drops the still-forming current bar and
  forecasts from the last closed bar.
- **Honest costs**: 20 bps round-trip modeled everywhere; the dashboard states plainly
  that the directional EV is usually negative after fees and that the trustworthy output
  is the expected-move (volatility), not a buy/sell signal.

## Retraining

```
python veritate_mri/market/models.py --horizons 5,15,60 --base 1m --pairs 28 --max-bars 500000
```
Writes `models/market/1m_h{5,15,60}.joblib` + `summary.json`. Add horizons or raise
`--pairs` / `--max-bars` as more data lands. To certify honestly first:
```
python veritate_mri/market/evaluate.py --horizon 15 --pairs 30
```

## Data & corpus (S3 pathway)

Raw 1m history is pulled by `external_data/pull_binance.py` (Binance Vision archive;
api.binance.us only for volume ranking) into `external_data/crypto|stocks/` — gitignored,
GB-scale. These are **experimental corpuses**: they appear in the Corpus library only when
the experimental toggle is on, and can be added to the training mix or shipped.

Because they are large, host them on Carpathian S3 and set `market_corpus_s3_url` in
settings; the experimental corpus panel then offers a download from that base URL. Until
set, the panel points at the local paths. The exact files to upload are listed by
`python veritate_mri/market/corpus_manifest.py` (built/raw paths + sizes).

## Live feed

`/market/live` polls Binance.US REST (`/api/v3/klines`, no API key, weight 1), computes
features on closed bars, predicts, and returns the forward cone. The dashboard's **● LIVE**
button polls it every 12s and advances the cone minute by minute. api.binance.com is
geo-blocked (HTTP 451) in the US — only api.binance.us is used. A websocket upgrade
(`wss://stream.binance.us:9443/ws/<sym>@kline_1m`, act on `k.x==true`) is the future
lower-latency path; REST polling is the dependency-free version shipped now.

## Honest framing (what to tell users)

This is a **risk/volatility forecaster with a calibrated directional lean**, not a
get-rich oracle. It is genuinely good at predicting *how big* the next move will be
(vol R² up to ~0.69) and honest about being ~coin-flip on *which way*, especially at
longer horizons. Used correctly it sizes risk, draws expected ranges, and flags regime —
the things that actually help, demonstrated rather than asserted.
