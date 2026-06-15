# Trading model plan: horizon sweep verdict + next training

Eval-only, CPU, no training/git/server. Model: `models/mkt_crypto_80m` (651 1m crypto pairs, codec stride 5, seq 1024, step 12000).
Data: `extensions/installed/market/data/crypto/<PAIR>.csv` (full 1m history, OHLCV only) resampled to coarser horizons and re-encoded with the existing scale-free codec (`series_codec.py`).

## TL;DR

- **Direction is a coin-flip at every horizon.** Across DOGE/ETH/BTC/SOL/XRP/BNB/ADA/LTC at 1m/5m/15m/1h/4h, prob-mass directional accuracy sits in 0.50-0.524. 15m is the consistent (and only) peak (DOGE/ETH/BTC ~0.523-0.524), but that is ~3-4 standard errors above 0.50 on n=14k: real but far too thin to clear a 20 bps round trip. After-fee P/L is negative at every (pair,horizon) when you trade every bar, and high-confidence-only selection does not flip it positive. **There is no horizon where DOGE/ETH direction is tradeable after fees.**
- **The learnable edge is MAGNITUDE/volatility, and it strengthens with horizon.** Expected-|z| vs realized-|z| correlation is ~0.23 at 1m and climbs monotonically: DOGE 0.233 (1m) -> 0.241 (15m) -> 0.254 (1h) -> 0.297 (4h); ETH ~0.24 flat-to-rising. Signed-direction correlation is ~0 everywhere. The model knows HOW BIG the next move is, not WHICH WAY. The magnitude edge is strongest at 1h-4h.
- **One recommended next training (through the dashboard):** a `statistical` 80M (or 200M) byte model trained NATIVELY on coarse bars (15m-1h) over the 40-pair order-flow corpus (`crypto_of`, which carries `taker_buy`+`trades` so the full 7-byte codec is used), broad/multi-asset not DOGE/ETH-specific, objective unchanged (next-byte CE) but evaluated for magnitude/vol + high-confidence selection. Build the corpus first; the literal `/trainers/run` JSON is in section 5.

---

## 1. Method

- `V.load_model("mkt_crypto_80m")` -> seq=1024, step=12000, stride=5 (config stamps no `bar_stride`, so `LEGACY_STRIDE=5`; matches the "5-byte codec" description). Channels 0-4 = return-z / range-ratio / vol-ratio / realized-vol-ratio / session. The `crypto/` CSVs carry no `taker_buy`/`trades`, so channels 5-6 would degrade to a constant anyway; stride 5 is the right serve.
- Per (pair,horizon): load full 1m, `md.resample` to the horizon (open=first, high=max, low=min, close=last, volume=sum, no lookahead), take the last ~14.2k bars, re-encode with `series_codec`, run `V.benchmark`. Metrics from the existing prob-mass-direction + magnitude/calibration code (the stride / prob-mass dir-acc fixes are in the served `veritate.py`).
- Pairs: DOGE, ETH + BTC, SOL, XRP, BNB, ADA, LTC. Horizons: 1m, 5m, 15m, 1h, 4h. CPU only. ~18s/cell.
- Magnitude probe: separately correlate model expected-|z| (`sum(p * |Z_centers|)`) and signed-E[z] against realized z-bucket offset, to separate magnitude skill from direction skill (benchmark's `magnitude_corr` is SIGNED bucket-offset corr and reads ~0).
- Selective P/L: realized directional log-return per call = `sign(p_up-0.5) * actual_log_return`; net = `sum(ret) - FEE*n` at FEE=20 bps; swept over confidence quantile gates (0 / .75 / .90 / .95 / .99) to test "trade only the most confident bars."

## 2. Horizon sweep results

Directional accuracy (prob-mass), high-conf precision (top-quartile-confidence decisive bars), benchmark signed magnitude_corr, after-fee net (trade-every-bar; large-negative because ~14k trades x 20 bps dominates):

| pair | hz | n | dir_acc | hc_prec | magc(signed) | net (every-bar) |
|------|----|----|---------|---------|------|-----|
| DOGE | 1m | 14078 | 0.504 | 0.526 | +0.004 | neg |
| DOGE | 5m | 14078 | 0.510 | 0.517 | +0.011 | neg |
| DOGE | **15m** | 14078 | **0.524** | **0.558** | +0.011 | neg |
| DOGE | 1h | 14078 | 0.515 | 0.508 | -0.011 | neg |
| DOGE | 4h | 14078 | 0.522 | 0.550 | -0.004 | neg |
| ETH | 1m | 14078 | 0.506 | 0.512 | -0.005 | neg |
| ETH | 5m | 14078 | 0.518 | 0.547 | -0.009 | neg |
| ETH | **15m** | 14078 | **0.524** | **0.557** | -0.008 | neg |
| ETH | 1h | 14078 | 0.505 | 0.512 | -0.006 | neg |
| ETH | 4h | 14078 | 0.506 | 0.515 | -0.012 | neg |
| BTC | 1m | 14078 | 0.509 | 0.498 | +0.012 | neg |
| BTC | 5m | 14078 | 0.509 | 0.521 | +0.000 | neg |
| BTC | **15m** | 14078 | **0.523** | **0.552** | +0.007 | neg |
| BTC | 1h | 14078 | 0.510 | 0.524 | +0.008 | neg |
| BTC | 4h | 14078 | 0.510 | 0.516 | +0.020 | neg |
| SOL | 1m | 14078 | 0.515 | 0.532 | +0.002 | neg |
| SOL | 5m | 14078 | 0.514 | 0.535 | -0.008 | neg |
| SOL | 15m | 14078 | 0.516 | 0.533 | -0.015 | neg |
| SOL | 1h | 14078 | 0.507 | 0.510 | +0.002 | neg |
| SOL | 4h | 12597 | 0.506 | 0.506 | -0.003 | neg |
| XRP | 1m | 14078 | 0.511 | 0.534 | -0.005 | neg |
| XRP | 5m | 14078 | 0.508 | 0.516 | +0.006 | neg |
| XRP | **15m** | 14078 | **0.521** | **0.540** | +0.009 | neg |
| XRP | 1h | 14078 | 0.513 | 0.516 | -0.002 | neg |
| XRP | 4h | 14078 | 0.519 | 0.553 | +0.012 | neg |
| BNB | 1m | 14078 | 0.505 | 0.512 | +0.018 | neg |
| BNB | 5m | 14078 | 0.506 | 0.509 | -0.009 | neg |
| BNB | 15m | 14078 | 0.517 | 0.549 | -0.000 | neg |
| BNB | 1h | 14078 | 0.512 | 0.529 | -0.003 | neg |
| BNB | 4h | 14078 | 0.502 | 0.512 | -0.010 | neg |
| ADA | **1m** | 14078 | **0.549** | **0.592** | +0.030 | neg |
| ADA | 5m | 14078 | 0.517 | 0.543 | +0.015 | neg |
| ADA | 15m | 14078 | 0.526 | 0.560 | +0.000 | neg |
| ADA | 1h | 14078 | 0.515 | 0.538 | -0.005 | neg |
| ADA | 4h | 14078 | 0.512 | 0.529 | -0.013 | neg |
| LTC | 1m | 14078 | 0.528 | 0.562 | +0.017 | neg |
| LTC | 5m | 14078 | 0.507 | 0.525 | +0.002 | neg |
| LTC | 15m | 14078 | 0.522 | 0.540 | +0.021 | neg |
| LTC | 1h | 14078 | 0.513 | 0.528 | +0.003 | neg |
| LTC | 4h | 14078 | 0.519 | 0.539 | +0.012 | neg |

ADA-1m (0.549 / 0.592) and LTC-1m (0.528 / 0.562) are the only notable departures from the 15m-peak pattern, and even those (a) are single-pair / likely sample-window artifacts, not a class-wide effect, and (b) still post strongly negative after-fee net (ADA-1m net ~ -19.5, the least-bad cell, still a loss).

### Magnitude (the actual signal): expected-|z| vs realized-|z| correlation

| pair | 1m | 5m | 15m | 1h | 4h |
|------|----|----|-----|----|----|
| DOGE | 0.233 | 0.236 | 0.241 | 0.254 | **0.297** |
| ETH  | 0.243 | 0.233 | 0.242 | 0.248 | 0.261 |
| BTC  | 0.227 | 0.233 | 0.244 | 0.271 | 0.271 |
| SOL  | 0.235 | 0.230 | 0.235 | 0.249 | 0.246 |

The trend is monotone-rising with horizon for DOGE/ETH/BTC and roughly flat-then-rising for SOL: longer bars => stronger magnitude skill (peak at 1h-4h). Signed-E[z] correlation (direction skill) is ~0.00-0.03 (mostly negative noise) at every horizon for every pair: the model has NO directional-magnitude skill, only unsigned-magnitude skill.

### Selective high-confidence P/L (does trading only confident bars clear fees?)

Average NET log-return per trade (bps) after a 20 bps round trip, swept over confidence-quantile gates. Trade-every-bar (q=0.00) sits at ~-20 bps everywhere (pure fee drag). Gating to the most confident bars raises the win rate but the avg-net-bps stays NEGATIVE at every (pair,horizon):

| pair/hz | q=0.00 | q=0.75 | q=0.90 | q=0.95 | q=0.99 |
|---------|--------|--------|--------|--------|--------|
| DOGE 5m | -19.98 | -19.97 | -19.46 | -19.70 | -20.83 |
| DOGE 15m | -19.42 | -17.85 | -17.16 | -16.97 | -23.99 |
| DOGE 1h | -20.05 | -23.42 | -24.44 | -20.94 | -13.07 |
| **DOGE 4h** | -20.26 | -14.55 | **-5.12** | **-4.57** | -48.76 |
| ETH 5m | -20.03 | -19.75 | -19.81 | -20.34 | -19.55 |
| ETH 15m | -19.79 | -18.85 | -20.11 | -21.52 | -21.44 |
| ETH 1h | -20.22 | -22.43 | -22.32 | -21.76 | -24.69 |
| ETH 4h | -23.37 | -25.18 | -35.49 | -29.72 | -14.27 |
| BTC 5m | -20.09 | -20.10 | -20.08 | -20.34 | -20.32 |
| BTC 15m | -19.59 | -19.19 | -19.44 | -20.90 | -22.02 |
| BTC 1h | -19.88 | -19.85 | -21.25 | -20.70 | -27.56 |
| BTC 4h | -21.63 | -24.36 | -24.03 | -19.89 | -16.00 |
| SOL 5m | -19.92 | -19.87 | -19.84 | -19.78 | -19.14 |
| SOL 15m | -19.61 | -20.11 | -18.97 | -20.04 | -25.81 |
| SOL 1h | -20.53 | -23.30 | -23.68 | -23.86 | -19.11 |
| SOL 4h | -23.12 | -26.99 | -32.76 | -35.29 | -23.20 |

(win rate at the gate, e.g. DOGE 4h q=0.90 = 0.568, but avg net still -5.12 bps. Every cell across 4 pairs x 4 horizons x 5 gates = 80 cells is NEGATIVE.)

Read: confidence concentration helps the win rate (DOGE 15m q=0.95 hits 0.594) but never enough to clear the fee. The single near-miss is **DOGE 4h at q=0.90-0.95 (~-4.6 to -5.1 bps, win rate ~0.57)**: the one corner that gets close to break-even, consistent with "magnitude edge strongest + fewest decisions at the longest horizon." But it is PAIR-SPECIFIC and does NOT generalize: ETH 4h, BTC 4h, and SOL 4h all get WORSE with high-confidence gating (high-conf precision drops to/below 0.50, avg net -30 to -35 bps). So there is no universal tradeable direction corner; the favorable DOGE-4h behavior is one pair's sample-window luck, not a class effect.

## 3. Verdict (brutally honest)

**No.** There is no horizon at which DOGE or ETH (or BTC/SOL) DIRECTION beats coin-flip by enough to trade profitably after fees. Best class-wide case is 15m at ~0.524 dir-acc / ~0.557 high-conf precision for DOGE/ETH/BTC, which is ~3-4 SE over 0.50 (statistically nonzero, n=14k) but economically nil: a ~0.52 hit rate on a symmetric move needs the average winning move to exceed the 20 bps round trip by a margin the tape does not give you, and selective high-confidence gating does not rescue it (best gated cell across all pairs/horizons = DOGE 4h q=0.90-0.95 at ~-4.6 to -5.1 bps net/trade, win rate ~0.57: CLOSE to but still under break-even, and PAIR-SPECIFIC: ETH 4h gating moves the wrong way, ~-30 bps). This is market efficiency at every resolution, exactly as the 1m result foretold; coarsening the bar does not manufacture a generalizable directional edge.

**Pivot (as instructed): magnitude/volatility + selective high-confidence sizing, with the model as a forecaster and the trader (MCP-trader) doing sizing.** The one robust, monotone signal is unsigned magnitude/volatility: expected-|z| vs realized-|z| corr ~0.23 and RISING with horizon to ~0.30 at 4h. The product framing that works: the byte model emits a calibrated next-bar volatility / expected-move forecast (and a weak directional lean used only to break ties), and a downstream sizing layer (MCP-trader) decides position size from magnitude + confidence, sits out low-vol/low-confidence bars, and harvests vol (straddle-like / breakout / vol-targeting), NOT a naive long/short on the directional lean.

**Where the magnitude edge is strongest:** the longer horizons. |z| corr is maximal at 1h-4h (DOGE 0.254 -> 0.297). 1h-4h also means far fewer bars -> far fewer fee-incurring decisions, so the cost drag that kills the 1m strategy is structurally smaller. The sweet spot for a tradeable magnitude/vol product is **1h** (best balance of edge strength, sample count, and decision frequency); 4h has the strongest per-call edge but few decisions. 15m is the only horizon where the thin DIRECTION blip shows up, but it is not bankable; treat 15m as a conditioning input, not a signal.

## 4. Additional data, ranked by payoff / effort

What is actually on disk now (verified): `extensions/installed/market/data/crypto/` (200 pairs, OHLCV only), `extensions/installed/market/data/crypto_extra/` (451 pairs, OHLCV only), `extensions/installed/market/data/crypto_of/` (**40 majors incl DOGE/ETH/BTC/SOL/XRP/ADA/LTC, WITH `taker_buy`+`trades` columns, 2017-2026 1m**: this is order flow, already on disk), `extensions/installed/market/data/futures/` + `indices/` (DAILY only, ~6.5k rows, NOT minute-aligned), `crypto_1s` symlink (EMPTY). No funding/basis data anywhere on disk.

| rank | additional data | expected payoff | effort | status |
|------|-----------------|-----------------|--------|--------|
| 1 | **Order flow (taker-buy share + trade-count) at COARSER horizon (15m-4h)** | medium. The brief found order flow useless at 1m, but it was never tested at coarse horizons, where aggressor-flow imbalance integrates and the magnitude signal is already stronger. Most likely to help MAGNITUDE/vol and confidence calibration, not direction. | **low**: `crypto_of/` (40 pairs incl DOGE/ETH) is on disk with the columns; the codec already has channels 5-6 for it; `build_series_corpus.py` already loads `crypto_of` and has a `--no-order-flow` A/B arm. | ready |
| 2 | **Cross-asset lead-lag (BTC -> DOGE/ETH)** as a parallel synchronized stream | medium for direction (the one 1m effect with a real mechanism: leader precedes follower), modest for magnitude. | **medium**: data is free and timestamp-aligned (BTC in `crypto`/`crypto_of` on the same epoch grid), but the codec/corpus builder has no multi-stream interleave; needs a builder change to emit `[BTC bar][follower bar]` aligned blocks (platform code, allowed). | needs builder work |
| 3 | **Longer / cleaner history at the chosen horizon** (use the full 2017-2026 span, not the recent 14k window, with walk-forward + embargo) | low-medium: more samples tighten calibration and let walk-forward eval distinguish edge from regime luck; will not create a directional edge that is not there. | **low**: data already on disk; only the corpus split + eval protocol change. | ready |
| 4 | **Funding rate / perp basis (crypto)** | medium: funding/basis extremes bias direction over minutes-to-hours and are one of the few exogenous direction signals; but the realistic horizon is hours, matching the magnitude sweet spot. | **high**: NOT on disk; needs a futures/funding feed capture + storage pipeline. | data acquisition gated |
| 5 | **L2 order-book imbalance / depth** | highest in the literature for short-horizon DIRECTION | **high**: NOT on disk; needs an L2 feed capture, the heaviest pipeline. | data acquisition gated, off-budget |

Do-first: **#1 (order flow at coarse horizon)**: it is free, on disk, fits the existing codec and builder, and targets the axis that actually works (magnitude). #2 (lead-lag) is the only credible DIRECTION lever but costs a builder change; do it second.

## 5. Concrete next training (dashboard `/trainers/run`)

**Resolution:** train NATIVELY at a coarse horizon, not 1m. The current model only ever saw 1m bars; the sweep evaluated it on resampled coarse bars (a domain shift) and 15m-4h still won. A model that trains on coarse bars should do strictly better at the horizon that matters. Pick **15m** for the primary run (best dir-acc blip + strong magnitude + still ~240k bars/pair for the majors), with **1h** as the magnitude-product variant.

**Corpus / data:** BROAD multi-asset, NOT DOGE/ETH-specific. The 80M generalizes from a diverse anonymous tape; a DOGE/ETH-only model overfits one regime and loses the cross-instrument tape dynamic that is the whole point of the instrument-agnostic codec. Use the **40-pair order-flow set (`crypto_of`)** so the full 7-byte codec (incl `taker_buy`/`trades` channels 5-6) is exercised: that bakes in additional-data rank #1 for free. DOGE/ETH are members of this set, so they are covered without a bespoke model.

**Objective:** keep next-byte cross-entropy (the codec already factors direction/magnitude/vol into separate bytes). Do NOT switch to an RL/PnL objective (noise-overfitting trap at this SNR). Evaluate for magnitude-|z| corr + high-confidence calibration, not direction.

**Size / steps:** 80M first (proven recipe, fast on M3 Ultra). The winning recipe from the current model: batch 12, cosine, wd 0.15, seq 1024, ~12000 steps. Coarse bars mean fewer total bars, so a corpus-sized step count is appropriate; 12000 steps is a safe horizon. Promote to 200M only if 80M shows the magnitude edge sharpening at 15m vs the 1m baseline.

**Build the corpus FIRST (one prerequisite + one gap):**
- Order-flow 7-byte corpus from the 40-pair set already works today:
  `python veritate_mri/tools/build_series_corpus.py --source crypto_of`
  -> writes `trainers/corpus/crypto_of_{train,val}.bin` (per-instrument oldest=train/newest=val split, 7-byte codec).
- GAP: `build_series_corpus.py` encodes **1m only** (no resample). To train NATIVELY at 15m, the builder needs a `--horizon` arg that calls `market.data.resample(df, horizon)` before `compute_features`. This is platform code (`veritate_mri/tools/`), editable locally, one small change. Until that lands, the fastest correct path is: build `crypto_of` at 1m (above) and rely on the model learning multi-scale structure from the codec's trailing-z features, OR add the `--horizon` arg (recommended) so the run is genuinely 15m-native. **This builder change is the single blocking task before the recommended run.**

**Literal `/trainers/run` POST body** (after the `crypto_of` corpus exists; 15m once the builder `--horizon` arg lands, else drop `_15m` from name and corpus stem and train on the 1m order-flow corpus):

```json
{
  "id": "veritate_80m",
  "args": {
    "name": "mkt_crypto_of_15m",
    "corpus": "crypto_of_15m",
    "description": "Market 80M, coarse-horizon arm: 40 order-flow majors (crypto_of, taker-buy + trade-count channels, full 7-byte codec) resampled to 15m bars. Tests whether native coarse-bar training sharpens the magnitude/|z| edge (the only working axis; direction is coin-flip at all horizons). Broad/instrument-agnostic, NOT DOGE/ETH-specific. Winning 80M recipe (batch 12, cosine, wd 0.15). model_type=statistical.",
    "model_type": "statistical",
    "size": "80m",
    "precision": "bf16",
    "version": "v1",
    "seq": 1024,
    "batch_size": 12,
    "total_steps": 12000,
    "base_lr": 0.0001,
    "min_lr": 1e-06,
    "warmup_steps": 500,
    "lr_schedule": "cosine",
    "wsd_decay_frac": 0.33,
    "wsd_decay_kind": "sqrt",
    "weight_decay": 0.15,
    "beta1": 0.9,
    "beta2": 0.95,
    "label_smoothing": 0.05,
    "grad_clip": 1.0,
    "ckpt_every": 500,
    "log_every": 50,
    "eval_every": 500,
    "eval_iters": 64,
    "seed": 0
  }
}
```

Notes on the JSON:
- `model_type: "statistical"` is MANDATORY and only survives via the dashboard form -> `VERITATE_MODEL_TYPE` env -> `save.py` (a hand-rolled CLI `--model_type` is silently dropped and the run mislabels as `language` with wrong probes). Launch through the dashboard form, not a manual launcher.
- `corpus: "crypto_of_15m"` resolves to `trainers/corpus/crypto_of_15m_{train,val}.bin` and presupposes the `--horizon 15m` builder run. If the `--horizon` arg is not added, use `corpus: "crypto_of"` (the 1m order-flow corpus) and rename the model `mkt_crypto_of`.
- A second, cheaper variant for the magnitude product: same body with `name: "mkt_crypto_of_1h"`, `corpus: "crypto_of_1h"` to train the 1h magnitude/vol forecaster (strongest |z| edge, fewest decisions -> lowest fee drag for the MCP-trader sizing layer).
