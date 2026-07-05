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

What is actually on disk now (verified): `extensions/installed/market/data/crypto/` (200 pairs, OHLCV only), `extensions/installed/market/data/crypto_extra/` (451 pairs, OHLCV only), `extensions/installed/market/data/crypto_of/` (**40 majors incl DOGE/ETH/BTC/SOL/XRP/ADA/LTC, WITH `taker_buy`+`trades` columns, 2017-2026 1m**: this is order flow, already on disk), `extensions/installed/market/data/futures/` + `indices/` (DAILY only, ~6.5k rows, NOT minute-aligned), `crypto_1s` symlink (EMPTY). Funding data now on disk: `extensions/installed/market/data/funding/` (40 Binance USDT pairs, 8h rates, 2020-01 to 2026-05-31).

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

## 6. Carry re-validation (2026-07-02): falsifier fired

Delta-neutral carry (long spot + short perp, collect funding) re-validated on current data. Smoke: `SMOKE_RESULTS/carry_revalidation_smoke.py` -> `carry_revalidation_stats.json`.

- Backtest (40 Binance funding CSVs; weekly top-K rotation by trailing 7d mean funding, positive-only gate, half capital as margin buffer, 4-leg rotation fees), %/yr on capital, K=3 at 2bp/leg: 2020 14.0, 2021 27.5, 2022 1.6, 2023 4.3, 2024 6.6, 2025 2.2, 2026 YTD 1.2. At 5bp/leg 2026 YTD is negative. Static BTC-only 2026 YTD 0.4. Worst 30d stretch across configs: -0.25 to -1.5% of capital (tail risk small; the yield is the problem).
- Hyperliquid 30d realized funding (annualized, on notional, 2026-07-02): BTC 5.7%, ETH 3.2%, SOL -2.0%, DOGE 1.5%; best liquid alts LIT 17.3%, FARTCOIN 11.0%, LINK 8.8%. Live snapshot rates pinned at the HL 10.95%/yr baseline are uninformative; realized is the anchor.
- $10k deployed today, top-3 rotation, half margin buffer: net on capital 5.5%/yr maker ($1.51/day) or 3.8%/yr taker ($1.04/day) vs T-bill 4.5% ($1.23/day), and only by holding meme-grade alts whose spot leg may need a second venue (basis + execution risk unpriced).
- Verdict: the 2024->2026 funding compression persists. Net carry is at or below T-bill yield; carry is not currently worth the venue risk. Retry condition: 30d realized funding on liquid majors sustained above ~15%/yr (bull-regime funding).

## 7. Cost-killed families retest at perps costs (2026-07-02): still no survivor

Four previously-killed families rerun as perps long/short at the 2026 US-legal cost stack (2.6/3.5/5bp per side swept, realized 8h funding paid long / received short). All rules fixed on pre-2024 train; scored 2024-2026-05 OOS. Survivor bar: positive net 2024 AND 2025 AND 2026YTD at 3.5bp, Sharpe >= 0.8, maxDD < 40%. Smoke: `SMOKE_RESULTS/costkilled_retest_smoke.py` -> `costkilled_retest_stats.json`. Zero survivors, but the autopsy changed:

- **Funding is no longer the killer.** Realized funding on the momentum L/S book was +1.85%/yr on test (short leg collected more than long paid). The old assumed 10%/yr drag was wrong.
- **XS momentum L/S (1d, L=7d)**: 2.6bp test CAGR 16.3% Sharpe 0.78 maxDD 27%; died on 2025 (-8.3%), not cost. **1w variant: positive all three years at every fee (2024 +1.8 / 2025 +16.7 / 2026 +0.2 at 3.5bp) but Sharpe 0.44** — closest thing to alive; fails the Sharpe bar.
- **1h XS reversal (fade top-decile 3-bar movers)**: gross edge is REAL OOS (+121%/yr, Sharpe 1.36 at zero fee; random-sign null -7.4) but turnover 10,443x/yr -> break-even 1.16bp/side all-in. Dead at 2.6bp (-85%/yr). Died purely on cost; retry condition: all-in cost < ~1bp/side (maker-rebate execution model, different problem).
- **Daily RSI<25/>75 mirror L/S**: train Sharpe negative for every hold (-1.3 to -1.5); short-overbought leg -94% on test. Signal-dead; the old long-only bounce does not survive a mirror short leg.
- **Seasonality**: nothing passes |t|>=3 on train (max 2.60 across 31 candidates). Task-spec top-3 tested anyway: combo Sharpe -0.27 at 3.5bp. Signal-dead, not cost-dead.
- **Vol-expansion breakout (1h)**: train per-trade edge NEGATIVE (-18.4bp, t=-13.4, n=22k — train says fade, not follow); test flipped to +4.2bp/trade (t=2.2), portfolio Sharpe 0.67 at 2.6bp with all years positive. Unstable sign across regimes -> not deployable on the pre-registered rule; break-even ~5.6bp/side on the test edge anyway.
- Verdict: at true 2026 costs the fee/short-leg excuses are gone and the families still fail — reversal on microstructure cost, the rest on signal. Retry conditions: reversal if all-in < 1bp/side; 1w momentum if a regime filter lifts Sharpe above 0.8 without touching test data.

## 8. Maker-execution test of the 1h reversal (2026-07-02): falsifier fired — the wall is adverse selection, not fees

Thesis tested: mean-reversion entries are maker-compatible (fading a crash = bidding into selling flow), so passive limit execution at ~0-1bp/side should clear the 1.16bp/side break-even from section 7. Rules train-fixed from `costkilled_retest_smoke.py` (fade top-decile 3-bar |ret| movers, 40 Binance USDT majors, hold 1 bar), test 2024-01..2026-05. Passive-fill sim: limit at signal-bar close, works the full next hour; the 1m sim reduces exactly to hourly extremes (constant level, hour-boundary fallback; re-verified vs raw 1m, 0 mismatches). Fill at level iff extreme penetrates x bp (x = 0/2/5 swept); unfilled entry: arm A skips, arm B takers at next hourly open +2.6bp; exit limit at hold-bar close, same rule, taker fallback. Smoke: `SMOKE_RESULTS/reversal_maker_exec_smoke.py` -> `reversal_maker_exec_stats.json`. Zero survivors.

- **Fill-selection eats the entire edge, and then some.** Entry fill rate 98.8% at x=0 — but the ~1.2% of unfillable signals carry the whole alpha: missed trades average +121bp (long) / +126bp (short) close-to-close, filled trades +0.6 / -0.5bp. All-trades gross +123%/yr collapses to **-2.1%/yr on the entry-filled set alone** — before exit drag, before any fee, at the optimistic touch-fill bound. The profitable reversal is precisely the one that never retraces to the signal close.
- **The exit is adversely selected too**: passive exit misses exactly when the position keeps moving against the book; exit drag -1.7bp/trade at x=0 (-5.7 at x=2). Net at ZERO maker fee, arm A x=0: -86%/yr, Sharpe -1.7. Queue-honest x=2bp: filled-set gross itself is -3.4/-4.7bp per trade, net Sharpe -9 to -10. Arm B (taker fallback) captures nothing: fallback entries average -2.0bp gross — the bounce completes inside the hour that cannot be entered passively.
- **Leg decomposition (0-fee, close-to-close)**: long (fade-crash) Sharpe 1.38, short 0.54; long carries 2024 (+361%, 3.5bp/side gross), short carries 2025; both die in 2026 (gross/side 0.44bp long, -0.14bp short — below any fee even at closes). The close-to-close edge is decaying across the test years independent of execution.
- **Long-only spot (Binance.US 0bp maker), the maximally-executable variant**: arm A x=0 at 0 maker: -50%/yr, Sharpe -0.73 (2024 +38%, 2025 -76%, 2026 -43%). Dead.
- **10% per-name cap**: same shape, still ruinous (arm A x=2 maker 0: -97%/yr). **Capacity (moot)**: traded names median $8.5M daily volume in 2026; 1% of entry-hour volume caps the account at median ~$13k (uncapped) / ~$33k (capped) — a $10k-scale strategy even if the edge existed.
- **Exploratory combo** with 1w XS momentum: daily corr 0.023 (genuinely uncorrelated) but the reversal book at any executable config is deeply negative, so the combo is moot (Sharpe -7 at pre-registered weights).
- Verdict: **passive execution does NOT rescue reversal; the 1bp wall from section 7 was never a fee wall — it is an adverse-selection wall.** "Maker-compatible mean reversion" is false at 1h: limit orders fill on the continuation, skip the bounce. Retry condition: none via execution; only a different signal whose alpha survives conditioning on retrace-to-entry (i.e. positive filled-set gross) is worth re-testing.

## 9. Pre-registered regime conditioning of 1w XS momentum (2026-07-02): bar cleared on the letter, evidence too weak to call it deployable

Section 7's retry condition tested: train-only regime filters on the 1w XS momentum L/S (40 majors, 7d lookback train-fixed, quintile L/S equal-weight, realized funding, 2.6/3.5/5bp per side). Multiple-testing denominator: 9 variants examined on pre-2024 train (XS-dispersion>trailing median, BTC 30d vol above/below trailing 365d median, BTC above/below 100d MA, factor 90d trailing mean>0, funding-positive-share above/below median, 30d vol-targeting capped 2x); all thresholds fixed a priori, none swept. Selection pre-registered before any test scoring: vol_scaled always + top-2 filters by train Sharpe at 3.5bp; only those 3 scored on 2024-2026-05 test. Smoke: `SMOKE_RESULTS/xsmom_regime_filter_smoke.py` -> `xsmom_regime_filter_stats.json`.

- **No filter lifts train Sharpe.** Baseline train 1.04; best filter 0.68; vol-scaling 1.05 (+0.008). In-sample, regime conditioning adds nothing; selection was rank-among-filters, not improvement-over-baseline.
- Selected: vol_scaled + btc_vol_low + btc_vol_high. The two vol signs were train-indistinguishable (0.680 vs 0.671) and filled both filter slots, so the test set adjudicated a train coin-flip.
- **btc_vol_low (trade only when BTC 30d vol < trailing 365d median) clears the full bar**: test Sharpe 0.93, CAGR 13.2%, maxDD 10.1%, per-year 2024 +9.2% / 2025 +16.1% / 2026YTD +6.4%, halves 0.76/1.11, fee-robust (Sharpe 0.95/0.93/0.89 at 2.6/3.5/5bp), 46% time invested, turnover 37x/yr (baseline: 0.44 Sharpe, maxDD 29%, 80x). Complement btc_vol_high: test Sharpe -0.25. vol_scaled FAILS (0.33; 2024 and 2026 negative): the standard momentum-crash fix hurts this factor.
- **Placebo null undercuts the pass**: 451 circular-shifted gates (same on-fraction and block structure) clear the FULL survivor bar 10.4% of the time alone and 21.5% under the actual either-complement exposure; observed Sharpe sits at the 92nd percentile of shifted gates. Family-wise luck probability of producing this "survivor" with zero information: roughly 1 in 5.
- For it, not just against: the winning direction matches the momentum-crash literature (crashes cluster in high-vol regimes), the weak stretch (H1-2024) improves most, the complement is decisively negative, and idle cash (54% of test time) would add ~2%/yr of unmodeled T-bill yield.
- Verdict: **bar cleared on the letter by btc_vol_low, but with zero train-side support and a ~20% placebo pass rate this is a test-set-generated hypothesis, not a validated edge. Not deployment quality.** Retry condition: btc_vol_low gating must hold on data none of this work touched: >=6 forward/paper months at a Sharpe-consistent rate, or replication on an untouched universe/venue. No further filter search on this test window; the window is spent.

## 10. marketof order-flow A/B, finally scored (2026-07-02): flow channels add nothing to direction

The A/B trained 2026-06-14 (`marketof_80m` real taker-buy/trade-count channels vs `marketof_noflow_80m` constant
fallback channels, identical bars, stride 7, 12k steps) was never evaluated; the val losses (1.30 vs 0.82) are not
comparable because the noflow corpus has two trivially-predictable constant bytes. Common-metric scoring on the
newest ~15k bars per pair (BTC/ETH/SOL/DOGE/XRP/LTC at 1m/15m/1h, CPU, prob-mass direction, return-byte-only CE):
`SMOKE_RESULTS/marketof_ab_score_smoke.py` + `marketof_ab_score_stats.json`.

| axis | flow better in | mean delta (flow - noflow) | read |
|---|---|---|---|
| directional accuracy | 7/18 cells | -0.003 | no lift; if anything worse |
| high-conf precision | 6/18 | -0.005 | no lift |
| magnitude abs-z corr | 15/18 | +0.002 | consistent sign, under the 0.02 noise bar |
| return-byte CE | 6/18 | -0.000 | flat |

Best after-fee cells exist only at 1bp/side with the 0.95 confidence gate (LTC 1h flow +3.5bp/trade, n=359) —
isolated cells out of 144, same sub-cost wall as every classical result. Verdict: real order flow in the codec does
NOT improve the byte model's direction; the 80M byte model matches classical GBM on the same features (AUC ~0.52).
The binding constraint is the signal content of public tape at retail cost, not model architecture. Do not spend
further training compute on codec-channel variants chasing direction.

## 11. Daily ML panel + adversarial audit (2026-07-04): first full survivor — h7 HGB long/short CONFIRMED, live forward arm shipped

The daily-horizon completion of the ML ladder: pooled panel over the 40 crypto_of majors, daily UTC bars 2017-11..2026-05, 28 features (multi-window vol-normalized returns, vol ratios, range/volume expansion, RSI, distance to 30/90d hi/lo, funding level + 7d change, cross-sectional 7/30d ranks, BTC-residual 7d z, day-of-week), labels = next-{1,3,7}d direction/magnitude, walk-forward yearly 2022-2026 (train = everything before the test year minus a horizon purge), quintile L/S overlapping 1/h tranches, 2.6/3.5/5bp per side + realized funding. Models: ridge logit, HGB classifier, HGB regressor vs plain mom7. Smokes: `SMOKE_RESULTS/daily_ml_panel_smoke.py` -> `daily_ml_panel_stats.json`, hardening in `daily_ml_audit_smoke.py` -> `daily_ml_audit_stats.json`.

- **Panel verdict: h7 is the only ML horizon that survives, and it beats momentum.** hgb_cls h7 LS at 3.5bp: net/yr 2022 +22.7% / 2023 +9.7% / 2024 +23.8% / 2025 +4.7% / 2026 +21.0%, pooled Sharpe 1.27 (hgb_reg 1.16, mom7 0.85, logit -0.52 dead). AUC only 0.51-0.56 — the edge is cross-sectional rank quality, not per-name prediction. h1/h3 die to turnover; shuffled-label and random-rank nulls are flat-to-negative.
- **Adversarial audit: CONFIRMED on every arm** (hgb_cls, seed-mean): baseline 15.2%/yr Sharpe 1.19; **entry lag +1 day 12.9%/yr Sharpe 1.04, 5/5 years positive** (the decisive execution-robustness test — plain mom7 FAILS the same lag at Sharpe 0.77); quantile 0.15/0.25 pass (17.0%/14.0%); 14d purge passes (14.3%, kills a 7d-label leak hypothesis); funding on/off immaterial (~1%/yr, funding_off passes alone); real seed variance (per-seed lag1 mean Sharpe 0.97-1.15, single seed-years to -0.08) — the 3-seed MEAN is what was validated. hgb_reg also CONFIRMED (lag1 19.4%/yr Sharpe 1.50) but shipped second: classifier chosen for the live arm as the pre-registered primary.
- Standing caveats, none testable offline: 2026-majors universe (survivorship, ambiguous sign for L/S), 2026 partial year, h7 selected as 1-of-3 horizons, backtest = upper bound.
- **Live forward arm SHIPPED (2026-07-04): `ml7` in the Trading extension** (`extensions/canonical/trading/server/ml7_trader.py` + `ml7_features.py`; contract in `developer_documentation/architecture/backend/trading_ml7.md`). Frozen 3-seed HGB ensemble (mean predict_proba) trained on the full panel through 2026-06-26 (85,778 rows, fit 87s), exact feature parity enforced by a byte-equality test against the smoke's builder, 7 daily-staggered tranches (one rebalanced per day after UTC close = lag1 timing by construction), 3.5bp/side, funding omitted (documented), $10k simulated, BTC benchmark. Forward record starts 2026-07-04; the survivor bar for the forward test is the audit's lag1 expectation (~13%/yr, Sharpe ~1.0), judged in months. Retrain cadence: quarterly via `--train`.

## 12. Daily pairs stat-arb with a real short leg (2026-07-03): falsifier fired — cointegration is unstable OOS, and the pool is evaporating

The one never-tested gap in the pairs/ratio family: proper DAILY stat-arb with a real short leg at perps costs (prior kills were long-only spot or 1h sub-fee). 40 Binance USDT majors, yearly walk-forward, formation on expanding train only: Engle-Granger log-price cointegration (OLS + no-constant ADF on residuals, AIC lag 0-4, MacKinnon N=2 5% CV; implementation property-tested: 0% false-reject on random walks, 100% on AR(0.9)) AND spread half-life 2-30d. z vs train-window mean/sd (picked in-sample on the pre-2022 window over rolling-90d, Sharpe 4.28 vs 2.37 at 3.5bp); enter on |z| cross of 2, exit |z|<0.5 / 30d timeout / |z|>4 stop, dollar-neutral legs, 1/10-capital slots cap 10, realized funding both legs, 2.6/3.5/5bp per side per leg. Smoke: `SMOKE_RESULTS/daily_statarb_smoke.py` -> `daily_statarb_stats.json`. No survivor.

- **The cost hypothesis was right, and it doesn't matter.** Turnover 1-14x/yr makes fees irrelevant: full-period Sharpe 0.171 / 0.163 / 0.149 at 2.6 / 3.5 / 5bp. The daily cost wall is gone; there is no edge behind it. This kill is signal, not cost — same autopsy as section 7.
- **Cointegration instability, measured directly.** Year-over-year selection persistence 39% -> 34% -> 23% -> 17%; the qualifying pool collapses 31 / 38 / 22 / 6 / 3 pairs (2022-2026) while raw EG passes stay at 45-72 — the half-life gate kills them, median selected half-life drifting 16.5 -> 27.7d against the 30d cap. Selected spreads stop mean-reverting OOS: timeouts are 70-80% of exits every year, z-convergence exits only 4-11/yr.
- Per-year net at 3.5bp: 2022 +8.7%, 2023 -17.2%, 2024 +18.5%, 2025 -2.8%, 2026 +1.4%. Passes the years prong (3/5 incl 2026) and maxDD (20.9%), fails Sharpe 0.163 << 0.8. By 2025-2026 the book is 4-7% deployed (nothing qualifies) — the strategy can no longer hold capital even if the edge existed.
- **Shuffled-entry nulls (same pair/count/duration, random timing+sign): observed inside null noise every year** — pctile 0.79 / 0.07 / 0.94 / 0.25 / 0.96. 2023's -17% is WORSE than random timing (7th pctile); 2026 (96th) is 2 trades. z-timing carries no information beyond pair exposure.
- **Beta-hedged daily dispersion (7d residual vs BTC, decile L/S, the low-turnover cousin of the dead 1h reversal): dead on arrival.** Sharpe -0.73 / -0.81 / -0.93 at 2.6 / 3.5 / 5bp, maxDD 83-87%, negative 4/5 years at every fee, turnover 266x/yr. Observed sits at the 81st pctile of random-pick nulls paying the same fee model: the residual-reversal gross signal is a rounding error on the fee drag.
- Caveats: dollar-neutral legs (β in signal only), fixed 1/10 slots (no 1/N resize churn billed), no intra-trade compounding of leg notionals, force-close at year end, funding zero before a pair's 2020+ history.
- Verdict: **family closed, now on its own terms.** The short leg and the low-turnover cost stack were the last two excuses; with both granted, in-sample cointegration does not survive out-of-sample in crypto (the documented risk, now measured: <40% persistence, pool -> 3 pairs) and the timing adds nothing over shuffled entries. Retry condition: none within this universe/frequency; only a structurally different spread anchor (e.g. same-ecosystem token pairs with a fundamental link) would be a new hypothesis, not a retest.

## 13. Long-only cross-sectional equity momentum vs SPY (2026-07-03): survivor bar passed on the letter, but the null quantifies the survivorship inflation

Canonical 12-1 momentum (skip last month, top decile equal-weight, long-only, monthly rebalance), fixed a priori from the literature, zero tuning, on ~495 current S&P 500 members, Yahoo daily adjclose (split+dividend adjusted) 2000-2026, costs 3bp/side all-in (2bp + 1bp spread), benchmark SPY total return. Smoke: `SMOKE_RESULTS/equity_momentum_smoke.py` -> `equity_momentum_stats.json`.

**Survivorship caveat first, it governs everything below.** Delisted ex-members are unobtainable free (Yahoo purges delisted symbols: probed TWTR/ATVI/SIVB/PXD/MRO/HES/X/CTLT all dead; Stooq per-ticker is JS-walled, bulk db 401), so the panel is current constituents only, which backfills winners (2023 momentum picks include APP/COIN/DASH/PLTR/SMCI/TTD: in the panel because they later entered the index). Bias direction: inflates every strategy number below; positive results are an upper bound, the negative reads are the strong ones.

- **vs SPY, the bar passed**: net beat-years 7/10 (2016-2025; lost only 2016/2021/2023, also beat in 2015 and 2026 YTD), full-period (2001-02..2026-07) CAGR 19.3% vs 8.85%, Sharpe 0.875 vs 0.540, maxDD -56.9% vs -55.2% (gap 1.7pp, within 10pp). 2-week variant nearly identical (Sharpe 0.861, 7/10). Top-100-by-dollar-volume subset (top-25 picks, less backfill-sensitive): Sharpe 0.696, 7/10, still above SPY.
- **The random-decile null is the honest yardstick**: 20 seeds of random top-decile-sized portfolios, same dates/sizes/costs, return **CAGR 13.7% (range 11.4-15.6), Sharpe 0.74: a coin-throwing monkey beats SPY by ~4.9pp/yr on this panel.** That gap IS the survivorship bias, measured. Momentum's defensible increment is over the null, not over SPY: +5.6pp CAGR and +0.135 Sharpe, momentum above all 20 nulls. And even that is an upper bound: backfilled winners land disproportionately in the top momentum decile, so the bias hits momentum harder than random.
- **Costs are a non-issue at this frequency**: one-way turnover 3.77x/yr -> ~23bp/yr total drag. Unlike every crypto family killed in sections 6-10, monthly equity momentum's problem was never fees.
- **Per-year net, momentum vs SPY**: 2015 +12.0/+1.2, 2016 +4.9/+12.0, 2017 +25.2/+21.7, 2018 +4.1/-4.6, 2019 +40.8/+31.2, 2020 +41.3/+18.3, 2021 +23.6/+28.7, 2022 **-1.5/-18.2** (end-2021 book had rotated into energy: DVN/COP/EOG/OXY/FANG, the mechanism is real), 2023 +19.1/+26.2, 2024 +46.2/+24.9 (SMCI/PLTR/APP backfill territory: treat with suspicion), 2025 +21.9/+17.7, 2026 YTD +31.6/+9.8.
- **Crashes behave as the literature says**: 2020-02-19..03-23 momentum -37.2% vs SPY -33.4% (crashes slightly harder), 2009 rebound lags (+23.0 vs +26.4). **15%-vol-target overlay (cap 1x, cash at 0%) is the documented fix and works here**: maxDD -31.1% (halved), Sharpe 0.924 (up), 2020 crash -21.3%; price is CAGR 14.1% and only 4/10 beat-years: it improves risk-adjusted, not the beat-SPY count. Rolling 3y excess vs SPY: positive 93% of windows, min -6.9pp, median +8.7pp.
- **Verdict:** the published post-2000 momentum-decay consensus is NOT confirmed on this data, but this data cannot cleanly refute it either. What survives honest accounting: (1) the momentum increment over a survivorship-matched null (+5.6pp/yr, 20/20 nulls beaten) is real on this panel; (2) costs at 2026 retail are negligible for this factor; (3) the vol overlay does in equities what it failed to do in crypto. What does not survive: any claim that "momentum beat SPY by 10pp/yr": roughly half of that gap is demonstrably the panel, and the recent-decade beat-years lean on names that are in the universe because they won.
- Retry condition for a clean answer: point-in-time constituent lists + delisted-name prices (CRSP-style), which are not free; or forward paper-tracking of the live rule (universe frozen today, no backfill): the only $0 way to measure the factor at today's costs without the bias.

## 14. Fading detected pumps on the shortable perp universe (2026-07-03): falsifier fired — the reversion edge does not exist where shorting exists

The 2026-06-26 probe measured real negative drift after dual-spike detections (-0.43% 1h to -1.32% 24h gross, P(mean<=0)=1.0) on Binance.US mid-caps ranks 50-400 — but those books were dead ($13-19/24h) and unshortable. Both constraints lifted (US-legal perps 2.6-4.5bp/side; OKX lists 394 USDT perps), so the short side got its test: detection replicated exactly from `extensions/canonical/trading/server/scanner.py` on hourly bars (z=(x-mean)/pop-std over 14-bar trailing baseline, min 5 samples, price_z>=2.5 on the 1-bar return AND vol_z>=3.0 on hourly quote volume, 6h cooldown, $50k 24h-vol floor; thresholds pre-registered by the shipped scanner, untuned). Universe = top-160 OKX USDT perps by 24h volume (the shortable set; LUNA dropped, no Binance history) mapped to Binance UM perps; data = data.binance.vision monthly 1h klines + fundingRate 2023-01..2026-06 (3455 kline + 3449 funding zips, 2.48M bars, max gap 0.07%). Smoke: `SMOKE_RESULTS/pump_fade_smoke.py` -> `pump_fade_stats.json`. 22,762 events, 160/160 symbols flagged, median 24h volume at detection $71M.

- **The sign flips on the shortable universe: detected pumps CONTINUE, they do not revert.** Pooled gross forward returns from detection close: +0.05% (1h), +0.11% (4h), +0.18% (12h), +0.15% (24h), +0.30% (48h), +0.47% (72h), every one with bootstrap P(mean<=0) <= 0.006. Per-year: 2023, 2024, 2026 all continuation at 24-72h (+0.26% to +1.11%); only 2025 shows reversion (-0.23% 24h P=0.98, -0.48% 48h P=0.997). The prior probe's negative drift was a property of dead-book unshortable names, not of pump detection.
- **Liquidity and reversion are mutually exclusive, measured directly**: the LARGE volume tercile shows the strongest continuation (+0.36% 24h, +0.61% 48h, +0.73% 72h, all P<=0.001); small tercile weakest and statistically marginal. Meme names continue like non-meme (+0.48% vs +0.27% at 48h). The scanner recipe is scale-free, so on liquid perps it selects momentum, not manipulation.
- **Strategy net (short at first close after detection, 24h exit picked on 2023-24 train where both 24h/48h lost >100% of book, 1/10 slots cap 10, +15% stop, 4.5bp/side, realized funding, $1M liquidity floor)**: 2023 -72.8% book (n=2272, Sharpe -1.60), 2024 -30.4% (n=2376, -0.57), 2025 +18.9% (n=2800, Sharpe 0.29, maxDD -104.7%), 2026 -45.6% (n=1662, -1.46). Test years: one positive, one deeply negative. Survivor bar failed on both prongs that matter (2025 AND 2026 positive: no; Sharpe>=1: no).
- **The null kills even the 2025 positive.** Exposure-matched random entries (same per-symbol-year counts, same costs/stop/funding): per-trade strategy vs null band — 2023 -0.32% vs [-0.48,-0.26]; 2024 -0.13% vs [-0.40,-0.13]; 2025 +0.07% vs [-0.10,+0.15]; 2026 -0.27% vs [-0.27,-0.03], i.e. inside the band in 2023/2025, a hair above p95 in 2024, and BELOW p5 in 2026 (detection timing is anti-selective there). 2025's profit is short-beta on a falling alt market, not the signal.
- **Funding is a cost, not a credit, in the test years**: shorts on flagged names paid on average -0.10%/-0.12% per 24h trade in 2025/2026 (2-3x the whole 9bp fee load); coverage 98.6-99.5% of trades. The worst events stack stop + funding: 0G 2025 -24.1% per unit (-15% stop plus ~-9% funding on a fresh listing), TRB 2023 -17.8%.
- **Squeeze tail measured**: MAE during holds p50 3.3%, p90 13.6%, p95 16.7%, p99 23.8%, max +195.7%; 8.4% of trades stopped; zero open-gaps through the stop on hourly bars (wick fills at the stop level assumed; real slippage would be worse). 1/10 slot sizing contains the single-event tail (-2.4% of book worst) — the only survivor-bar prong that passed, and it is irrelevant given the sign.
- Caveats: universe frozen at today's OKX listing (collapsed names absent — also unshortable today, so this matches the executable forward strategy; it does understate fade P/L on names that later died); median symbol history 19 months (2023-24 panels tilt to older, larger names); hourly-bar mapping of the 5-min live scanner is the pre-registered 2026-06-26 probe recipe, and the live events.jsonl population (OKX spot incl sub-$1M books) skews smaller than this panel — the smallest live-flagged names are exactly the ones with no perp.
- Verdict: **falsifier fired, structurally.** The mean-reversion edge lives only in books too dead to short; where perps exist, the same detection selects continuation, funding runs against the short, and timing adds nothing over random shorting of the same names. The radar stays a radar. Retry condition: a forward events.jsonl accumulation of several hundred live-flagged names that DO have listed perps with measured negative 24-48h drift — i.e. evidence the live population behaves like the 2026-06-26 mid-caps rather than like this panel — would justify one re-run on that forward dataset; nothing in the historical data does.
