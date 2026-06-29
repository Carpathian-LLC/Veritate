# Market prediction: more variables, feedback training, multimind reuse

Research synthesis. Not shipped code. Lane: byte-level, energy-efficient, consumer hardware (M3 Ultra / MPS), non-stationary low-SNR market tape.

## Caveat on gating

What I grounded this on instead: the actual codec (`series_codec.py`), corpus builder (`build_series_corpus.py`), market data/serving layer (`market/data.py`, `market/veritate.py`, `market/live.py`), the surviving multimind route stub (only a stale `.pyc` remains; source deleted), and the empirical findings stated in the brief.

---

## Verdicts (lead)

1. **More variables — yes, partially, and cheaply.** A handful of exogenous channels (cross-asset lead-lag, order-book imbalance, funding/basis, session/calendar) plausibly carry next-minute signal that price-of-one-instrument alone cannot. They fit the byte codec as **extra byte-channels per bar** or **parallel synchronized streams**, preserving scale-free + no-lookahead. BUT: most of the easy ones (book imbalance, full order flow) are **not in the current data** (you have OHLCV only) and the highest-signal ones are exactly the ones you don't have. Rank below.
2. **Feedback / punish-when-wrong — yes for the supervised reframe, mostly no for live RL.** Reframing the objective (magnitude + calibration + triple-barrier/meta-labeling) and **walk-forward continual fine-tune on a rolling window** is feasible and on-mission. Live RL with reward = realized PnL is a trap on this SNR/non-stationarity: reward noise dwarfs signal, it overfits the recent regime, and look-ahead leakage is easy to introduce by accident. Feasible: better loss + rolling refit. Aspirational/risky: online RL.
3. **Multimind reuse — mostly no, one idea worth salvaging.** Multimind as built was a **language/chat slot-memory ("hippocampus") + sleep-consolidation PoC** (W3 region naming, W5 sleep-cycle LoRA, W6 slot memory, W6b cross-sample fact recall). The *mechanism* (a retrieval memory of past episodes keyed by current context) maps onto a market predictor only as **analog-regime retrieval** (find past windows that rhyme with now, condition on what happened next). That is a real, mappable idea. The rest of multimind (conversations, fact recall, sleep LoRA on chat) does not transfer. And the source is gone — you would be rebuilding the concept, not reusing the code.

---

## A. More variables: what carries next-move signal, and how to encode it

### The hard constraint first

The codec is **3 bytes per bar**: byte0 = return bucket (33 bins, z-scored over trailing 20 bars), byte1 = range ratio (16 bins), byte2 = volume ratio (16 bins). Vocab 256, next-byte cross-entropy. Everything is **scale-free** (ratios / trailing-z) and **strictly trailing** (no lookahead, enforced in `compute_features` via `_trailing`). Any new variable MUST keep both properties or it breaks the train/serve contract (`series_codec` is shared by corpus build and live serving).

So the encoding question is always: can I express variable X as a trailing-normalized, scale-free, bucketable scalar per bar? If yes, it's a new byte-channel. If it's a separate time series at the same clock, it's a parallel stream.

### Three viable encoding shapes

- **Extra byte-channels per bar (BAR_STRIDE 3 -> 4,5,6...).** Append one byte per new per-bar scalar. Cheapest, keeps one sequence, model learns cross-channel structure for free via attention. Cost: every added byte lengthens the sequence ~33% per channel, so context in *bars* shrinks for a fixed seq_len, and vocab pressure rises (more distinct byte values). Best for 1-3 high-value channels.
- **Parallel synchronized streams (multi-instrument tape).** Encode several instruments on the same clock as interleaved bar-blocks (e.g. `[BTC bar][ETH bar][SPY bar]` per timestep). This is how you get **cross-asset lead-lag** without a new architecture. Cost: sequence length multiplies by stream count; you must guarantee timestamp alignment (resample to a common grid, forward-fill nothing across the prediction boundary). The no-lookahead rule means the "other" streams at time t may only use data closed at or before t.
- **Interleaved context tokens (low-cardinality, slow-moving).** Session/calendar/regime flags that change rarely. Emit them as a small set of marker bytes at block boundaries (e.g. one byte at the start of each session). Cheapest for categorical, low-rate variables.

### Variables ranked by expected signal-per-effort

Effort assumes you must also *acquire* the data (you currently have OHLCV only).

| rank | variable | plausible next-min signal | codec fit | data on hand? | effort |
|---|---|---|---|---|---|
| 1 | **Cross-asset lead-lag** (BTC->alts, SPY/QQQ->single names, sector ETF->name) | high — leader moves precede follower at 1m; this is one of the few genuinely exploitable 1m effects | parallel streams; each instrument already encodes cleanly with the existing 3-byte codec | **YES** (you have 200 crypto pairs + ~500 stocks) | **low** — pure reuse of existing data + codec, just interleave aligned bars |
| 2 | **Realized-vol / vol-of-vol channel** (you already have part of this) | high for magnitude (already your strongest axis, vol_r2 up to 0.71); modest for direction | extra byte: trailing realized vol regime bucket; you already encode range+volume, add a multi-bar realized-vol-z channel | YES | low |
| 3 | **Session / time-of-day / calendar** (UTC hour, weekday, session open/close, weekend for crypto) | moderate — vol and drift have strong intraday/weekly seasonality; helps magnitude + conditioning | interleaved marker bytes at block start; cheap | YES (timestamps already in data) | low |
| 4 | **Order-book imbalance / depth at top N levels** | **highest** of all 1m direction signals in the literature; bid/ask pressure leads short-horizon price | extra byte-channels (imbalance ratio is naturally scale-free) | **NO** — not in OHLCV; needs L2 feed capture | high (data acquisition + storage) |
| 5 | **Trade-flow / signed volume (aggressor side, CVD)** | high for short horizon; directional | extra byte: trailing-z signed-volume channel | **NO** — needs trade-tick or aggTrade feed | high |
| 6 | **Funding rate / perp basis (crypto)** | moderate — funding/basis extremes mean-revert and bias direction over minutes-to-hours | extra byte: funding-z or basis-z, scale-free already | **NO** — needs futures/funding feed | medium |
| 7 | **On-chain (crypto): exchange in/outflows, stablecoin mints** | low at 1m (these move on hour+ scale; SNR at 1m near zero) | extra byte if resampled, but rate-mismatched | NO | medium, low payoff at 1m |
| 8 | **Macro releases / scheduled events** (CPI, FOMC, earnings) | spiky — huge vol at known timestamps, near-zero between | interleaved event-proximity marker byte ("minutes to next scheduled event" bucket) | partially (calendars are free) | medium |
| 9 | **Text / news / sentiment** | low-to-moderate and **off-mission for a byte price model**; latency + non-stationarity + needs an NLP stack you don't want in this lane | does NOT fit the price codec cleanly; would be a separate model | NO | high, off-lane |

**Top of the do-first list:** #1 (cross-asset lead-lag, free, big), #2 (realized-vol channel, free, reinforces your one working axis), #3 (session markers, free). All three use data you already have and the existing codec philosophy. #4/#5 (book + flow) are the theoretically strongest for *direction* but gated on data you must go capture and store — that is the real cost, not the modeling.

**Honest note on direction:** even with #4/#5, sub-minute direction edge is small and gets eaten by ~20 bps round-trip costs (the doc already records this). The realistic win from "more variables" at 1m is **better magnitude/vol forecasting and better conditioning of confidence**, plus a *thin* directional edge from lead-lag — not a flip from coin-flip to reliable. Set expectations there.

---

## B. Feedback / online training: how to "punish when wrong" under non-stationarity

### 1. Fix the objective before anything else (biggest leverage, lowest risk)

Current objective: next-byte cross-entropy over all 3 channels equally. Direction (byte0 sign) is buried in a 33-way classification weighted the same as range/volume bytes. Three concrete reframes, all feasible on-box:

- **Magnitude + calibration head, not pure direction.** You already evaluate magnitude-corr and calibration in `_bench_metrics`. Train to them: weight the loss toward correct |z| (magnitude) and add a calibration penalty (e.g. a proper scoring rule / focal-style down-weighting of confident-wrong). This matches where signal actually exists (magnitude) and where the product already leans.
- **Triple-barrier labeling + meta-labeling (Lopez de Prado).** Instead of "next bar up/down," label each bar by which barrier it hits first within a horizon: up-target, down-target, or timeout (vol-scaled barriers, so it's scale-free and fits the trailing-z philosophy). Then a **meta-label**: given the primary model says "up," a second head predicts *whether to act* (confidence gate). This directly encodes "act only when the edge is real," which is the right framing for a coin-flip-direction asset.
- **Asymmetric / cost-aware loss.** Bake the ~20 bps round-trip cost into the training target so the model is only rewarded for moves big enough to clear costs. Prevents it from learning a "directionally right but unprofitable" policy.

These are all **supervised** — deterministic targets, no live loop, fully testable, low compute. This is the highest-leverage feedback you can add.

### 2. Continual / rolling-window fine-tune (feasible, the right kind of "online")

Non-stationarity means a model trained on 2017-2024 tape is partly stale for this week. The on-mission, low-risk version of "feedback on live data":

- **Walk-forward refit.** Periodically (daily/weekly) fine-tune the existing checkpoint on the most recent closed bars with a small LR, short schedule. Cheap on M3 Ultra at 80M-200M. Keep the broad-mix base (the diverse 80M generalizes; don't let the refit collapse onto one instrument/regime — mix recent + historical).
- **Replay buffer to avoid catastrophic forgetting.** Each refit = recent window + a sample of historical windows. This is the only "memory" you actually need for online adaptation and it's trivial.
- This is **feedback** in the honest sense: the model is corrected by realized outcomes, just batched and offline-per-cycle rather than per-tick. It avoids every RL pitfall below.

### 3. RL framing (reward = realized PnL / correct magnitude): mostly a trap here

Why it's risky on *this* problem specifically:

- **Reward noise >> signal.** At ~coin-flip direction and costs near the edge, realized-PnL reward per step is almost pure noise. Policy gradient on near-zero-SNR reward overfits noise and is sample-inefficient exactly when you have few effective samples (regimes are short).
- **Overfits the recent regime.** RL chases whatever paid in the training window; markets change, the learned policy is stale by deployment. The non-stationarity that motivates feedback is the same thing that breaks RL.
- **Look-ahead leakage is insidious.** Reward shaping, normalization stats, or barrier definitions computed over windows that include the future silently leak. The codec is already careful (trailing-only); an RL reward pipeline is a fresh place to reintroduce leakage.
- **Energy/mission fit.** RL training loops are compute-hungry for what they return here; off-lane for "energy-efficient on consumer hardware."

If you ever do RL: use it only as a **thin policy/sizing layer on top of a frozen supervised forecaster** (the forecaster gives calibrated probabilities; a tiny RL or even a rule maps probability+magnitude to position size), reward = cost-aware realized PnL, evaluated walk-forward. Don't RL the byte model end-to-end.

### 4. Walk-forward eval that won't fool itself

Mandatory regardless of which path:

- **Strict time split, per instrument** (you already do oldest-train/newest-val). For online refit, use **anchored or rolling walk-forward**: train on [start, t], test on (t, t+Δ], roll forward, never test on a bar before its training cutoff.
- **Embargo / purge** around the split boundary (drop bars within the label horizon of the cutoff) so a multi-bar label can't straddle train/test.
- **Costs in the metric** (20 bps round-trip already modeled — keep it).
- **Report magnitude/calibration, not just direction** (the page already does).
- **Multi-seed** for any architecture/channel comparison under 5% delta (agent_roe rule: >=3 seeds; byte-level single-seed deltas are noise).
- **Null baselines:** persistence for magnitude, coin-flip + always-flat for direction. A new channel must beat these *after costs, walk-forward*, or it's noise.

### Feasible vs aspirational

- Feasible now, on-box: objective reframe (B1), rolling refit + replay (B2), the eval protocol (B4).
- Aspirational/risky: end-to-end RL on PnL (B3) — only as a frozen-forecaster + thin sizing layer, and only after B1/B2 are proven.

---

## C. Multimind / memory reuse: what it was, and whether it maps

### What multimind actually was (recovered from the surviving route stub)

Source is **deleted** — only `veritate_mri/routes/__pycache__/multimind_routes.cpython-313.pyc` remains; no docs, no results dir, no model on disk. From its strings, the project was a **language/chat memory PoC**:

- **W3** — region naming via specialty corpora (carve the net into named regions by training on topic-specific data).
- **W5** — "sleep cycle LoRA adaptation" (periodic consolidation: adapt weights during a downtime "sleep" pass).
- **W6** — "slot memory hippocampus PoC" (an external slot/key-value memory the model reads/writes — a hippocampus analog).
- **W6b** — "cross-sample fact recall" (retrieve a fact stored from one sample while processing another).
- Plus recent-context recall, a conversation store, and a `/multimind/sleep/trigger` consolidation endpoint with `bytes_since_sleep` / `next_sleep_eta` bookkeeping.

So: an episodic-memory + consolidation experiment for a **chat** model, not a market one.

### Does it map to a market predictor? Honestly:

- **Maps (one idea worth salvaging): analog-regime retrieval.** The W6 "slot memory + read by current context" mechanism = a market **regime memory**: encode the current trailing window, retrieve the K most similar past windows, condition the forecast on what actually happened after those past windows. This is a real, well-known approach (analog forecasting / nearest-neighbor in embedding space) and it fits the byte model cleanly: the model's own hidden state at the last bar is the query key; the value is the realized next-bar outcome. It's also cheap and on-mission (no extra training loop; it's a retrieval index). **This is the salvageable concept.** Note you'd be re-implementing it, not reusing dead code.
- **Maps weakly: sleep-cycle consolidation (W5)** = the rolling-window refit in section B2 (periodic offline adaptation). Same shape, different dressing. So the *idea* survives but as ordinary continual learning, not as the multimind apparatus.
- **Does NOT map:** conversations, cross-sample *fact* recall (W6b — facts are a language construct), region-naming via topic corpora (W3 — there are no "topics" in anonymous price tape). These are chat-specific.

Bottom line: don't try to "reuse multimind." Reuse exactly one idea from it — **a retrieval memory of analog regimes keyed by the model's own representation of the current window** — and recognize the rest as off-target chat machinery whose code no longer exists.

---

## D. Minimal first experiment (highest leverage, lowest cost)

**Do the cross-asset lead-lag channel.** It is the only top-ranked variable that (a) uses data you already have, (b) fits the existing codec with zero new acquisition, (c) targets the one 1m effect with a real directional basis (leaders precede followers), and (d) is small enough to run in a day.

### The experiment

- **Build a paired/parallel corpus:** for each follower instrument, interleave its 3-byte bars with the 3-byte bars of one obvious leader on the same minute clock (e.g. follower-alt + BTC; or single-name + sector ETF/SPY), strictly aligned, leader bar only using data closed at or before t. Reuse `series_codec` and `build_series_corpus` philosophy — extend, don't fork the format contract.
- **Train two small (80M) byte models, >=3 seeds each:** baseline = follower-only tape (current setup); treatment = follower + leader parallel stream. Same seq_len, same schedule, walk-forward split with embargo.
- **Score with the existing `benchmark` metric set** (directional accuracy, magnitude-corr, calibration, **after-cost** equity) on the held-out newest window.

### Falsifier (state up front)

> Adding the aligned leader stream does **not** improve held-out, walk-forward, after-cost directional accuracy by more than the across-seed noise band (and does not improve magnitude-corr / calibration), versus the follower-only baseline.

If the lead-lag channel can't beat its own seed noise after costs, "more variables" is not the lever at 1m and you should redirect effort to the magnitude/calibration objective reframe (B1) instead.

### Wall-clock estimate (M3 Ultra, 80M, CPU/MPS as available)

- Corpus build (reuse of existing data, alignment + interleave): ~1-3 h.
- 2 conditions x 3 seeds x 80M short training: this is the variable cost; per the project's measured 80M throughput, budget **~1-2 days** total if trained to the same point as the current 80M, less if you use a shorter walk-forward window. Day-class to small week-class.
- Eval: minutes (benchmark already exists).

It is the smallest thing that genuinely tests "more variables + a feedback-shaped split beats price-only," using a variable with a real mechanistic basis and no new data pipeline.

---

## One-paragraph honest summary

You can add variables and they'll help **magnitude/volatility and confidence calibration** — your already-working axis — and a **thin** directional edge from cross-asset lead-lag, which is free to try and is the right first experiment. You cannot, on OHLCV alone at 1m, turn direction from a coin flip into a reliable signal; the variables that would (order-book imbalance, signed trade flow) are exactly the ones you don't yet have, and even those yield a small edge that costs erode. "Punish when wrong" is best done as a **better supervised objective (magnitude + calibration + triple-barrier/meta-labeling) plus rolling walk-forward refit with a replay buffer**, not live RL — RL on this SNR/non-stationarity is a noise-overfitting trap. Multimind doesn't transfer except for one idea: a **retrieval memory of analog regimes** keyed by the model's own window representation, which you'd rebuild (the code is gone). Start with the lead-lag channel; let its after-cost, walk-forward, multi-seed result decide whether "more variables" is the lever at all.
