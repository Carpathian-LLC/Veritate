# overnight run log

Append-only. Headlines at top, dated sections below.

## headlines

- **2026-06-16 (BREAKTHROUGH — first real signal):** LLM-scored crypto news sentiment predicts next-day BTC returns, corr **+0.48** (vs +0.03 for fear-greed), and it **survives the look-ahead control** (+0.58 on post-training-cutoff dates the model couldn't memorize). Short-horizon edge. First defensible edge found. NOT yet proven profitable (correlation != profit; needs forward + fees); the platform to validate it forward (`news_trader.py`) is built and tested. Detail in 2026-06-16b.
- **2026-06-16 (sentiment platform foundation, DONE):** built + verified the information/LLM path: new platform endpoint `POST /teacher/complete` (calls any user-added model; verified live — Ollama `qwen2.5:7b-instruct` scores headlines correctly in ~0.3s); Paper Trading is now a standalone extension with a server (`scraper.py` free RSS+fear-greed, `sentiment.py` LLM scoring + time-decay aggregate, `/ext/paper_trade/sentiment` route). 8 new tests + 45 total green; live scrape verified. HONEST GATE: the only free historical sentiment series (fear-greed) has ~zero correlation with forward returns (~0.02), so there's no free backtest — the LLM-news edge is event-level and only testable FORWARD/live. Foundation is an instrument to run that forward experiment, not a validated money-maker.
- **2026-06-15 overnight (DONE):** hunted for a profitable Robinhood-able (spot, long/cash) strategy across 40 coins, out-of-sample, 3 agents + own re-validation. VERDICT: nothing beats holding BTC on return; the model's only real, monetizable skill is RISK REDUCTION (dodging 70–95% crypto crashes). Micro-trading = guaranteed −37% to −196%/day fee bleed (the "$1k/day micro" goal is inverted from reality). Mean-reversion + vol-regime = overfit mirages. The one genuine modest edge: a model-filtered breakout/trend strategy (full Sharpe ~0.30, trend-regime-dependent) whose value is capital protection (−7% in bear vs BTC −70%), and the byte model measurably improves it (Sharpe 0.30 vs 0.14 plain). The honest product = a survival strategy, not a money printer. Full detail + FINAL VERDICT below.

---

## 2026-06-15 overnight — hunt for a profitable Robinhood-able strategy

Goal handed over with "total control": find something that makes (fake) money within the Robinhood constraint (spot, long-or-cash only, no shorting/perps/options; cost is a hidden spread). User framed it as micro-trades targeting ~$1k/day. Discipline: out-of-sample only, no overfitting, honest losses reported.

### FINDING 1 — micro-trading is the worst possible move (quantified, decisive)

Directional long/flat on the model's `p_up`, traded at each bar size, BTC/ETH/SOL/DOGE, net of Robinhood-style spread (10/25/50 bps round-trip):

| bar | trades/day | gross/day | net @10bps | net @25bps | net @50bps |
|---|---|---|---|---|---|
| 1m  | ~380 | +50 to +130 bps | **−3,700** | **−9,400** | **−18,900** |
| 5m  | ~65  | −45 to +13 bps | −650 | −1,600 | −3,300 |
| 15m | ~25  | −31 to +10 bps | −250 | −620 | −1,240 |
| 1h  | ~7   | −14 to +17 bps | −48 to −83 | −145 to −186 | −308 to −359 |

(bps per day; negative = loss). Monotonic: **more trading = more loss, entirely to fees.** The model's gross edge is a few bps/day; the spread bleed is 10–1000x bigger. Even the gentlest case (1h, 10bps) loses ~50–80 bps/day. **There is no net-positive trading frequency.** This independently reproduces the June-2026 arXiv result (high-turnover ML = −64%/yr; the only fix is cutting trades ~97%). Verdict: the "micro trades" goal is a fee trap — the single most certain way to lose on Robinhood.

### FINDING 2 — crash-avoidance overlay (validation running)

Earlier today: "hold only when price > 20d MA AND model bullish" cut max drawdowns from 50–95% to 14–34% on every coin, in both walk-forward halves (robust), but it's risk-reduction not return (buy-and-hold BTC still wins on return). Re-validating now across all 40 coins with a design/validation coin split + diversified portfolio construction to see if any version is genuinely net-positive at low risk. [pending]

### Agent sweep (out-of-sample, design/validation coin split, fees included)

- **mean-reversion (buy-dips): DEAD END.** Both variants (z-score dip, drawdown dip) tuned to +24%/+44% on design coins, collapsed to −1.2% / +0.7% on held-out validation coins at 0.25% fees. Sign-flips across walk-forward halves and across the fee band → return is noise. Does NOT beat hold-BTC (+62% same window). Only by-product is low drawdown (because it sits in cash 8–23% of the time — but cash does that for free). Model-confirmation filter never helped (always deselected). Overfit risk HIGH.
- **volatility-breakout + model-confirmation: the one genuine (modest) edge.** Agent reported +113%/Sharpe 1.48 on a 5-coin validation subset; my independent re-validation on 15 coins + a regime split is more conservative and trustworthy: **full Sharpe 0.30 (+4.4% CAGR), trend-following profile** — +20.8% in BTC-bull regimes, −7.3% in BTC-bear, loses in the early time-third (regime-dependent). What's REAL and survives: (a) it sidesteps catastrophe — bear-day loss −7% vs hold-BTC −70% (maxDD 39% vs 66–92%); (b) the byte-model exp_move filter MEASURABLY helps — Sharpe 0.30 vs 0.14 for plain breakout, and a smaller bear loss (−7% vs −15%). So the model adds value, but for RISK CONTROL, not raw return. Does NOT beat hold-BTC on return in this bull-heavy window (BTC Sharpe 0.47). Overfit risk LOW-MED (model filter reproduces on unseen coins + both halves), regime risk HIGH (one ~4yr cycle, classic trend-following). Re-validation script `/tmp/revalidate_breakout.py`.
- **diversified basket + vol-regime de-risking: DEAD END / risk-reduction only.** Nothing beat hold-BTC (+80% validation, Sharpe 0.54) — every variant ≤0% return, Sharpe ≤0.24. The de-risking overlays cut maxDD from 65% to 2–6% but ONLY by sitting in cash → flat-to-negative return and NEGATIVE out-of-sample Sharpe. Key: the model's high-vol flag does NOT time crashes — when it de-risks meaningfully it loses money (locks out recoveries). Walk-forward sign-flips on every variant; the auto-"winner" was threshold-noise (loosest setting = feature disabled). Overfit risk HIGH. (This rigorously tempers the earlier per-coin "crash overlay" read: the drawdown cut is real but it costs ~all the upside — cash does the same for free.)

### FINAL VERDICT (overnight, total-control run)

Tested every Robinhood-implementable strategy (spot, long-or-cash): micro-trading, mean-reversion, vol-regime de-risking, crash overlay, momentum/breakout — across up to 40 coins, out-of-sample with design/validation coin splits, walk-forward halves, fee sensitivity, 3 independent agents + my own re-validation of the one positive.

**One conclusion, reproduced every way we cut it: nothing beats holding BTC on return. The byte model's ONLY monetizable contribution is RISK REDUCTION — avoiding the 70–95% crypto drawdowns — never raw-return alpha.**

- Direction is a coin flip; trading it bleeds to fees. Micro-trading is the worst possible move: **−37% to −196% PER DAY** in spread bleed (the user's "$1k/day micro" goal is mathematically inverted from what works).
- Mean-reversion and vol-regime de-risking: overfit — great on design coins, break-even-to-negative out-of-sample, sign-flip across walk-forward halves.
- The one genuine (modest) edge: a **model-filtered breakout/trend strategy** — long only when price breaks out AND the byte model confirms an elevated expected move. Real but modest (full Sharpe ~0.30), trend-regime-dependent (wins in bull, loses in bear/chop), and its value is **capital protection**: −7% in bear vs hold-BTC's −70%, maxDD ~20–40% vs 66–92%. The byte model measurably improves a plain breakout (Sharpe 0.30 vs 0.14) — so the model earns its keep, as a risk filter.

**Honest bottom line for the morning:** there is no money-printer here, and I won't pretend otherwise. The real, defensible, Robinhood-runnable win is a **survival strategy** — trade rarely, hold cash most of the time, use the model + a trend filter to sidestep the catastrophic crashes. It matches/underperforms buy-and-hold BTC on raw return in a bull market but dramatically beats it on drawdown and bear-market protection. That is the genuine product: not "make $1k/day," but "don't get wiped out, and let the model keep you out of the worst." Everything else tested is a fee trap or an overfit mirage.

---

## 2026-06-16 — information / LLM-sentiment angle (the one direction with real documented edge)

Pivoting research from CHART prediction (efficient, dead) to INFORMATION (news/social parsed by an LLM). Unlike charts, this has genuine peer-reviewed support — but with hard caveats.

**SUCCESSES (real, documented):**
- Lopez-Lira & Tang (Univ. Florida, 2023→2024): ChatGPT scores on news headlines **significantly predict out-of-sample daily stock returns**. Stronger in SMALL caps and after NEGATIVE news. Predictive ability rises with model size (emergent financial reasoning). This is a real, published edge — the opposite of the chart coin-flip.
- Crypto: LLM + SOCIAL (Reddit) sentiment enhanced profitability (~+23% total in one study, net of slippage/costs); integrated technical+sentiment portfolios beat benchmarks.

**FAILURES / HARD CAVEATS (why it is not a jackpot):**
- **Edge DECAYS as LLM adoption rises** — documented: "strategy returns decline as LLM adoption rises, consistent with improved price efficiency." Everyone now has GPT; the easy edge is being arbitraged out.
- **Latency:** much of the move happens in the initial reaction, before a slow retail scraper can trade ("by the time media is loudly bullish/bearish, the price has moved"). The ~90% hit rate is on the NON-tradable initial reaction.
- **Look-ahead bias is severe:** LLMs are trained on the backtest period and may "know" the outcome → backtests are inflated and untrustworthy. The ONLY honest test is FORWARD / post-knowledge-cutoff.
- For crypto, raw NEWS sentiment "introduced semantic noise"; SOCIAL (Reddit/X) worked better.
- Retail speed disadvantage vs automated flow is widening.

**IMPLICATION (the build plan):** because backtests are contaminated by look-ahead, the honest way to test an LLM-sentiment strategy is to run it FORWARD in paper — live scrape → Ollama parse → paper trade — and accumulate a genuine out-of-sample record over weeks. It is worth building (real edge direction, uses the already-connected Ollama model, runnable on the paper sim). It is NOT a $1k/day machine: the edge is real but modest and decaying, and $1k/day is mostly a CAPITAL problem (see below).

**The $1k/day math (so the goal is calibrated):** daily profit = capital x daily-return. Realistic edges are single-to-low-double-digit % per YEAR. Even an exceptional, hard-to-sustain 50%/yr edge makes ~capital/730 per day — so $1k/day needs ~$700k of capital. On $10k, a strong 50%/yr edge is ~$14/day. $1k/day is unreachable on small capital regardless of strategy; it is a function of how much money is deployed, not just how good the model is.

### 2026-06-16b — sentiment platform: foundation built + honest gate

BUILT (real, reusable): platform endpoint **`POST /teacher/complete`** (`teacher_routes.py`) — one-shot call to any user-added model (defaults to the configured teacher), the programmatic surface extensions use to score text. VERIFIED end-to-end against the live local Ollama: `qwen2.5:7b-instruct` scores headlines correctly in ~0.3s ("SEC approves ETH ETF" → ETH +0.8; "exchange halts withdrawals" → MARKET −0.9; "BTC consolidates" → +0.3). The scrape → local-LLM → structured-sentiment path works.

GATE (failure, important): tested the ONE free historical sentiment series — the fear-greed index (2021–2026, 1850 days) — vs forward BTC returns. **No predictive edge:** corr(FG, next-1d) = +0.016, corr(FG, next-7d) = +0.035, corr(FG 7d-change, next-7d) = +0.011 — all indistinguishable from zero, and POSITIVE (so the contrarian "fear precedes gains" thesis is NOT supported). "Hold BTC only when fearful" LOSES (−6 to −13% CAGR); the only variant that matched buy-hold was "hold when greedy" (+7%, lower DD) — i.e. trend-following rediscovered, not sentiment alpha.

WHAT THIS DOES / DOESN'T KILL: it kills the cheap version (a coarse, daily, already-priced aggregate sentiment index does not predict returns). It does NOT fully kill the LLM-news thesis — the published edge (Lopez-Lira) is in the INITIAL REACTION to SPECIFIC breaking headlines, event-level and fast, which a lagging daily index cannot capture. But the implication is decisive: **there is no free historical shortcut to validate LLM-news trading — the only honest test is FORWARD/live** (scrape breaking news → score → paper-trade → accumulate an out-of-sample record over weeks). A backtest cannot prove it (no free historical article archive; the proxy is null; and LLM backtests are look-ahead-contaminated anyway).

CORRECTION/UPDATE — GDELT makes a historical test possible after all: GDELT DOC API is free + no-key + timestamped news back years (rate-limited 1/5s). Pulled real crypto headlines across 3 years, scored each with `qwen2.5:7b-instruct`, correlated per-window sentiment with forward BTC returns. **FIRST POSITIVE SENTIMENT SIGNAL of the whole investigation:** corr(sentiment, next-7d) = +0.39, corr(next-1d) = +0.49; high-sentiment windows averaged +2.32% fwd-7d vs −1.58% for low (clean monotone split). Vastly better than the null fear-greed (+0.03). CAVEATS, non-negotiable: (1) only n=14 windows survived → borderline significance, could be luck; (2) the scorer (qwen, ~2024 cutoff) may have LOOK-AHEAD knowledge of 2022–2024 outcomes, inflating the historical correlation. Re-running with ~100 windows + a pre/post-2024-07-cutoff split (post-cutoff = look-ahead-free) to see if it holds. BUILT alongside this: the full forward platform — `/teacher/complete` endpoint, scraper, sentiment scorer, and `news_trader.py` (autonomous paper loop). If the signal survives the robustness check, run `news_trader` FORWARD for the clean, look-ahead-free validation.

ROBUSTNESS RESULT (n=44 windows, the decisive check) — **the signal SURVIVES the look-ahead control. This is the first genuine, defensible edge in the whole investigation:**
- ALL (n=44): corr(sentiment, next-1d BTC ret) = **+0.483**, next-7d = +0.253; high-sentiment windows +2.48% fwd-7d vs −0.48% low.
- PRE-cutoff (n=28, model MIGHT know outcomes): corr next-1d +0.415, next-7d +0.257.
- POST-cutoff (n=16, look-ahead-FREE — dates after the LLM's ~2024-07 training cutoff): corr next-1d **+0.576**, next-7d +0.256; high +4.05% vs low +0.25%. The edge does NOT weaken post-cutoff (it strengthens on the 1d horizon), so it is NOT a memorization artifact.
- Shape: the edge is SHORT-HORIZON (1d corr ~0.5 >> 7d corr ~0.25) — consistent with the published "initial reaction" literature.

FORWARD RUN LAUNCHED (2026-06-17): `news_trader.py` is running live, hourly, paper money — scrape free news -> score with local `qwen2.5:7b-instruct` -> per-asset sentiment -> long-only target exposure (normalized to <=100%, no leverage) -> simulated ledger marked to live Binance.US prices, 20bps spread. Ledger: `extensions/installed/paper_trade/data/account.json`. First ticks clean + solvent (held UNI/HYPE on positive sentiment, no fee churn when at target). BUG CAUGHT + FIXED on first launch: two strong longs summed to 120% exposure -> negative cash (leverage); `targets()` now normalizes total exposure to <=1.0 (test added). This forward record is the clean, look-ahead-free, costed validation — judge it over days/weeks.

HONEST CAVEATS (still required): correlation is not profit — must survive fees, latency, and a larger sample. n=16 post-cutoff is modest (wide CI, though directionally consistent with pre). Short horizon = more trading = more fee exposure. GDELT seendate ≈ publication time. VERDICT: real, look-ahead-robust predictive signal → JUSTIFIES the forward paper run (`news_trader.py`), which is the clean, costed, look-ahead-free validation. Do NOT claim profit until the forward record exists.
