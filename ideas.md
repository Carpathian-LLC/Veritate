# ideas

Open ideas and active campaigns. Mechanism + falsifier only. Ideas graduate to `successes.md` or `failures.md` when their falsifier resolves.

## index

One line per idea; sections below hold mechanism + falsifier. Numbering gaps are graduated ideas — the ledgers hold their outcomes.

- IDEA 1 — distilled core + external memory + recurrent working state — open
- IDEA 2 — hierarchical dual memory at trillion-char scale — open; drill-down prototype graduated (successes.md), RARS revival + re-ranker remain, background pass parked
- IDEA 3 — cut FLOPs/token (MoE / dynamic patching / mixture-of-depths) — open; MoE deferred to an 80M A/B (failures.md)
- IDEA 4 — corpus style bake-off (raw / filtered / textbook / Q&A) — open, not started
- IDEA 6 — under-training explains non-fluency (chin200m at 20 tok/param) — closeout pending; run ends ~2026-08-12
- IDEA 7 — unbounded talk length + edge-box KV cache — open; prerequisite for IDEA 17
- IDEA 8 — smallest strong conversationalist, bind-then-read + private-doc product — open
- IDEA 9 remnant — value/aversion module — parked, no falsifier
- IDEA 10 — byte n-gram speculative decode — open; gated on mean accept length ≥2.5
- IDEA 11 — chat quality set by corpus ratio, not steps — blocked: format-adherence eval set does not exist (failures.md)
- IDEA 12 — aspect ratio as CPU throughput lever — measured; quality falsifier (second seed) pending
- IDEA 13 — device-aware shape selection in the trainer — blocked on IDEA 12
- IDEA 14 — Monarch-factored FFN — open; blocked on an efficient kernel, then equal-wall-clock quality arm
- IDEA 15 — beyond-RAM training on external NVMe — queued: the next run's test, small shape first
- IDEA 16 — co-tenant training (GPU split across concurrent runs) — open
- IDEA 17 — long-form multi-turn SFT from the rebuilt corpus — corpus built (successes.md); pack + SFT + falsifier remain, gated on IDEA 7

## the goal (2026-08-10)

**Fluency, not intelligence.** The target model does not need to answer correctly. It needs to (a) produce words that make sense, (b) hold a conversation for many turns without cutting off or degenerating, and (c) do that on a WELL-trained base. Factual accuracy and reasoning are explicitly out of scope — role binding is already a documented architectural wall at 122M/200M/800M (failures.md), so correctness is not what this size buys. Measure runs against coherence and sustained length, not accuracy.

## frontier chat system

**IDEA 1 — small distilled core + trillion-char external memory + O(1) recurrent working state**
Core ≤1.5B active distilled from a frozen teacher; knowledge externalized into IDEA 2's memory; the hybrid trunk's recurrent state as cross-turn working memory. Falsifiers: beats bare core and window-RAG on needle QA at equal wall-clock; distilled core clears a chat bar over chat80m; grounded vs open-domain accuracy reported separately.

**IDEA 2 — hierarchical dual memory (compressive gist + addressable exact recall) at trillion-char scale**
Byte-native keys learned with the LM, hierarchical drill-down index for exact recall, recurrent state as always-on gist. Falsifiers: recall flat as the corpus grows 1e9→1e12; grounded-QA non-decreasing with memory growth; learned drill-down beats flat IVF at equal latency, else ship flat ANN.
- Gated-slot RARS revival: K=4 learned-gated overwrite slots. Kill line: beat top-1 prefix-inject by >5pt on natural queries at 200M, else RARS stays dead.
- Re-ranker: rescue the recall@5 0.63 / recall@1 0.37 gap. Kill line: re-ranked top-1 beats raw top-1 by >5pt on held-out natural queries.
- Background "thinking" pass over retrieved context before answering — unshaped, parked.

## throughput and flops

**IDEA 3 — cut FLOPs/token; the only door open on this box (no tensor cores)**
T0: profile the dominant cost first. T1 (MoE, `trunk=hybrid_moe`, wired): matched-active-FLOPs must beat hybrid by >5pt val at equal wall-clock; blockers: trainer/exporter mismatch, engine refuses top-k=2, no batched MoE prefill. T2 (entropy-based dynamic patching): >5% tok/s at equal val on matched 10M runs. T3 (mixture-of-depths): matched val at >10% fewer FLOPs/token — lowest priority. T4 (patching scale curve at 121.8M, run before T2). Decision: ≥1.82x holds → fund T2; 1.2-1.82x → restate headline; <1.2x → correct the composed-stack claim.

**IDEA 12 — aspect ratio is a real but small CPU throughput lever; the machine is already near its roofline**
Depth is a serial dependency and width is a bigger matmul, so at equal parameter count the two are not interchangeable. Measured on cardinal (i7-9700T, 800 MHz clamp, 8 threads, vanilla trunk, seq 256 b8), params matched within 2% and FLOPs within 8%, against cardinal's sysprobe peak of **166.7 GF/s**: at 5m, 12L h180 **80.4 GF/s (48% of peak)** / 6L h256 **101.1 (61%)** / 3L h360 **115.8 (69%)** / 2L h444 **117.3 (70%)**, best-vs-convention **1.23x**. At 25m the effect shrinks: 32L h256 **103.6** / 8L h512 **127.9 (77%)** / 2L h1024 **141.6 (85%)**, best-vs-convention **1.15x**. Bigger matmuls already saturate the machine, so the deeper the shape the more headroom shape recovers, and the larger the model the less there is to recover.
Honest boundary: the same sweep on a loaded M3 Ultra CPU read 2.44x at 85M. That did not transfer. Aspect-ratio numbers are per-box and must be measured on the target, never carried over.
Falsifier: quality. Matched-param arms trained to equal steps on real text; a wider-shallower arm must hold val within the 5% seed rule against the convention shape to bank the speedup, second seed required. If val degrades with depth, report the val-per-wall-clock crossover and pick the shape there, not the fastest one.

**IDEA 13 — device-aware automatic shape selection in the trainer**
Trainer picks its aspect ratio from the device it resolved rather than from a fixed `trainer_sizes.json` preset: same target parameter count, shallower-and-wider on CPU, convention depth on GPU. `hardware.pick_device()` and `/trainers/sysprobe` (peak GF/s, core count, copy GB/s) already supply everything the decision needs. Blocked on IDEA 12's quality falsifier — a shape switch that costs val is a regression no matter what it does to tok/s. Do not implement until matched-param arms clear the seed rule; if they do, the lever is worth 1.15-1.23x on cardinal-class hardware and nothing on a GPU.

**IDEA 14 — Monarch-factored FFN removes 16x of the work and the kernel gives back only 1.19x; the prize is locked behind efficiency, not math**
Total training cost is ~6*params*tokens. Conditional compute is dead for training (MoE, PKM) and tokens are a data question, so the remaining lever is fewer params per token at the SAME hidden dimension. Square-block Monarch (two block-diagonal factors over transposed axes) costs H*(s1+s2) against dense H^2 with identical activation shapes. Measured on an IDLE cardinal, 8 threads, fwd+bwd, `veritate_core/model_monarch.py`: single map H=1024 **89.1 ms dense vs 66.5 Monarch (1.34x)** at 2048 tokens; full FFN 1024->4096->1024 **710.1 vs 595.0 (1.19x)** at 2048 tokens and **186.2 vs 202.6 (0.92x)** at 512 tokens.
The gap that matters: dense reaches **145 GF/s (87% of cardinal's 166.7 peak)**, Monarch reaches **10.8 GF/s (6.5%)**. The factorization removes 16x of the arithmetic and the implementation loses 13x of it back to permutes, `.contiguous()` copies and many small bmms. A kernel at dense's efficiency would be worth far more than 1.19x; the math is not the constraint, the memory movement around it is.
Boundary: needs ~2048 tokens per call to pay at all, and the patched trunk's global blocks see only seq*batch/PATCH_STRIDE slots, so seq 1024 b8 lands in the winning region and seq 256 b8 does not. All contended measurements are void: with a training run on the other 8 threads, dense slows 2.6x and Monarch 30x, which inverts the result entirely. Measure structured layers on an idle box or not at all.
Falsifier: quality per wall-clock, not per parameter. The FFN holds ~16x fewer parameters, so it must train against a dense arm at equal wall-clock (not equal steps) and match val within the 5% seed rule; second seed required.

**IDEA 16 — co-tenant training: split the GPU across concurrent small runs; aggregate tok/s is the open headroom (2026-08-12)**
Single-run levers on M3 Ultra total 1.19x combined (successes.md 2026-08-03), so single-run throughput is closed. The open lever is aggregate: two concurrent runs on separate Metal command queues, optionally plus a CPU arm on the otherwise-idle cores, may sum to more than sequential when one small run does not saturate the GPU. Natural payload is the seed pairs the 5% seed rule already requires — a concurrent seed A/B halves an experiment's wall-clock even at modest aggregate gain.
Falsifier: two matched ~100M runs concurrent vs the same two sequential. Bank only if aggregate tok/s ≥1.2x sequential AND each concurrent arm's val curve tracks its solo twin — contention that costs val is a regression, not a speedup.
Boundary: contention inverts benchmarks entirely (IDEA 14: dense slowed 2.6x, Monarch 30x under a co-running trainer). While co-tenant runs are live, no other measurement on the box is valid.

## beyond-ram scale

**IDEA 15 — beyond-RAM training: explicit streaming tier on external NVMe, never OS swap (queued: validate on the next run, small shape first) (2026-08-12)**
Target: train shapes whose params + grads + optimizer footprint exceeds unified memory. OS swap is the wrong mechanism — page-granularity random I/O, no prefetch control, jetsam under pressure, MPS buffers effectively wired. The mechanism is explicit streaming to an external Thunderbolt 5 NVMe volume (~5-6 GB/s per enclosure, stripeable, and replaceable — petabyte-class optimizer/grad writes stop landing on the soldered internal SSD). Step 1, near-zero code: point `PagedAdamW` `state_dir` at the external volume. Step 2: a planner tier past `page_optimizer_to_nvme` that streams params/grads layer-by-layer with prefetch (ZeRO-Infinity pattern, single box). Stream cost is a fixed ~30 B/param/step independent of batch, so a large enough batch re-hides I/O behind compute.
Falsifier (the queued next-run test, small shape first): matched A/B at a shape that fits in RAM (~100-200M), RAM-resident AdamW vs forced-streaming arm, same seed. The streamed arm must match the loss curve within tolerance and hold amortized step time within 10% of the resident arm at its best batch. Pre-gate: sustained enclosure read/write bandwidth measured under MPS load; kill below ~5 GB/s. Only after the small-shape pass does a super-large run get scheduled.
Boundary: this widens the I/O lane only. Wall-clock to a proper token budget scales ~P² at the measured ~20 realized TFLOP/s, so the ceiling for a fully-trained model on this box is ~2-3B regardless of streaming. eGPU is not a successor — Apple-silicon macOS has no eGPU support at all; more compute means another machine via `/mesh/*`, a different idea.
Coupling: the same next-run campaign stacks IDEA 7 Track A (rolling-window KV + `state_carry=chunks`) and IDEA 17's long-form multi-turn SFT — scale, unbounded context, and sustained conversation are one test matrix, and IDEA 17 already names IDEA 7 as its prerequisite.

## corpus style

**IDEA 4 — which byte corpus style wins: raw / filtered / textbook / Q&A-interleaved**
Four matched bins, same 200M recipe, rank on held-out val + code evals. Falsifier: a style must beat mixed_raw by >5% val bpb or a clear code-eval margin, second seed required. A 200M winner picks the family, not final ratios — revalidate at 1-3B before farm scale.

## chinchilla-scale bet

**IDEA 6 — is under-training, not architecture, why no 200-270M run has been fully fluent?**
chin200m (270.5M) to 20 tok/param (5.4B tokens) vs prior 7-9. Falsifier: val at step 55,000 materially below chat200m's 0.812 at matched effective data. chin200m@55000 is already the SFT-campaign base — outcome may just need formal closeout.

**Purpose of the wren_sft / chin200m run (in flight, ends ~2026-08-12):** it is a *pretraining continuation*, not an SFT, despite the folder name. Description: "Vocabulary continuation… adds 15.3 GB to reach 20 tok/param = Chinchilla for 270.5M at the measured 4.55 bytes/token. Pretraining resumes (loss_mask off) with instruct held at 0.05 as replay so the measured 43.9% form obedience is not eroded." Its job is the "WELL-trained base" half of the goal and to close IDEA 6. Its chat success criterion is *do not erode* — a 16-prompt A/B at 65k vs 95k came out a wash, which is the run passing, not failing (failures.md). Val 0.7798 (65k) → 0.7286 (132.5k), still dropping in WSD decay. Chat ability comes from a separate SFT afterward, not from this.

## unbounded context

**IDEA 7 — unbounded talk length + a KV cache that fits a $300 edge box**
Track A: rolling-window KV for the 4 local-attention blocks + `state_carry=chunks`. Falsifier: generate 8x seq with no bpb cliff at the wrap point. Track B: KV fp16→int8, ring+sink eviction. Falsifier: greedy parity holds AND p50 ms/byte improves or footprint drops ≥2x with bpb delta <0.005.

## green-ai product

**IDEA 8 — smallest strong conversationalist, all facts external; attack the role-binding wall directly**
Bind-then-read: a discrete slot/register layer between retrieval and the LM body, supervised on grounded_v3. Kill line: lifts OBJECT-role accuracy off the ~0% floor without breaking slot copy.
Product P1 (committed): on-device private-document assistant. Falsifier: retrieval gates re-run with paraphrased queries sharing no token with the gold chunk.
Product P2: local-first with cloud fallback, router gated on retrieval score, never LM self-judgment. Falsifier: escalation precision/recall on a mixed set.
Below 122M is untested — minimum viable reader size unknown.

## developmental training

**IDEA 9 remnant — value/aversion module**
A small separate network learns "avoid this output" and modulates decoding. No falsifier defined. Parked.

## speculative decode

**IDEA 10 — byte n-gram speculative decode: the only lossless lever left on bandwidth-bound hardware**
Draft K bytes by longest-suffix match against prompt+generated text, verify in one batched weight stream, accept the longest matching prefix (greedy output byte-identical). Non-boundary decode already at 83% of cardinal's bandwidth ceiling. Gates: mean accepted length ≥2.5 → ship; 1.5-2.5 → ship only if ms/byte improves; <1.5 → failures.md, retry once an MTP-head draft source exists. Report accept length by traffic type, not just the mean.

## corpus mix

**IDEA 17 — long-form multi-turn SFT: install sustained conversation from the rebuilt corpus (2026-08-10)**
The brevity diagnosis of the old chat bin (median assistant turn 14 B, 2.4 turns/conv) and the rebuilt no-decay corpus answering it (median 267 B, no decay through turn 7) are banked in successes.md 2026-08-10. Open work: pack the corpus into a bin turn-boundary-aligned at the current `seq`, then SFT with **loss_mask on** (assistant turns only, unlike the current run) at a high chat ratio.
Falsifier: on held-out multi-turn prompts the SFT'd model must raise median generated reply length and sustain ≥6 coherent turns without degenerating (loop, topic-collapse, or premature `<|im_end|>`), judged blind against the current base. Correctness explicitly not scored. If replies stay short after training on long-form data, brevity is coming from the base or the decode path, not the SFT corpus — check EOS calibration before building more data.
Coupling: turn lengths already brush the ceiling (max 1509 B vs `seq` 1024 B), so anything past ~1 KB of context is truncated today. Long-form SFT and IDEA 7 (unbounded context) are the same project — a bin with long conversations is untrainable at seq 1024, so IDEA 7's rolling-window KV + `state_carry=chunks` is a prerequisite, not a follow-on.

**IDEA 11 — is chat quality set by corpus ratio rather than steps? (from the wren_sft wash, failures.md)**
wren_sft ran 70% general web / 20% chat+instruct. Over 30,000 steps val improved 0.0142 while format adherence flat-to-down and only factual recall gained. Hypothesis: instruction-format adherence is ratio-bound, not step-bound, and further steps at this mix cannot buy it. Test: matched 200M runs at chat+instruct 0.20 (control) vs 0.50, equal total steps, same seed. Falsifier: the 0.50 arm must beat control by a clear margin on a held-out format-adherence set (answer-length limits, "only the word X", list-of-N) with aggregate val allowed to be *worse* — if aggregate val is the only thing that moves, the hypothesis is dead. Needs the format-adherence set built first; it does not exist yet (see failures.md eval-data entry).