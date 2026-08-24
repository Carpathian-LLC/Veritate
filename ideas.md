# ideas

Open ideas and active campaigns. Mechanism + falsifier only. Ideas graduate to `successes.md` or `failures.md` when their falsifier resolves.

## index

One line per idea; sections below hold mechanism + falsifier. Numbering gaps are graduated ideas — the ledgers hold their outcomes.

- IDEA 1 — distilled core + external memory + recurrent working state — open
- IDEA 2 — hierarchical dual memory at trillion-char scale — open; drill-down prototype and re-ranker graduated (successes.md), RARS revival remains, background pass parked
- IDEA 3 — cut FLOPs/token (MoE / dynamic patching / mixture-of-depths) — open; MoE deferred to an 80M A/B (failures.md)
- IDEA 4 — corpus style bake-off (raw / filtered / textbook / Q&A) — open, not started
- IDEA 7 — unbounded talk length + edge-box KV cache — Track A arm 1 graduated 2026-08-19 (state retention learnable, cliff 1.44 -> 1.30); gap remains, escalation + streaming generation loop next; prerequisite for IDEA 17
- IDEA 8 — smallest strong conversationalist, bind-then-read + private-doc product — open
- IDEA 9 remnant — value/aversion module — parked, no falsifier
- IDEA 10 — byte n-gram speculative decode — gate measured 2026-08-18: grounded traffic clears (2.6-6.5 accept), plain chat marginal; engine implementation remains, RAG-path first
- IDEA 11 — chat quality set by corpus ratio, not steps — unblocked 2026-08-18: the format set exists (`ifeval form`, 280 obedience-only items, deterministic checkers); needs two matched 200M runs, user to size the compute
- IDEA 12 — aspect ratio as CPU throughput lever — measured; quality falsifier (second seed) pending
- IDEA 13 — device-aware shape selection in the trainer — blocked on IDEA 12
- IDEA 14 — Monarch-factored FFN — open; blocked on an efficient kernel, then equal-wall-clock quality arm
- IDEA 15 — beyond-RAM training on external NVMe — queued: the next run's test, small shape first
- IDEA 16 — co-tenant training (GPU split across concurrent runs) — open
- IDEA 17 — long-form multi-turn SFT from the rebuilt corpus — corpus built (successes.md); pack + SFT + falsifier remain, gated on IDEA 7
- IDEA 18 — agentic retrieval: the model calls Carpathian's API as a tool — open; blocked on the endpoint contract
- IDEA 19 — train the no-repeat behavior into the weights — mechanism 1 (guard-distilled SFT) killed 2026-08-18 (failures.md); unlikelihood loss and DPO remain open
- IDEA 20 — persistent memory: tell wren once and it never forgets (three-timescale program) — THE research focus; E1 dose killed, E1b delta-alone killed BUT state became content-bearing (anticopy 0.06 -> 0.47); E1c (delta + recall curriculum) running; E3 serving/persistence + experience log + sleep tooling shipped
- IDEA 21 — grow the trained net instead of retraining: function-preserving expansion of wren to the 500M flagship — opened 2026-08-20 on user directive

## the goal (2026-08-10)

**Fluency, not intelligence.** The target model does not need to answer correctly. It needs to (a) produce words that make sense, (b) hold a conversation for many turns without cutting off or degenerating, and (c) do that on a WELL-trained base. Factual accuracy and reasoning are explicitly out of scope — role binding is already a documented architectural wall at 122M/200M/800M (failures.md), so correctness is not what this size buys. Measure runs against coherence and sustained length, not accuracy.

## frontier chat system

**IDEA 1 — small distilled core + trillion-char external memory + O(1) recurrent working state**
Core ≤1.5B active distilled from a frozen teacher; knowledge externalized into IDEA 2's memory; the hybrid trunk's recurrent state as cross-turn working memory. Falsifiers: beats bare core and window-RAG on needle QA at equal wall-clock; distilled core clears a chat bar over chat80m; grounded vs open-domain accuracy reported separately.

**IDEA 2 — hierarchical dual memory (compressive gist + addressable exact recall) at trillion-char scale**
Byte-native keys learned with the LM, hierarchical drill-down index for exact recall, recurrent state as always-on gist. Falsifiers: recall flat as the corpus grows 1e9→1e12; grounded-QA non-decreasing with memory growth; learned drill-down beats flat IVF at equal latency, else ship flat ANN.
- Gated-slot RARS revival: K=4 learned-gated overwrite slots. Kill line: beat top-1 prefix-inject by >5pt on natural queries at 200M, else RARS stays dead.
- Re-ranker: graduated (successes.md 2026-08-17): embedding re-rank of BM25 top-5 collapsed to k=1 took paraphrased-query precision@1 0.297 -> 0.500; natural queries flat. Serving must be self-contained (no ollama, 2026-08-18), so the shippable form is the platform's own trained key head as the re-rank scorer; the +20pt is the bar it has to match.
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

## unbounded context

**IDEA 7 — unbounded talk length + a KV cache that fits a $300 edge box**
Track A: rolling-window KV for the 4 local-attention blocks + `state_carry=chunks`. Falsifier: generate 8x seq with no bpb cliff at the wrap point. Track B: KV fp16→int8, ring+sink eviction. Falsifier: greedy parity holds AND p50 ms/byte improves or footprint drops ≥2x with bpb delta <0.005.
Baseline measured 2026-08-18 (wren1_0@1250, hansard_val 9x1024 B, teacher-forced): full-context slide is flat (0.89-0.95 bpb) but costs one full forward per byte; the window walk (`forward_streaming`, one forward per window, the only serving shape that scales) pays **1.44 bpb in the first 64 B after a wrap** (+62%) decaying to 1.01 late-window. Carried state contributed exactly nothing: `s` arrived at 1e-15 (GLA decay, carry-off training).
Arm 1 result 2026-08-19 (wren1_3, graduated to successes.md): retention IS learnable -- state absmax 1e-15 -> 1.27, stream beats stream0 everywhere, wrap bucket 1.44 -> 1.30, mid-window 0.99 ~ slide 0.91, chat held. Remaining gap: wrap bucket 1.30 vs slide 0.89. Escalation, one variable at a time: (1b) longer adaptation / higher carry dose (arm 1 was only 1,000 steps and the trajectory had not flattened); (2) `state_rule=delta`; (3) `state_rule=pinned` (non-decaying addressable slots). Then the serving side: a streaming generation loop over `forward_streaming` (windowed, state carried, partial-window slot alignment to CHUNK) so generation stops recomputing full windows per byte -- that is also where IDEA 10's RAG-lane speculative decode lands. Secondary defect: the streaming conv tail still carries padding-derived columns when a window is not slot-full.

## persistent memory

**IDEA 20 — the net updates itself from what it processes: tell wren something once and it never forgets (2026-08-19)**

User directive (permanent): stateless serving with re-injected context is the problem to eliminate, not a solution. Program = three memory timescales, complementary-learning-systems style, all self-contained. Grounded in a three-survey literature pass (worklog 2026-08-19); strongest anchors: linear attention IS fast-weight memory (Schlag et al. 2102.11174) — wren's GLA state matrices are already an outer-product associative memory, just one with decay-to-zero writes; the delta rule turns that state from a decaying superposition into an in-place-editable regression memory holding up to D orthogonal associations exactly, revisable without residue (DeltaNet 2406.06484, Gated DeltaNet 2412.06464); Titans' surprise-gated MLP memory works at 170-760M — our regime (2501.00663); RWKV community practice proves per-user serialized recurrent state in deployment; and the weight-consolidation recipe is measured (Ibrahim 2403.08763: 5-25% generic replay + constant low LR never re-warmed; Berglund 2309.12288 reversal curse + Allen-Zhu 2309.14316: a fact stated once is memorized-but-unextractable — every fact needs ~10-30 augmentations in both directions at learning time; Meta 2510.15103: gradient-sparse memory-slot FT cuts fact-injection forgetting 89% -> 11% vs full FT).

Tiers mapped onto wren:
- **T1 working memory** (within a session): the GLA carried state. IDEA 7 IS this tier's substrate and arm 1 proved it trainable. Whole state = 4.2 MB fp32 (16 layers x 16 heads x 64x64 + conv tails).
- **T2 episodic memory** (across sessions/days): (a) serialize the carried state per conversation, reload on resume — pure engineering, RWKV precedent, blocked only on the streaming generation loop; (b) a write rule that does not decay everything: delta-rule state and/or a surprise-gated slow lane (decay-exempt heads or a small Titans-lite MLP branch, 1-5M params, per-user file). Own-ledger caveats: delta NaN'd at 10M pretraining (failures.md) — but the retry conditions now exist (non-finite step-skip guard, capped beta, and low-LR *adaptation from trained weights* instead of scratch pretraining); pinned slots were killed on quality tax (retry only with an explicit recall loss).
- **T3 semantic memory** (permanent): sleep consolidation — a nightly run that trains the day into the weights. **Substrate (user, 2026-08-19): the model's OWN experience — its internal thought, its generations, its actions — not curated external data.** Biological anchor: sleep replay replays the day's neural activity, not the day's stimuli. Two mechanisms, run as arms: (m1) fact-SFT — extract stated facts from the day's exchanges, ~20 templated augmentations each (both directions + QA + dialogue, templated in-house), 25% generic corpus replay, constant low LR, early-stop per fact, spaced repetition, merge fuse theta <- 0.7*theta_ft + 0.3*theta_prev; (m2, cleaner and novel) **trace consolidation / state-to-weights self-distillation** — replay the day's (context, carried-state, prediction) triples and train the weights to reproduce from bare context what the model could only do WITH its working-memory state; dense logit signal instead of sparse fact labels, no fact-extraction pipeline, may sidestep the augmentation requirement (that is the experiment). What the fast tiers held yesterday, the cortex knows today. Estimated 10-20 min/night at 200M on this box; doubles as work on the faster-consumer-training axis. Prerequisite for both arms: an **experience log** — the platform records every serving exchange (bytes in/out + generation params + model id) as the day's replay substrate.

Experiment ladder (one variable at a time):
- **E1 RESOLVED 2026-08-19 (failures.md)**: dose falsified — gap 0.407 @1000 -> 0.383 @3000, content probes at zero both doses. Side result: wren1_3@3000 improved generally (val 0.5491, identity 1.00, grounded 0.38, loop 0.20) and replaces @1000 as the streaming base. **E1b RESOLVED 2026-08-20 (failures.md)**: delta-alone configuration falsified (wrap gap unchanged, chat degraded, E2/recall flat) BUT the state became content-bearing — anticopy absorbed copy 0.06 -> 0.47, first nonzero absorbed engagement. Capacity without curriculum. Two kernel defects fixed en route (decay-ratio underflow; pre-mask exp inf in backward — tests/training/test_delta_underflow.py). **E1c RESOLVED 2026-08-20 (failures.md)**: curriculum v1 falsified — binding still absent (state carries familiarity + primacy, not noun->word binding; three state arms now corroborate the IDEA 8 wall inside the state mechanism), template overfit into behavior; wrap gap 0.358 (best yet) and content transport stable are the retained wins. Curriculum v2 retry conditions parked in failures.md. **Next: E4 sleep** — exact facts belong in weights (the tier built for them); the state's proven role is transport, gist, and working memory. E4 harness = fact-set + templated augmenter + closed-book QA probe + night runner over the existing sleep pieces.
- **E2 retention half-life + revised-fact probe**: inject N codeword facts, bury under 1K/10K/100K bytes, probe across a save -> reload cycle; include *revised* facts (same key, new value). GLA vs delta discriminates on the revised-fact curve. **Probe form is discrimination, not generation (2026-08-19)**: exact-recall generation of arbitrary bindings is 0/8 even in-window — the role-binding wall (failures.md), not a memory result. v2 metric: margin = NLL(foil|state) - NLL(true|state); revised facts foil with the OLD value so residue is a sign flip. gla@1000 baseline: no signal at any K (win rates 0.29-0.71, inside n=24 noise), and save->reload margins exactly equal live — E3 persistence measured lossless. Verdict run vs delta: paired same-fact margin deltas, n >= 64.
- **E3 streaming serving + state persistence**: generation loop over `forward_streaming` (today every `stream()` variant recomputes `ctx[-seq:]` per byte and discards state — chat cannot use wren1_3's memory until this lands), then state save/reload per conversation. Falsifier: told-once next-session recall (teach a fact, kill the process, reload state, probe after K bytes of filler, sweep K), plus IDEA 7's original falsifier (generate 8x seq, no cliff). Watch: reloaded-state drift — bpb on neutral text with loaded vs fresh state.
- **E4 sleep consolidation**: 50 novel facts on night 1; closed-book recall both directions after nights 1/7/30 plus the base-ability probes. Falsified if recall <80% at night 30 or base bpb degrades >2% cumulative. Sweep replay {5,25,50}% and augmentations/fact {1,5,20} — if 1 matches 20, the augmentation hypothesis is wrong at byte level. Run fact-SFT (m1) vs trace-consolidation (m2) as matched arms: if m2 reaches m1's recall without the fact-extraction/augmentation pipeline, m2 is the keeper (cleaner, and it consolidates thought/actions, not just declaratives). Prerequisite: the experience log (serving exchanges recorded platform-side). **E4 m1 RESOLVED 2026-08-20/21 (successes.md): VALIDATED** — wren1_5@700 closed-book fwd 45/50 rev 49/50 (baseline 0/50), forgetting +1.50% of the +2% budget; recall is sigmoidal in dose and the budget binds at ~700 steps at lr 5e-6 (tripwire auto-stopped night 3 at 800). Reversal curse beaten by both-directions training (rev ≥ fwd throughout). Nights 1/7/30 retention legs pending (tools/e4_retention_quiz.py, 2026-08-27 / 2026-09-19). Follow-on levers, in order of expected value: (a) **rarity-scaled augmentation** — the residual misses are rare-word occupations (jobs fwd 21/25 vs residences 24/25; 3 never landed); scale augmentations/fact with object-word corpus frequency instead of a flat 20; (b) **extend the forgetting budget across nights** — flat 5e-6 spent the budget in ~700 steps; try per-night LR decay (5e-6 → 2e-6 → 1e-6, mimicking synaptic downscaling) or interleaving no-fact rehearsal-only nights to let bpb anneal back; (c) **m2 trace arm** still unrun — matched comparison remains the open E4 question; (d) replay/augmentation sweeps ({5,25,50}% replay, {1,5,20} aug/fact) still unrun — the flat-20 recipe validated first try, so sweeps are now about efficiency, not existence.
- **E5 (contingent on E2/E4 failures)**: surprise-gated slow-lane heads or Titans-lite MLP branch if decay kills multi-day state; a product-key memory layer with inference-time top-k slot writes if the monolithic state shows interference — capacity lives in DRAM, the one resource consumer CPUs have in surplus, and gradient-sparse slot writes are the best published forgetting number (11%), unverified on a GLA byte model (that is the experiment).

Pre-registered skips (evidence, not taste): EWC/SI (dominated by plain replay at every tested scale), ROME/MEMIT-class editing as the lifelong mechanism (collapses after ~250 sequential edits; localization assumes transformer MLP layers), LoRA as a forgetting cure (worse than full FT on TRACE; loses 71% NQ F1 in the Meta comparison), generative self-replay (the disk corpus is strictly better than 200M-byte-model self-generations). IDEAs 1/2 remain the retrieval-scale complements for knowledge that should NOT live in weights or state.

Goal refinement (user, 2026-08-19): mechanisms must be scale-free — 200M is the falsifier bench, not the target; capacity objections are not kill lines if the mechanism rides parameter count. Three named properties beyond recall: (1) **create-then-know** — after the model writes a component, that knowledge must be IN it (its own generations pass through the same state writes as reading, and consolidation treats its own work as first-class experience) — no re-reading its own output; (2) **no context-full failure mode** — the streaming walk makes the window a working buffer, not a ceiling; (3) **anti-copy** — the named pathology where an LLM given a big file and asked for a "new" one produces a near-duplicate, because a context window is perception (raw surface pinned in view) rather than memory (compressed abstraction). Generation from absorbed state cannot copy surface it no longer has — this is testable. Brain framing is explicit: multi-timescale regions that affect each other (fast state = working memory, surprise-gated store = hippocampus, slow weights = cortex, sleep consolidation = replay); the CLS/HOPE literature is the published anchor that a frequency-spectrum of interacting modules works in one net at 340M-1.3B.
- **E6 anti-copy falsifier**: same authoring task two ways on a state-carry model — (a) reference file in the context window, (b) reference file absorbed via the streaming walk, then window cleared, generate from state alone. Measure verbatim n-gram overlap with the reference AND task success. Prediction: (b) copies materially less at comparable task success; falsified if (b) only succeeds by copying or fails the task outright. First run: doc/paragraph scale on wren1_3+, before code scale. Baseline measured 2026-08-19 (wren1_3@1000, scratchpad anticopy_probe.py): in-window copy 0.044 / engagement 0.028; absorbed copy 0.000 / engagement 0.000 (loops, no content survived 1,000 carry steps) — the yardstick works, the verdict waits for higher-dose and delta checkpoints; the probe grows in importance with model scale (big models copy more in-window).

Sleep self-containment (user, 2026-08-19): sleep never imports new external training data — the improvement signal is the model's own conversation, actions, and study. Rehearsal to prevent forgetting draws ONLY from the model's own past (its base corpus already on disk = its own prior experience, plus prior experience-log days) — that is memory rehearsal, not data import. The model MAY write notes/md as externalized thought, but notes are study artifacts to be memorized in sleep, never permanent load-bearing infrastructure the model must re-read (the Hermes-harness failure mode, minus the sprawl). Over its life the model generates text and learns from itself.
- **E7 skill ingestion**: the directed-learning path — user drops a resource (doc, book, spec) in; while "awake" the model studies it: absorbs it through the streaming walk AND authors its own study artifacts (rephrasings, notes, Q&A, exercises — self-generated augmentation, the self-contained answer to the 10-30-exposures requirement; SEAL 2506.10943 showed self-authored study data beating GPT-4.1-authored at knowledge incorporation); sleep consolidates resource + study artifacts into weights. Falsifier: closed-book task success on the ingested skill materially above pre-ingest after one sleep cycle, base probes held. Honest caveat: study-artifact quality scales with model quality — at 200M expect weak self-teaching; the mechanism, not the ceiling, is what 200M proves. Platform piece shipped 2026-08-19: the experience log (data/experience/, one JSONL/day, both serving backends record every exchange; inference/experience.py, tests in tests/mri/test_experience_log.py).
- **The acceptance test (user, 2026-08-19)**: "ask the model what did you just say — it can remember without re-reading everything." Harness: scratchpad chat_state_recall.py — two-turn conversations over the streaming path, turn 2 sent alone, fresh-state leak control per item. Baseline wren1_3@1000: **1/6, zero leaks** — one genuine cross-turn recall through pure state ("You traveled to Norway"), the rest abstain ("I don't know", the sft_idk profile — right failure mode, no hallucination). Every memory arm reports this number; the goal is 6/6. **2026-08-21: wren1_3@3000 (E4 fork parent) = 0/6 (all abstentions); wren1_5@700 (slept child) = 3/6, zero leaks** — sleep consolidation itself moved the acceptance number, not the parent's quality: fact-QA consolidation appears to unteach the abstention reflex in favor of answering from held state, at no hallucination cost. Secondary discovery of E4; mechanism unplanned.

## model growth

**IDEA 21 — grow wren into the flagship: function-preserving expansion instead of from-scratch retraining (2026-08-20)**

User directive: build off what wren has learned — expand the network to more params/layers and continue, never restart. The technique family is established: Net2Net (function-preserving width growth by neuron duplication + outgoing-weight splitting; depth growth by identity-initialized layers, 1511.05641), bert2BERT/LiGO (~45-50% compute saved vs scratch), depth up-scaling (SOLAR 10.7B from Mistral 7B), LLaMA-Pro identity-init block expansion. Mapping to VeritatePatched: widen hidden (duplicate columns in tok_emb/qkv/FFN, split outgoing rows, RMSNorm scales copy), grow heads or head-dim in the recurrent mixer (new heads zero-init their output contribution so the function is preserved; state matrices grow with them — bigger memory boards for free), deepen with identity-init global blocks (LLaMA-Pro style). Serialized stream states do NOT survive growth (checkpoint-bound by design — states re-form from conversation).
Falsifiers, in order: (1) function preservation — grown 500M at step 0 reproduces wren source-checkpoint val within noise (target: exact); (2) growth pays — grown model reaches the source val floor and beats it in materially fewer tokens than a from-scratch 500M arm given the same budget (the bert2BERT bar is ~45% savings; claim it only if measured here); (3) nothing regresses — chat ladder + cliff + content probes at parity or better after the continue run. Precondition: the memory-native recipe (IDEA 20 write rule + curriculum + state carry) is settled first, so the flagship is grown INTO it, not adapted after. Note: role-binding wall evidence at 800M is from an undertrained model (user, 2026-08-20) — the wall is established only at well-trained 122M/200M; the grown flagship retests it properly.
Requirement added 2026-08-20 (user): the flagship must serve int8 on cardinal-01. CORRECTED 2026-08-20 (agent verification on cardinal): the engine gap does NOT exist — v13 int8 compute already shipped (hybrid.c scalar+AVX2+sdot, CPUID-dispatched, parity-tested; wren1_0_int8 serves as v13 int8 on cardinal at 6.63 ms/byte vs 9.48 fp16, 1.43x). Post-training int8 export is already greedy-byte-identical at 200M (successes.md grd1 12/12), so QAT is insurance for 500M, not a prerequisite: run the 10M-class hybrid+QAT pilot only if PTQ parity degrades at the grown size. If QAT is used, it must target the v13 scheme (per-output-row weight scales + dynamic activation scale), NOT qat.py's v9 scheme (per-tensor + fixed scale-32) — a scheme mismatch trains against the wrong rounding. Remaining cardinal-track work: the AVX-512 SIGILL load bug on AVX2-only x86 (failures.md 2026-08-20) and further end-to-end speedup only if 500M needs it (attention/GLA are the fp32 dilution). Sequence: memory recipe settles (IDEA 20) → grow (this idea) → PTQ int8 export → parity gate decides whether QAT enters.

## green-ai product

**IDEA 8 — smallest strong conversationalist, all facts external; attack the role-binding wall directly**
Bind-then-read: a discrete slot/register layer between retrieval and the LM body, supervised on grounded_v3. Kill line: lifts OBJECT-role accuracy off the ~0% floor without breaking slot copy.
Product P1 (committed): on-device private-document assistant. Falsifier: retrieval gates re-run with paraphrased queries sharing no token with the gold chunk.
Product P2: local-first with cloud fallback, router gated on retrieval score, never LM self-judgment. Falsifier: escalation precision/recall on a mixed set.
Below 122M is untested — minimum viable reader size unknown.

## agentic retrieval

**IDEA 18 — the model calls Carpathian's API for the facts it cannot hold at 270M (2026-08-17)**
Knowledge is fetched at inference instead of trained in. Most of the machinery already exists: Hermes function calling is
the canonical framing (`documentation.md`), `inference/agent/loop.py` runs the tool loop, `fetch` and `retrieve` are
registered tools, and `config.json::capabilities` already has an `agent` tier. Missing: a `carpathian` tool and the
corpus that teaches when to call it.
Mechanism: register the tool beside `retrieve`; extend `tools/build_agent_corpus.py` with a call-the-API family
(question, `<tool_call>`, `<tool_response>`, grounded answer); SFT at low dose with heavy chat replay per the dose rule.
Blocker: the credential, not the contract. `data/mri_settings.json::teacher_configs.carpathian` already names the
endpoint (`https://api.carpathian.ai/ai/v1`, OpenAI-compatible, bearer key), but `/models` answers "Invalid or inactive
API key" and the `model` field is empty. A live key and a model name are the first gate; a corpus written against a
guessed schema is dead data.
Falsifiers, in order, each gating the next. (1) Call decision: a well-formed `<tool_call>` on questions outside the
training corpus, a direct answer on questions inside it. Kill above 0.20 spurious call rate or 0.05 malformed JSON.
(2) Grounded read: accuracy on the returned document must match the same model's in-context copy rate with that document
pasted straight into the prompt. Materially lower means the tool-response framing is the defect, not retrieval.
(3) End-to-end: beats the same model with no tool on the same held-out questions.
Boundary: `system_acc ~= retriever_precision@1 x reader_acc(1 chunk)` (successes.md) holds for any external source, so an
API returning several candidates inherits the multi-candidate failure (top-1 0.130 vs top-3 0.120, failures.md). The API
returns one document, or a re-ranker collapses k to 1 before injection.
Coupling: web browsing is this same mechanism with a different tool. Build the Carpathian tool first: controlled response
shape, trusted content, no network flakiness in the falsifier.

## developmental training

**IDEA 9 remnant — value/aversion module**
A small separate network learns "avoid this output" and modulates decoding. No falsifier defined. Parked.

## speculative decode

**IDEA 10 — byte n-gram speculative decode: the only lossless lever left on bandwidth-bound hardware**
Draft K bytes by longest-suffix match against prompt+generated text, verify in one batched weight stream, accept the longest matching prefix (greedy output byte-identical). Non-boundary decode already at 83% of cardinal's bandwidth ceiling. Gates: mean accepted length ≥2.5 → ship; 1.5-2.5 → ship only if ms/byte improves; <1.5 → failures.md, retry once an MTP-head draft source exists. Report accept length by traffic type, not just the mean.
Gate measured 2026-08-18, simulated offline over 67 real greedy transcripts (wren1_0@1250; exact by construction since the transcripts are the model's own emissions, K=16). Grounded/RAG traffic clears at every min-match m: accept-per-draft 2.62 (m2) / 4.98 (m3) / 6.54 (m4), 2.0-2.5 bytes per verify round (~2.2x fewer weight streams), driven by copying injected context, which the repetition guard does not ban. Plain chat: 0.84 (m2, fail) / 2.0 (m3) / 2.99 (m4 at 6.9% coverage), 1.2-1.36 bytes/round; the pre-guard transcripts flatter chat via loops, so real chat numbers are lower. Decision: fund the engine implementation for the RAG path first (draft + batched verify in the C engine, bitwise-identity check per kernel rule); plain chat rides along only if it is free.

## corpus mix

**IDEA 17 — long-form multi-turn SFT: install sustained conversation from the rebuilt corpus (2026-08-10)**
The brevity diagnosis of the old chat bin (median assistant turn 14 B, 2.4 turns/conv) and the rebuilt no-decay corpus answering it (median 267 B, no decay through turn 7) are banked in successes.md 2026-08-10. Open work: pack the corpus into a bin turn-boundary-aligned at the current `seq`, then SFT with **loss_mask on** (assistant turns only, unlike the current run) at a high chat ratio.
Falsifier: on held-out multi-turn prompts the SFT'd model must raise median generated reply length and sustain ≥6 coherent turns without degenerating (loop, topic-collapse, or premature `<|im_end|>`), judged blind against the current base. Correctness explicitly not scored. If replies stay short after training on long-form data, brevity is coming from the base or the decode path, not the SFT corpus — check EOS calibration before building more data.
Coupling: turn lengths already brush the ceiling (max 1509 B vs `seq` 1024 B), so anything past ~1 KB of context is truncated today. Long-form SFT and IDEA 7 (unbounded context) are the same project — a bin with long conversations is untrainable at seq 1024, so IDEA 7's rolling-window KV + `state_carry=chunks` is a prerequisite, not a follow-on.

**IDEA 11 — is chat quality set by corpus ratio rather than steps? (from the wren_sft wash, failures.md)**
wren_sft ran 70% general web / 20% chat+instruct. Over 30,000 steps val improved 0.0142 while format adherence flat-to-down and only factual recall gained. Hypothesis: instruction-format adherence is ratio-bound, not step-bound, and further steps at this mix cannot buy it. Test: matched 200M runs at chat+instruct 0.20 (control) vs 0.50, equal total steps, same seed. Falsifier: the 0.50 arm must beat control by a clear margin on a held-out format-adherence set (answer-length limits, "only the word X", list-of-N) with aggregate val allowed to be *worse* — if aggregate val is the only thing that moves, the hypothesis is dead. The format set exists: `ifeval form` (`veritate_mri/data/eval/samples/ifeval_form.json`, 280 obedience-only items, deterministic checkers, hand-written so not comparable to published IFEval). Ready to run once the two matched 200M arms are budgeted.

## decode-to-weights distillation

**IDEA 19 — train the no-repeat behavior into the weights instead of the decoder (2026-08-18)**
The `no_repeat_ngram=16` ban erases looping at zero accuracy cost (successes.md 2026-08-17) but lives in the decoder. Frontier labs get non-looping from preference optimization, not decode rules -- their decode penalties ship but default off -- so the behavior is learnable in principle; the open question is whether 200M learns it. Mechanisms, cheapest first:
1. Guard-distilled SFT -- **killed 2026-08-18** (failures.md): 0.08 dose / 500 steps on 1,929 own guarded replies bought loop 0.20 -> 0.13 (under half the gap, target <0.05) and paid grounded 0.25 -> 0.12 and identity 1.00 -> 0.83 at the same checkpoints. Do not re-run below 1B or without a dose sweep.
2. Unlikelihood loss on repeated n-grams as an opt-in trainer lever (`veritate_trainer` flag, Training-tab control), penalizing loop continuations at training time. Now the front of the queue.
3. DPO pairs (banned vs bare-greedy reply to the same prompt) once a preference trainer exists. Heaviest; last.
Falsifier: bare-greedy loop rate on the 30-prompt ladder drops 0.20 -> <0.05 without grounded/identity/median regression. Kill a mechanism if it buys less than half the gap; if all three fail, the behavior is not learnable at this scale and the decoder keeps the job permanently.
Boundary: the guard stays shipped regardless -- serving defaults and measurement stay separate (grading is always bare-greedy).


## engine decode

**IDEA 22 — calibrate the engine's thread count per step class, not once on the cheap class (2026-08-23)**
`hybrid.c::hybrid_calibrate` climbs a 1,2,4,..cap ladder while each rung beats the previous by
`HYBRID_CALIB_KNEE` (13%), and times **non-boundary** decode steps only. But decode is bimodal:
boundary steps run the GLA global-block stack, cost ~5x, and are ~54% of decode time
(successes.md 2026-08-23). One pick timed on the cheap class then governs the expensive one.
Measured on cardinal: 4 threads 16.58 ms/byte vs 8 threads 14.85, a 10.4% gain the knee rejects,
and the pick is unstable across models on one box (`wren1_0` cached 8, `wren1_3` cached 4) because
each rung is compared only against its immediate predecessor and run-to-run noise straddles the
threshold. Mechanisms, cheapest first: (1) compare each rung against the best-so-far rather than
the previous, which removes the instability without changing the knee; (2) time both step classes
during calibration and cache two picks, switching per step (`is_boundary` is already computed at
the top of `hybrid_step`); (3) make `HYBRID_CALIB_KNEE` an env-overridable tunable (rule 19 -- it
is a hardcoded tunable today). Falsifier: (2) must beat the single-pick baseline by >5% end-to-end
on at least two boxes with different core counts; if the two classes want the same worker count
everywhere, only (1) and (3) are worth keeping. Needs an engine rebuild per arch, so it carries
rule-25 bitwise-parity obligations; row-split parity holds at every worker count, so output cannot
change. Workaround shipped meanwhile: the `engine_threads` setting pins the count.
**SHIPPED 2026-08-24, falsifier NOT met for (2).** All three mechanisms are in: calibration times both
step classes and switches workers for the global-stack loop, rung selection compares against the best
rung so far instead of its predecessor, and `VERITATE_HYBRID_CALIB_KNEE` / `VERITATE_HYBRID_BOUNDARY_THREADS`
are env knobs. Parity holds through the mid-step switch (new test drives the classes to different
counts). Measured on cardinal at 2.0 GHz, 8 cores: plain 10.742/6.261/4.775/**3.939** ms/byte and
boundary 57.662/34.511/27.320/**23.144** at 1/2/4/8 workers -- a boundary step costs **5.9x** a plain
one, exactly as predicted, but BOTH classes pick 8, so the per-class switch is inert on this box and
(2) gains 0%, not the >5% its falsifier demanded. The second box (Mac arm64) was unavailable: wren2
is mid-pretrain and off limits. (1) is still worth having on its own -- it is what made the pick
stable -- and the clamp lift independently removed the symptom that motivated (2): at 800 MHz the
4->8 rung gained 10.4% and the 13% knee rejected it; at 2.0 GHz it gains 17.5% and clears. Retry (2)
on a box where the two classes actually diverge (more cores, or a shape whose global stack is a
different fraction of the step).
