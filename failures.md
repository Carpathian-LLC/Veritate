# research failures (kill list)

Falsified approaches. Each entry: what was tested, the measured result that killed it, and the retry condition. An approach on this list is dead until its retry condition is met. Entries are research outcomes, not bug fixes. Entries arrive here from `ideas.md` when an idea hits its kill line; index at `research.md`.

## 2026-06-27: from-scratch dense byte-level 2.5B coder

- Tested: `veritate_3b` (2.52B dense, vocab=256) pretrained from scratch on distilled Python chat corpus, dashboard run.
- Result: val loss plateaued near 1.0 by 205M tokens, zero code ability, looping English. Model deleted.
- Cause: compute wall. Competitive coding needs tens of billions of tokens; box delivers ~550 tok/s at 2.5B.
- Retry condition: a training-efficiency lever of 10x or more at equal quality, measured on this box.

## 2026-07-01: cheap byteification of a subword coder

- Tested: Qwen2.5-Coder-0.5B converted to vocab=256 (embed/head swap, exact 256/256 warm-init), continue-trained on 10 MB distilled Python, 2000 steps.
- Result: HumanEval pass@1 = 0/30 = 0.0 percent on held-out problems. In-distribution samples looked correct (memorized surface patterns); held-out eval exposed them.
- Cause: 1-byte-per-step granularity is an unseen input distribution for the pretrained body; recovery needs billions of adaptation bytes (Bolmo needed ~49B tokens with patching). Tiny corpus cannot buy it.
- Retry condition: patching front-end (body sees ~4 bytes per step) plus GB-scale diverse corpus, or frozen-body codec route. Full writeup removed; see git history for `bigmodel_byteification_handoff.md`.
- Standing rule from this failure: never believe in-distribution samples; only held-out executed benchmarks count.

## 2026-07-03: backprop-free learning rules for language modeling

- Tested: literature verdict (Forward-Forward arXiv 2301.01452, predictive coding 2212.00720 / 2308.07870, Mono-Forward 2501.09238, surveys through 2026).
- Result: no backprop-free rule has reached backprop parity on real language modeling at any scale. Demonstrations stop at MNIST/CIFAR/toy translation, with worse perplexity and higher convergence cost.
- Retry condition: a peer-reviewed result showing LM parity at 100M params or above.

## 2026-07-03: ternary (BitNet) as a training-cost lever below 3B

- Tested: literature verdict (BitNet b1.58 2402.17764, BitNet Reloaded 2407.09527, 2B4T 2504.12285).
- Result: below ~3B params ternary needs roughly 2x hidden width to match fp16 quality, and QAT adds training cost. It is a decode/energy lever, not a training lever.
- Standing use: keep ternary at EXPORT time (engine ternary path exists). Retry condition for training: parity evidence at or below 1B without the width penalty.

## 2026-07-06: pinned memory register (decay-exempt state slots) on the hybrid trunk (M2)

- Tested twice at 10M (matched arms, muon, 12k steps). Pure chat: trained stably (77k tok/s = 97% of gla) but a real LM tax — tail-10 val 0.7637 vs gla 0.6907 (+0.073, ~7 sigma) — with needle recall 0.00 = 0.00 (floor, no discriminating power). Recall mix (fair test vs m0recall, single delta = mechanism): quality-neutral (tail-10 0.7010 vs 0.6990) but in-distribution recall 0.338 vs 0.423 at n=130 (z=1.4 worse; 0.28 vs 0.47 at n=40) — consistently below baseline, never above.
- KILLED per the pre-registered condition "pinned recall no better than M0": no measured benefit in any sample, one measured cost (the pure-chat val tax). The hand-designed salience writer + decay-exempt pins subtract learnable state capacity without buying retention at this scale.
- Retry conditions: (a) much larger scale where raw state capacity stops being the binding constraint, (b) pins trained with an explicit auxiliary recall loss (the writer currently gets only the LM gradient), (c) pins in a streaming-trained regime where cross-window retention is actually trainable (see campaign route 4).
- Runs: `models/m2pinned_10m_qat/`, `models/m2pinned2_10m_qat/`.

## 2026-07-06: late-phase recall-SFT at 25% — degrades out-of-distribution recall at 121M

- Tested: chat80m phase 4 (resume from the 48k SFT checkpoint, 4k steps, 1e-5 WSD, 25% `chat_recall_v1` synthetic recall-pressure conversations in the chat mix). Falsifier scoreboard: needle curve step_52000 vs the step_48000 baseline (0.92 / 0.25 / 0.00 at ~190 B / ~480 B / past-window).
- KILLED the phase: recall at ~190 B collapsed 0.92 -> 0.08; ~480 B 0.25 -> 0.17; past-window unchanged at 0.00. Coherence contradiction 0.25 -> 0.0 — the model no longer echoes even the WRONG planted fact; it answers from the trained template distribution. Mechanism: the pre-recall model had an emergent copy behavior ("echo the salient unusual token from context") which the narrow, highly-templated recall corpus REPLACED with fact-type-bound template recall that does not transfer to alien surface forms (the benchmark's hyphenated codes / 5-digit amounts). Interference, not augmentation.
- Standing result that survives alongside this kill: the same corpus at 10M (10% in-pretrain-style mix) taught genuine in-distribution retrieval (0.47 vs baseline 0.05 on held-out recall-val). The lever is real; late-phase concentrated dosing is what failed.
- Action: flagship chat model = step_48000 (rolled back). step_52000 was later pruned in the model cleanup; its needle curve JSON remains the evidence.
- Retry conditions: (a) small fraction (~5%) mixed into PRETRAIN, not a late phase — the copy behavior then never faces a concentrated narrow distribution late; (b) surface-form-diverse recall needles (codes, amounts, names, dates) which requires re-balancing the contamination guard toward held-out template splits rather than wholly disjoint fact types; (c) score any retry on BOTH in-distribution recall-val AND the needle benchmark before adoption.
- Runs: `models/chat80m_80m/` steps 48000 (keep) vs 52000 (research); curves: `experiments/v2/longctx/chat80m_needle_curve.json` vs `chat80m_recallsft_needle_curve.json`.

## 2026-07-08: identity SFT (round 1b) silently destroys conversation-copy at 121M — the needle gate was skipped

- Tested: post-hoc needle A/B on the identity round the tuning journal recorded as a clean PASS (chat80m resume 48000 -> 51000, ~3k steps, 1e-5 WSD, 10-15% `chat_identity_v1` in a chat mix; gates run at the time: identity battery, general-chat battery, val). The "needle ~unchanged" gate was in the plan and did NOT run. This entry is the makeup measurement.
- KILLED the clean-PASS verdict: needle recall at ~190 B collapsed **0.917 -> 0.167**; ~480 B 0.250 -> 0.000; contradiction rate 0.25 -> 0.00 (12 trials/row, 4 seeds, same protocol as the recall-SFT kill). Identical interference signature to the 25% recall-SFT kill above, from a much smaller dose: the model stops echoing planted context and answers from the trained template distribution. Val is blind to it — val IMPROVED (0.647 -> 0.644) while the skill collapsed. A skipped falsifier is a false pass.
- What the SFT actually bought (measured, live C engine, temp 0.5, n=6-8 per cell): persona-context application. 51000 with the persona line "You are Veritate..." in the context block answers its name 6/8 and maker 7-8/8; 48000 with the SAME persona line scores 0/6 across the board ("Your name is Jack Thompson", "Yes, I am human"). Bare (no persona), both checkpoints fail the name at sampling temps (0/8). Identity is not "in" 51000 as a fact; the SFT taught it to APPLY persona-shaped context, and the serving layer now supplies that context (persona line shipped in /hybrid/chat, name 0/8 -> 4-6/8 deployed).
- Control that separates the failures: alien-fact extraction from a `context:` block (invented entities, the RAG surface) scores 1/4 greedy at BOTH 48000 and 51000 (48000 sampled: 4/12). That skill never existed at either checkpoint — a training-dose gap (chat_v1/v2 grounded examples are common-surface facts at trace dose), NOT damage from this round. The 121M transfer gap mirrors the 10M one (in-distribution 0.47 vs alien 0.00).
- Action: serving flagship stays 51000 (identity via persona + all decode guards; RAG is equally broken at both checkpoints so the rollback buys nothing deployed). 48000 retains the conversation-copy skill and is the designated BASE for the repair round.
- Retry conditions: (a) one combined SFT from 48000 — grounded_v3 (~25%, alien-entity extractive QA, 4 families incl. honest-miss) + chat_identity_v1 (~15%) + chat mix, ~2-3k steps at 1e-5, gated on ALL of: needle@190B >= 0.8, identity >= 5/8 sampled temp 0.5, alien-fact context battery >= 3/4 greedy, chat battery no-regress, tail val. Queued behind the 200m pretrain (GPU). (b) in-pretrain dosing from step 0 — LIVE: chat200m launched 2026-07-08 with grounded_v3 2.5% + identity 2% + recall 4% in the mix.
- Runs: `models/chat80m_80m/` steps 48000 vs 51000; needle JSONs in the session scratchpad (protocol identical to `chat80m_needle_curve.json`); persona/alien-fact batteries logged in `worklog.md` 2026-07-08 section.

## 2026-07-06: delta-rule state update (Gated DeltaNet) on the hybrid trunk — not trainable on this stack (M1)

- Tested twice at 10M (hybrid trunk global mixer, muon, fp32, bs32/seq512, 12k steps, cosine base 6e-4). Run 1 (m1delta, pure chat): crept to divergence at step ~4420 — root-caused to fp32 catastrophic cancellation in the whole-chunk WY nilpotent inverse under beta-gate saturation (intermediates ~2e17; the long-memory head was numerically broken from ~step 800; full chain in `m1delta_divergence_analysis.md`). Fix shipped: block-recursive triangular inverse, same exact math, bounded intermediates (~20), triple-verified incl. a saturation stress test (old code 2.4e31, new 5.4e-7) and a 50-step exact-resume repro (finite, flat grad norms).
- Run 2 (m1delta2, recall mix, fixed inverse): diverged AGAIN at step 2094 — different signature: sudden single-step NaN with no precursor (grad norms flat 0.37-0.44, loss descending 0.77). The stabilizer survived the replay of run 1's divergence but from-scratch training found a second path. Throughput was fine both times (85% / 77% of gla, above the 70% line). Baseline gla and pinned arms trained clean on the same configs, so both failures are delta-specific.
- Verdict: KILLED on trainability — two divergences under two different numeric regimes, one retry per line spent. The oracle-exact chunkwise math and the published DeltaNet results are not in question; the failure is this formulation's stability under muon/fp32/MPS at byte level without the specialized fused kernels the literature trains with.
- Retry conditions: (a) beta parameterization capped away from saturation (e.g. beta_max ~0.95, changes the math where it currently saturates — needs its own A/B), (b) markedly lower LR for the delta projections (breaks matched-arm simplicity), (c) a proper fused/chunked kernel path (CUDA fla-style) if the platform ever leaves MPS, (d) autopsy of the m1delta2 sudden-NaN checkpoint if the mechanism is revisited.
- Runs: `models/m1delta_10m_qat/` (creeping), `models/m1delta2_10m_qat/` (sudden, stopped at ~2600 to free the GPU).

## 2026-07-05: selective language modeling (RHO-1 top-k excess-loss tokens) at byte level (E6)

- Tested: `slm_ref=e1muon_10m_qat, slm_keep=0.6` (frozen dense-muon 10M reference scores tokens, train loss on top-60% by excess loss), single delta vs `e2patched` (identical trunk=patched, muon, corpus sha, seed, 12000 steps).
- KILLED on the pre-registered condition "quality regression at equal steps": tail-10 val 1.0638 vs e2patched 0.9763 (+0.0875, ~8 sigma beyond tail-mean noise). It never reaches e2patched-final val at all, so byte-savings-to-target is zero.
- What the curve shows: SLM is fast to a mediocre plateau — it reaches the 1.064 level 1.24x sooner in steps than the baseline passes it, then flatlines. At byte granularity the "easy" 40% of tokens is not skippable filler; it carries the structure (whitespace, morphology, tag syntax) the hard tokens sit in.
- Second observation: per-eval val variance is large on this arm (stdev ~0.035 over 6000-8000); single milestone evals lied in both directions. Verdicts on any arm now use tail-averaged val (same lesson class as E7's random-R noise).
- Retry conditions: (a) subword/patch-level selection where a "token" is a semantic unit, not a byte; (b) a reference model much stronger than the student scoring at patch granularity; (c) SLM restricted to a mid-train anneal phase instead of the full run (the early-speed effect is real).
- Runs: `models/e6slm_10m_qat/` vs `models/e2patched_10m_qat/`.

## 2026-07-05: test-time depth scaling via weight-tied loops at byte level (E7)

- Tested: `trunk=looped` (patched local blocks + half the unique global blocks, weight-tied and iterated R times, R sampled uniform 1-4 per training step Huginn-style, per-loop input injection), muon, params-matched to dense (10.12M vs 10.08M), fineweb_edu, 12000 steps.
- The "think longer" claim is DEAD at this scale: R sweep on the trained checkpoint (paired, 64 identical val batches) gives CE 1.0103 / 0.9975 / 0.9965 / 0.9983 / 1.0050 / 1.0142 at R=1/2/3/4/6/8. Quality peaks at R=3, the training-time mean depth (2.5), and degrades monotonically past it. No test-time compute scaling emerges; the model interpolates within its trained depth distribution.
- Second cost, measured: random-R training makes validation noisy (tail-20 stdev 0.0087, ~9x the other arms), so any single eval of a looped run is untrustworthy without averaging.
- What survives (recorded here, not scaled): at matched params it beats dense-muon (final val 0.9920 vs 0.9990, ahead 111/120 matched evals, 1.63x wall-clock to dense-final). Attribution is unproven: no loop-free control with the same halved unique blocks was trained, so the win may belong to the patched front-end, not the loop. It loses to full patched (0.9776, 1.82x) and hybrid (0.9707, 1.70x) on quality AND wall-clock, so it is not the trunk to scale.
- Retry conditions: (a) params-bound deployment where 10M-class weights must act deeper than they are (then train the loop-free halved-blocks control first for attribution), (b) reasoning-style executed benchmarks where test-time compute is the claim, with loop counts trained beyond 4.
- Run: `models/e7looped_10m_qat/`; sweep: `SMOKE_RESULTS/e7_loop_sweep_smoke.py` + `_stats.json`.

## 2026-07-05: fast-weight memory for explicit fact recall at 10M (E4b, closes the E4 line)

- Tested: `trunk=memory` retrained with memory CARRIED across the 4 contiguous chunks per step (the E4 retry condition). Language quality: val 0.9867, BEATS dense-muon 0.9990: the carried memory works as cheap context extension (attention sees 512 bytes, memory spans 2048). One benign non-finite skip in 12000 steps.
- Knowledge-injection quiz on the trained checkpoint: exact-match recall lift 0.0 at every distractor length. Distance profile: within the trained horizon (1024 bytes) a soft trace exists (fact-span bpb win rate 0.667, on 5.42 vs off 5.52); at 4096+ the stale memory actively hurts (win rate 0.167, on worse than off).
- Verdict per the pre-registered condition: the fast-weight route to explicit knowledge recall at 10M is DEAD. It stores soft traces within its training horizon, not retrievable facts.
- Kept: the context-extension effect is real and free (E4b val beats dense); the memory trunk stays available as a long-context device, not a knowledge store.
- Retry conditions: (a) training carry horizons at or beyond the recall distance (seq x chunks of 8k+), (b) larger scale (Titans floor was 170M), or (c) pivot to retrieval-based memory (rank 3 of the memory shortlist), which needs no gradient at all.
- Runs: `models/e4memory_10m_qat/` (v1), `models/e4bmem_10m_qat/` (carry); quiz: `SMOKE_RESULTS/e4_knowledge_injection_smoke.py`.

## 2026-07-04: surprise-gated memory with per-window reset training (E4)

- Tested: `trunk=memory` (Titans MAG class, stabilized inner loop) trained 12000 steps on fineweb_edu with fast weights RESET every 512-byte forward. Plain-text quality preserved (val 1.0026 vs dense 0.9990, regression +0.004, under the 0.05 limit). Knowledge-injection eval on the trained checkpoint: exact-match recall lift 0.0 points (0/12 both arms; falsifier kill line +10), fact-span bpb win rate 0.583 (barely above coin flip).
- Diagnosis: with per-window reset, the training loss never rewards reading memory across windows: attention already covers the whole window, so the write rule gets no retrieval pressure. Persistence was exercised only at eval time.
- Retry condition (E4b, concrete): train with memory CARRIED across the 4 contiguous 512-byte chunks the loader already feeds per step (reset between steps). Memory then serves bytes 513-2048 beyond the attention window, giving the loss a reason to store and read. If E4b also fails recall at +10 points, the fast-weight route at 10M is dead and consolidation moves to retrieval-based memory (rank 3 in the memory shortlist).
- Run: `models/e4memory_10m_qat/`; eval: `SMOKE_RESULTS/e4_knowledge_injection_stats.json`.

## 2026-07-04: TRM/HRM puzzle-recursion ported to language modeling

- Tested: literature verdict. TRM (7M) / HRM (27M) ARC-AGI results are per-task grid transduction with heavy augmentation; ARC Prize ablations attribute the gains to outer-loop refinement + per-task memorization, not the architecture. A direct ~1M-param TRM-on-TinyStories port lost to a plain transformer in every configuration.
- Retry condition: a peer-reviewed TRM-style result beating a params-matched transformer on held-out language perplexity or executed benchmarks.
- The transferable ingredient survives separately: weight-tied looped depth (Ouro 1.4B, RRT 1B, MoR 135M+), queued as the looped-trunk experiment.

## 2026-07-04: student self-improvement (STaR/BoN on own outputs) below 1B

- Tested: literature verdict. Label-free self-bootstrapping collapses on sub-1B students (marginal-to-negative on Qwen2.5-0.5B, 2511.04902): rollouts contain no correct answers to amplify. A 10-200M byte model has ~zero base success rate.
- Retry condition: the student clears a nonzero executed-benchmark pass rate after SFT; until then teacher-side judge-filtered distillation (72B judges, 14B generates) is the working form of the idea.

## 2026-07-04: unnormalized fast-weight inner loop at real width

- Tested: Titans-style memory branch (E4) with raw q/k projections and sum-scale inner gradients. Stable at hidden=64 (all smokes passed); at hidden=320 the fast weights hit NaN inside the FIRST forward (inner grads scale ~3600, momentum compounds across chunks).
- Fix that works (relaunched, finite): unit-normalize keys and queries, divide inner grads by chunk size, init memory projections at 0.02 std like the trunk. Persistence signal also improved 200x with normalized keys.
- Standing rule: any test-time-learning module must be stability-smoked at the REAL width, not a toy width (same lesson class as preflight 24d for dumps).
- Infrastructure lesson (fixed in trainer): a run whose every loss is non-finite used to complete silently with exit 0 and zero output; `vanilla_trainer` now prints a loud WARNING per skipped non-finite step batch.

## 2026-07-03: dynamic-shape batching on MPS

- Tested: SFT loop with per-batch max-length padding (every batch a new shape) on PyTorch MPS.
- Result: ~200 tok/s versus ~4,858 tok/s benchmarked at fixed shapes on the same model and box, a 23x slowdown. MPS recompiles kernels per new shape.
- Rule: all MPS training uses fixed or bucketed shapes (small fixed set of padded lengths).
