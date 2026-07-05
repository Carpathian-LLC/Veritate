# research failures (kill list)

Falsified approaches. Each entry: what was tested, the measured result that killed it, and the retry condition. An approach on this list is dead until its retry condition is met. Entries are research outcomes, not bug fixes.

## 2026-06-27: from-scratch dense byte-level 2.5B coder

- Tested: `veritate_3b` (2.52B dense, vocab=256) pretrained from scratch on distilled Python chat corpus, dashboard run.
- Result: val loss plateaued near 1.0 by 205M tokens, zero code ability, looping English. Model deleted.
- Cause: compute wall. Competitive coding needs tens of billions of tokens; box delivers ~550 tok/s at 2.5B.
- Retry condition: a training-efficiency lever of 10x or more at equal quality, measured on this box.

## 2026-07-01: cheap byteification of a subword coder

- Tested: Qwen2.5-Coder-0.5B converted to vocab=256 (embed/head swap, exact 256/256 warm-init), continue-trained on 10 MB distilled Python, 2000 steps.
- Result: HumanEval pass@1 = 0/30 = 0.0 percent on held-out problems. In-distribution samples looked correct (memorized surface patterns); held-out eval exposed them.
- Cause: 1-byte-per-step granularity is an unseen input distribution for the pretrained body; recovery needs billions of adaptation bytes (Bolmo needed ~49B tokens with patching). Tiny corpus cannot buy it.
- Retry condition: patching front-end (body sees ~4 bytes per step) plus GB-scale diverse corpus, or frozen-body codec route. Full writeup: `bigmodel_byteification_handoff.md` (repo root).
- Standing rule from this failure: never believe in-distribution samples; only held-out executed benchmarks count.

## 2026-07-03: backprop-free learning rules for language modeling

- Tested: literature verdict (Forward-Forward arXiv 2301.01452, predictive coding 2212.00720 / 2308.07870, Mono-Forward 2501.09238, surveys through 2026).
- Result: no backprop-free rule has reached backprop parity on real language modeling at any scale. Demonstrations stop at MNIST/CIFAR/toy translation, with worse perplexity and higher convergence cost.
- Retry condition: a peer-reviewed result showing LM parity at 100M params or above.

## 2026-07-03: ternary (BitNet) as a training-cost lever below 3B

- Tested: literature verdict (BitNet b1.58 2402.17764, BitNet Reloaded 2407.09527, 2B4T 2504.12285).
- Result: below ~3B params ternary needs roughly 2x hidden width to match fp16 quality, and QAT adds training cost. It is a decode/energy lever, not a training lever.
- Standing use: keep ternary at EXPORT time (engine ternary path exists). Retry condition for training: parity evidence at or below 1B without the width penalty.

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
