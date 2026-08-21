# Technical note: consolidating novel facts into a 200M model's weights with low-LR "sleep" runs

*Sam Malkasian, Carpathian LLC — 2026-08-21*

Result: 50 never-seen person→attribute facts moved into the weights of a 200M byte-level model (vocab 256, GLA hybrid recurrent trunk) by ~700 steps of low-LR fine-tuning on one M3 Ultra Mac. Closed-book greedy recall, keyword-graded, no context injection: 0/50 → **45/50 forward, 49/50 reverse**, with held-out val bpb degradation capped at +1.50% and all chat-quality metrics held. Caveats up front: n=50 facts, one schema (25 residences, 25 occupations), one model scale, one seed; 7/30-day retention quizzes are pre-registered and pending. Acquisition proven, retention not yet.

## The procedure

1. **Fork** the serving checkpoint. Ours: 200M params, byte-level, recurrent hybrid trunk. Verify exam = 0/50 both directions before training (facts must be genuinely novel).
2. **Augment every fact ~20 ways, in both directions.** Templated QA, statement, and dialogue forms; person→attribute AND attribute→person. This is not optional — one-exposure facts are memorized-but-unextractable (arXiv:2309.14316) and single-direction training hits the reversal curse (arXiv:2309.12288).
3. **Mix 75% fact corpus / 25% generic rehearsal** drawn from the model's original training distribution (the 5–25% replay band of arXiv:2403.08763).
4. **Train at constant LR 5e-6, no warmup, never re-warmed**, loss masked to assistant tokens. Our shape: batch 48, seq 1024 × 4 chunks, bf16, ~14.3 s/step on the M3 Ultra.
5. **Checkpoint densely and grade every checkpoint**: the closed-book exam (both directions, bare greedy — no decode guards during measurement) plus val bpb on held-out generic text.
6. **Pre-register a forgetting budget and automate it.** Ours: mixed-domain val bpb ≤ +2% over the fork baseline (0.82396 → kill line 0.840), checked per checkpoint by a tripwire that calls the trainer's stop endpoint. It fired at step 800 (0.84217, +2.21%) and ended the campaign; no human in the loop.
7. **Select the peak checkpoint, not the last one.** Ours: step 700 (fwd 45, rev 49, bpb +1.50%) over step 800 (fwd 46, rev 47, budget blown).

## Full dose-response (50 facts, fwd/rev)

| step | fwd | rev | val bpb vs baseline |
|---|---|---|---|
| 0 | 0 | 0 | — |
| 300 | 6 | 6 | +0.26% |
| 600 | 45 | 47 | +1.49% |
| 700 | 45 | 49 | +1.50% (peak) |
| 800 | 46 | 47 | +2.21% (tripwire) |

## Three rules we learned

**1. Acquisition is sigmoidal — never judge at the toe.** 300 steps bought 6/50 and looked like a dead end; the next 300 bought 39 more (6→26→38→45). The visible mid-curve failure mode is diagnostic: correct trained format, in-set attributes, wrong bindings (right towns, wrong people). Format and vocabulary land before bindings lock. If you see cross-binding errors, you are mid-sigmoid: extend, don't redesign.

**2. Budget the campaign, not the night.** Forgetting accumulated +0.26% → +1.49% → +2.21% across three nights that each looked individually fine. A per-night check approves the night that kills you. Track cumulative drift against the fork baseline and tripwire per checkpoint once past half the budget. At 5e-6 the +2% budget bound at ~700 steps — past the sigmoid body, more dose bought forgetting, not memory (recall 45→46 while bpb jumped).

**3. Both-directions training beats the reversal curse.** Reverse recall ≥ forward at *every* checkpoint (49 vs 45 at peak). No architectural mechanism needed — just write every fact both ways at augmentation time. Corollary from our residuals: misses concentrate in rare-word attributes (farrier, milliner, potter, mapmaker — answered in perfect format with the wrong job; 3 never landed, 2 landed-then-lost). Flat 20 augmentations/fact under-serves rare object words; scale augmentation with object-word corpus frequency.

## Measured constants (this config; expect them to move with scale/LR)

- ~700 steps @ 5e-6 = the +2% forgetting ceiling; ~600 steps = 90–94% of a 50-fact set.
- Cost: ~800 total steps ≈ 3.5 h GPU on one M3 Ultra (14.3 s/step).
- Floor hardware: the same recipe on an i7-9700T clamped to 800 MHz (AVX2-only) runs ≥920 s/step, ≥65–80× slower — sleep-class consolidation wants a real consumer GPU, not an idle NAS-tier CPU.
- Safety at peak: identity 1.00, loop rate 0.17, turn closure 0.97; a grounded-retrieval dip was cleared as noise by per-item A/B against the parent (7/8 identical).
- Unplanned side effect worth checking in your own runs: a two-turn "what did you just say" state-recall test moved 0/6 (parent, all abstentions) → 3/6 (slept child, zero hallucination leaks). Fact-QA consolidation appears to unteach a trained abstention reflex toward answering from held state.

Open questions we've pre-registered rather than answered: retention at 7/30 days (frozen quiz tool, 2026-08-27 / 2026-09-19; falsifier <80% at day 30), rarity-scaled augmentation, per-night LR decay to stretch the budget, and whether replaying (context, state, prediction) traces can match fact-SFT without a fact-extraction pipeline.
