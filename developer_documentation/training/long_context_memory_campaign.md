# Long-context chat memory — research campaign (Wave 4)

Goal: a small byte-level model that holds **full, long, coherent conversations without
degrading**, with the memory cost bounded (constant-state), plus a novel inference-time
memory mechanism. This is the "chat that stays coherent over a huge context" thrust.

## The winnable claim (stated honestly)

The hybrid trunk carries long-range memory in a **constant-size state** (O(1) per byte —
`model_recurrent.py::RecurrentMixer`, one `[d_k,d_v]` matrix per head). Memory cost does
NOT grow with conversation length. That is the real edge: a *small* model can hold a *huge*
conversation cheaply, which a KV-cache transformer cannot. The target is a tiny, fast byte
model that coherently holds a 100k+ byte conversation because its memory is engineered not
to forget.

Not claimed: Opus-level quality. The single-M3-Ultra compute wall stands (`failures.md`
2026-06-27). This campaign is a *capability* win (long coherent context on a small fast model),
not raw-quality parity.

## The one hard problem, named

Constant state is a lossy summary, so it forgets. In the current mixer the forgetting is
**uniform per head**: `la = -softplus(a_proj(x))` is one scalar decay per head applied to the
whole state matrix (`model_recurrent.py:76`, `:91`). Everything fades at the same rate
regardless of importance. "So it doesn't degrade" == flatten the recall-vs-length curve.

## Falsifier — the conversation-needle benchmark (built first, GPU-free harness)

Generate synthetic multi-turn conversations of controlled byte-length L. Plant a fact
(a "needle": an access code, a name, a constraint) at turn k near the start. At the end, ask
a question whose answer requires the needle. Score:
- **recall(L, distance)**: exact-match accuracy of the needle answer vs total conversation
  length L and vs plant→query byte distance. The *degradation curve*.
- **coherence**: self-consistency probes (does the model contradict an earlier stated fact
  later in the same conversation).

A mechanism passes only if it measurably flattens recall(L) relative to the plain-hybrid
baseline M0. No curve improvement over M0 at matched params/steps → killed. The harness
scores any checkpoint (10M or 80M) so mechanisms are validated cheaply at 10M first.

## Mechanisms (each a matched-arm A/B, pre-registered kill conditions)

| id | mechanism | change | prior | kill condition |
|----|-----------|--------|-------|----------------|
| M0 | plain hybrid | baseline (current recurrent mixer) | — | (reference curve) |
| M1 | **delta-rule state** (Gated DeltaNet) | state update becomes error-correcting overwrite `S ← S(αI − βkkᵀ) + βkvᵀ` instead of uniform decay+add; selectively deletes a stale key's slot before writing | HIGH (grounded: Gated DeltaNet beats Mamba2/GLA on recall) | if recall(L) not flatter than M0 by the needle margin at matched params/≥2 seeds, OR throughput <70% of M0 on MPS |
| M2 | **pinned memory register** (invention) | reserve a small FIXED set of state slots exempt from the decay gate; a learned controller writes salient content into pins (StreamingLLM sinks, but inside the recurrent state) | MED | if pinned recall no better than M0 at the same slot budget, OR pins collapse to unused (share ≈ 0) |
| M3 | **surprise-gated rehearsal** (invention, inference-time) | track per-turn prediction surprise (existing surprise hooks); when it spikes on back-referencing content, re-encode a learned compressed digest of the conversation-so-far and re-inject it into the state — closed-loop memory refresh triggered by measured forgetting | MED | if triggered rehearsal does not raise recall vs the no-rehearsal same checkpoint, OR fires on >X% of turns (cost blowup) |
| M4 | conversation self-retrieval (stretch) | RAG over the model's own turns: chunk+embed past turns, retrieve relevant ones back into the local window on demand (platform RAG infra) | robust but heavy | only pursued if M1–M3 leave a recall gap |

NOT re-run: naive Titans-style test-time fast-weight memory — already killed at 10M (E4/E4b,
`failures.md`). M3 is explicitly a different mechanism (surprise-triggered rehearsal
of a digest, not gradient learning of fast weights at test time).

## Huge context (streaming to unbounded length)

- Local blocks: sliding-window attention + attention sink (StreamingLLM) so per-byte local
  cost is bounded and the stream never OOMs; the constant-state global carries long range.
- Length curriculum: extend training seq over the run so the state *learns* to carry
  information far, not just architecturally can. Fixed shapes per stage (MPS 24c).

## Execution order (GPU-aware)

1. (GPU-free, now) build the needle harness; build M1+M2 as gated options on RecurrentMixer,
   MPS-safe, dump-verified at real shape, baseline (M0) path bit-identical when flags off.
2. (when GPU free) 10M A/B: M0 vs M1 vs M2 on the needle curve, ≥2 seeds for sub-5% margins.
3. fold the winning memory mechanism into the 80M hybrid chat model.
4. build + test M3 (surprise-gated rehearsal) on the winning checkpoint at inference.
5. add streaming (sliding window + sink) + length curriculum for the huge-context delivery.

Results and kills land here and in `successes.md` / `failures.md`.

## 2026-07-06: first real degradation curve (chat80m SFT checkpoint, M0-class gla state)

Measured via streaming state-carry, 12 trials/distance, 3 needle kinds, greedy decode
(`experiments/v2/longctx/chat80m_needle_curve.json`):

| plant->query bytes | recall |
|---|---|
| ~190 | 0.92 |
| ~480 | 0.25 |
| 2k / 8k / 32k | 0.00 |

Coherence probe: recall_a 0.0, contradiction_rate 0.25.

Two distinct failures, so the campaign now tests two hypotheses:
- **H1 (mechanism)**: past-window recall is 0 — the uniform-decay state carries nothing usable
  across windows. Tested by the M0/M1/M2 race (arms differ only in `state_rule`).
- **H2 (training pressure)**: recall collapses 92->25 percent INSIDE the window, where local
  attention can see the fact — ordinary chat data never demands long-range exact recall, so
  retrieval is never trained. Tested by a fourth arm: gla + `chat_recall_v1` (synthetic
  recall-pressure corpus, built with a contamination guard vs the eval templates — see
  `experiments/v2/longctx/README.md`) as a minority mix component.

If H1 arms flatline equally, the mechanism is not the bottleneck at this scale; if the H2 arm
alone lifts the curve, data is the lever and the mechanisms retest on top of recall-trained
baselines. Both can be true (mechanism helps only once the skill is trained).

## 2026-07-06: 10M race interim — floor effect + one divergence

- **m0gla (baseline) needle curve is 0.00 at EVERY distance** including ~190 B in-window,
  where chat80m scores 0.92. Coherence contradiction 0.0 (it does not even echo a wrong
  fact — retrieval behavior is entirely absent). A 10M/12k-step/pure-chat model sits on a
  recall floor; mechanism arms cannot show recall gains against 0.00. The discriminating arm
  is therefore m0recall (H2): pre-registered decision tree —
  (a) m0recall lifts off the floor at short distance => H2 confirmed; the mechanism race
      (delta/pinned) reruns ON TOP of the recall mix, where differences can show;
  (b) m0recall stays at 0.00 => 10M cannot learn retrieval at this budget regardless of data;
      the race moves to the 80M scale (cheapest: resume chat80m with a recall-mix anneal and
      re-measure its curve — it already has the skill at short range, the question is whether
      recall data flattens the 92->25 percent in-window collapse).
- **m1delta DIVERGED at step ~4420/12000** (non-finite loss for the rest of the run; precursor
  grad-norm creep 0.32->0.57 over the final ~40 finite steps; LR ~4.45e-4 near cosine peak).
  Passed all pre-launch smokes (CPU+tiny-MPS finite, dump battery, oracle exactness) — the
  instability only appears thousands of real steps in. Throughput was fine (85% of gla).
  Autopsy running (root cause + principled stabilizer vs kill). m0gla and m2pinned trained
  clean, so the instability is delta-specific.
- **m2pinned trained stably** (zero non-finite skips, 77k tok/s = 97% of gla) but with a real
  LM-quality tax: tail-10 val 0.7637 vs m0gla 0.6907 (+0.073, ~7 sigma). Needle curve pending;
  pins must buy dramatic recall retention to survive that tax.

## 2026-07-06: race results — floor effect decomposed into a TRANSFER gap; H2 confirmed at 10M

- All three 10M arms (m0gla, m2pinned, m0recall) scored 0.00 on needle_bench at every
  distance. First read: retrieval is absent below some scale threshold. WRONG — the
  in-distribution diagnostic (greedy answer completion on held-out chat_recall_v1 VAL
  conversations, last-token exact match, 40 trials each) shows:
  **m0recall 19/40 = 0.47 vs m0gla 2/40 = 0.05.**
- Corrected finding: **recall-pressure data teaches retrieval even at 10M (10x lift); what
  10M lacks is TRANSFER** — the contamination guard makes needle_bench's fact surface forms
  (hyphenated codes, 5-digit dollar amounts) alien to the everyday-facts training pool, and
  10M cannot bridge that. "Sub-threshold capability" is falsified; the correct model is
  "narrow skill, no surface-form generalization at 10M".
- m1delta divergence root-caused (see `m1delta_divergence_analysis.md`): fp32 catastrophic
  cancellation in the whole-chunk WY nilpotent inverse under beta saturation — the mechanism
  itself was never fairly tested (the saturated head was numerically broken from ~step 800).
  Block-recursive inverse shipped (same exact math, bounded intermediates), triple-verified.
  Retry as m1delta2 from scratch; MPS throughput vs the 70% line is the one open risk.
- Revised race design: the mechanism A/Bs (m1delta2, m2pinned re-scored) run at 10M with
  **in-distribution recall (chat_recall_v1 val) as the scoreboard** — a 0.47 baseline gives
  mechanisms something to move at 10M cost. needle_bench (out-of-distribution surface forms)
  remains the gold standard, applied at 80M where transfer is plausible.
- At-scale H2 test in flight: chat80m recall-SFT phase 4 (25% chat_recall_v1, 4k steps,
  1e-5 WSD). Scoreboard: needle curve step_52000 vs the step_48000 baseline (0.92/0.25/0.00) —
  does recall data flatten the in-window collapse AND transfer to alien fact forms at 121M?

## 2026-07-06: at-scale H2 verdict — late-phase recall-SFT KILLED (made it worse)

Needle recall step_52000 vs step_48000: ~190 B **0.92 -> 0.08**, ~480 B 0.25 -> 0.17,
past-window 0.00 -> 0.00. Contradiction 0.25 -> 0.0 (answers from the template distribution,
no longer echoes context at all). The narrow templated corpus REPLACED the emergent
copy-from-context behavior instead of augmenting it. Flagship rolled back to step_48000.
Full entry + retry conditions (small in-pretrain dose, surface-form diversity) in
`failures.md` 2026-07-06. M1 delta also killed on trainability (same-day entry).

Campaign state after wave-4 round 1: the surviving routes to degradation-free long chat are
(1) recall data IN PRETRAIN at low dose with diverse surface forms (retry conditions above),
(2) M3 surprise-gated rehearsal — inference-time, zero training interference by construction,
(3) M2 pinned register pending its fair test (m2pinned2, in flight), judged on in-distribution
recall vs m0recall 0.47, and (4) the streaming/length-curriculum training phase (state carry
across windows DURING training so past-window recall becomes trainable at all).
