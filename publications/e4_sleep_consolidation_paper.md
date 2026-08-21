# Sleep Consolidation Moves Novel Facts into the Weights of a 200M Byte-Level Model on Consumer Hardware

**Sam Malkasian**
Carpathian LLC
Draft — 2026-08-21

## Abstract

Stateless serving discards everything a language model is told: a fact stated in one session must be re-injected as context in every later one. We test whether short, low-learning-rate "sleep" consolidation runs can instead move novel facts into a model's weights, on consumer hardware, without damaging its existing abilities. Starting from a 200M-parameter byte-level model (vocab 256, GLA hybrid recurrent architecture), we ran three nightly consolidation sessions totalling ~800 optimizer steps (~3.5 h of GPU time on a single Mac Studio, M3 Ultra) over a corpus of 50 never-before-seen person→attribute facts, each templated into 20 augmentations covering both directions, mixed 3:1 with generic chat rehearsal at a constant learning rate of 5e-6 with an assistant-only loss mask. Closed-book greedy recall of the 50 facts rose from 0/50 (both directions) to a peak of 45/50 forward and 49/50 reverse at step 700, following a sigmoidal dose-response curve (6/50 at step 300, 45–47/50 at step 600). Reverse recall matched or exceeded forward recall at every checkpoint, indicating that both-directions training mitigates the reversal curse at this scale. Forgetting stayed inside a pre-registered +2% validation-bits-per-byte budget through step 700 (+1.50%); at step 800 the budget was crossed (+2.21%) and an automated tripwire stopped training — recall had already plateaued (45→46 forward), so past the sigmoid body additional dose bought forgetting, not memory. A secondary finding: consolidation moved a two-turn "what did you just say" state-recall test from 0/6 (parent model, all abstentions) to 3/6 (consolidated child, zero hallucinated answers), suggesting fact-QA consolidation unteaches a trained abstention reflex in favour of answering from held state. Results are from one model scale, one fact schema, and n=50 facts; 7- and 30-day retention quizzes are pre-registered and pending. Acquisition is demonstrated; retention is not yet.

## 1. Introduction

A deployed language model is amnesic by construction. Its weights are frozen at ship time; anything a user tells it lives only in the context window and evaporates when the session ends. The standard workaround — retrieval-augmented re-injection of prior exchanges — treats this as a search problem. We treat it as the problem itself: a system you must tell twice has not remembered, it has been reminded. The target property is *tell-it-once*: information stated to the model once should later be recalled closed-book, from the weights, with no context assistance.

The complementary learning systems (CLS) view of biological memory suggests an architecture for this: a fast store acquires experience during the day, and slow cortical weights absorb it during sleep, when replay of the day's activity trains the cortex at low intensity without catastrophic interference. Mapped onto a language model, the fast store is recurrent working state and logged experience; the slow store is the weights; and sleep is a short, low-learning-rate fine-tuning run over the day's material mixed with rehearsal of old material, executed during idle time on the same consumer machine that serves the model.

This paper reports the first controlled experiment (E4 in our program) on the sleep half of that design. The questions are narrow and quantitative:

1. Does short low-LR consolidation move genuinely novel facts into the weights of a small model, measured by closed-book recall?
2. What is the dose-response curve — how much recall does each unit of training buy?
3. Where does forgetting of prior capability bind the process, and can a pre-registered automated rule find that ceiling without human judgment?
4. Does training each fact in both directions defeat the reversal curse at this scale?

All experiments run on consumer hardware: acquisition on one Mac Studio (M3 Ultra), with a companion feasibility benchmark on a deliberately weak always-on x86 box. The model is byte-level (vocab 256), 200M parameters, with a hybrid trunk of gated-linear-attention (GLA) recurrent blocks and local attention. 200M is the falsifier bench, not the target: the mechanisms under test are chosen to be scale-free.

## 2. Related work

**Complementary learning systems and sleep replay.** The CLS framework (McClelland, McNaughton and O'Reilly; cited by name — no arXiv ID) holds that mammalian brains avoid catastrophic interference by pairing a fast hippocampal learner with slow cortical consolidation driven by replay during sleep, and that replay reactivates the day's *neural activity*, not the day's raw stimuli (Wilson and McNaughton; by name). Our nightly recipe is a direct engineering transcription: the day's material plus rehearsal, at low intensity, while the system is idle. Recent multi-timescale architectures (the CLS/HOPE line; by name) show frequency-spectra of interacting memory modules working in single networks at 340M–1.3B parameters.

**Linear attention as fast-weight memory.** Schlag, Irie and Schmidhuber (arXiv:2102.11174) showed linear attention is a fast weight programmer: its state matrices are an outer-product associative memory. Our model's GLA state is exactly such a store, which grounds the program's fast tier; the delta-rule line (DeltaNet, arXiv:2406.06484; Gated DeltaNet, arXiv:2412.06464) makes that state an in-place-editable regression memory, and Titans (arXiv:2501.00663) demonstrates surprise-gated MLP memory at 170–760M parameters — our size regime. This paper concerns the *slow* tier, but the fast tier defines what sleep must consolidate.

**Continual pretraining and replay.** Ibrahim et al. (arXiv:2403.08763) establish the recipe we adopt for the slow tier: a modest fraction of generic replay data (5–25%) plus a constant low learning rate that is never re-warmed suffices to continue training without catastrophic forgetting. Our 25% rehearsal share and flat 5e-6 LR are that recipe applied at small scale.

**Knowledge injection and its failure modes.** Berglund et al. (arXiv:2309.12288) document the reversal curse: models trained on "A is B" fail to recall "B is A". Allen-Zhu and Li (arXiv:2309.14316) show a fact stated once is often memorized but unextractable, and that ~10–30 paraphrased augmentations at learning time make it extractable. Both results shaped our corpus: every fact is templated ~20 ways, in both directions. Model-editing methods (ROME, MEMIT; by name) are pre-registered skips for the lifelong setting: reported collapse after roughly 250 sequential edits, and their localization assumptions target transformer MLP layers our architecture does not share. Likewise skipped on published evidence: EWC/SI (dominated by plain replay at tested scales) and LoRA as a forgetting cure (worse than full fine-tuning on TRACE-style continual benchmarks; large retrieval-QA losses in the comparison reported alongside gradient-sparse memory-slot fine-tuning, arXiv:2510.15103, which is the strongest published forgetting number and remains future work for our architecture).

**Self-directed study.** SEAL (arXiv:2506.10943) showed a model's self-authored study data can beat stronger-teacher-authored data for knowledge incorporation, which motivates the program's eventual self-generated augmentation path; in this experiment augmentations are templated in-house.

## 3. Method

**Base model and fork.** The subject is `wren1_5`, forked from checkpoint `wren1_3@3000` — a 200M-parameter byte-level model (vocab 256) with a hybrid GLA recurrent trunk, previously adapted for cross-window state carry and serving as our streaming-lane base. The fork's pre-consolidation exam score is exactly 0/50 in both directions: the facts are novel by construction.

**Fact set.** 50 synthetic person→attribute facts about invented people: 25 residences ("X lives in TOWN") and 25 occupations ("X works as a JOB"). Facts are novel — no overlap with training data — and each is expanded into 20 templated augmentations spanning question-answer, statement, and dialogue forms, in *both directions* (person→attribute and attribute→person), following the ~10–30-augmentation requirement of Allen-Zhu and Li and the both-directions prescription implied by the reversal curse.

**Consolidation recipe.** Each night is a short fine-tuning run with:

- data mix `fact_sft:0.75, mixed_chat:0.25` — the fact corpus against 25% generic chat rehearsal drawn from the model's own base corpus (the Ibrahim et al. replay band);
- constant learning rate 5e-6, no warmup, never re-warmed across nights;
- loss masked to assistant tokens only;
- dense checkpointing, with the closed-book exam and validation bits-per-byte (bpb) measured per checkpoint.

**Forgetting budget and tripwire.** Pre-registered before night 1: cumulative degradation of held-out `mixed_chat` validation bpb must stay within +2% of the fork baseline (0.82396), giving a kill line of 0.840. From night 3 the check ran as an automated per-checkpoint tripwire that stops the trainer the moment the kill line is crossed — the campaign's end is a rule, not a judgment call.

**Evaluation.** Closed-book exam: greedy decoding, no context injection, keyword-graded, both directions, all 50 facts, at every checkpoint. Capability safety: bpb on three held-out validation bins (`mixed_chat`, `veritate_chat`, `hansard`) and a fixed 30-prompt behavioural ladder scoring loop rate, turn closure, median reply length, identity, and grounded retrieval, compared to the fork parent's anchor values. Grading is always bare-greedy — serving-time decode guards are never active during measurement.

**Hardware.** All training on one consumer Mac Studio (M3 Ultra) at ~14.3 s/step; three nights totalling ~800 steps ≈ 3.5 h of GPU time. A companion benchmark ran the identical recipe on a deliberately weak always-on box (i7-9700T BIOS-clamped to 800 MHz, AVX2-only): ≥920 s/step, i.e. ≥65–80× slower, establishing the recipe's floor-hardware cost.

## 4. Results

### 4.1 Dose-response

| step | fwd | rev | mixed_chat val bpb | note |
|---|---|---|---|---|
| 0 | 0/50 | 0/50 | 0.82396 | baseline |
| 300 | 6 | 6 | 0.82608 (+0.26%) | night 1: toe of the curve |
| 600 | 45 | 47 | 0.83626 (+1.49%) | night 2: sigmoid body |
| **700** | **45** | **49** | **0.83632 (+1.50%)** | **peak — closeout checkpoint** |
| 800 | 46 | 47 | 0.84217 (+2.21%) | kill line crossed; auto-stopped |

Acquisition is sigmoidal, not linear. Night 1 (300 steps) bought 6/50; the curve then went vertical — within night 2, forward recall moved 6→26→38→45 across steps 300–600. Judged at the toe of the curve, the recipe would have looked like a failure; judged at step 600 it is a 90% acquisition mechanism. Intermediate failure modes en route were ordered: abstention ("I don't know") → fact-shaped-wrong → correct template with in-set attributes cross-bound to the wrong people → correct bindings. Format and vocabulary are absorbed before bindings lock; cross-binding errors are a mid-curve stage, not a terminal failure mode.

Past the sigmoid body, dose stopped buying memory. From step 700 to 800, forward recall moved 45→46 while `mixed_chat` bpb jumped +1.50%→+2.21%, crossing the kill line; the tripwire stopped the run automatically. The forgetting budget binds at ~700 consolidation steps at LR 5e-6 for this recipe. `wren1_5@700` is the ship checkpoint: 94/100 directional recalls from weights alone.

### 4.2 Reversal curse

Reverse recall (attribute→person) equalled or exceeded forward recall at every checkpoint: 6/6 at 300, 47 vs 45 at 600, 49 vs 45 at 700. A both-directions training corpus fully defeated the reversal curse in this setting — the direction that is normally catastrophically worse was, here, the *better* one throughout. We attribute this to the corpus, not the architecture, and note it required no mechanism beyond writing each fact both ways.

### 4.3 Capability safety

At the step-700 closeout checkpoint, against the fork parent's anchors:

| metric | @700 | anchor / budget | verdict |
|---|---|---|---|
| mixed_chat val bpb | 0.83632 (+1.50%) | +2% budget (kill 0.840) | inside budget |
| veritate_chat val bpb | improved (2.003 vs 2.071 @600) | — | improved |
| hansard val bpb | +1.14% (@600) | — | inside budget |
| identity | 1.00 | 1.00 | held (dipped 0.83 night 1, recovered) |
| loop rate | 0.17 | 0.20 | held |
| turn closure | 0.97 | — | held |
| median reply | 161 B | — | held |
| grounded retrieval | 0.25 | 0.38 | cleared as noise (below) |

The one apparent regression — grounded retrieval 0.25 vs the 0.38 anchor — was resolved by a per-item A/B against the parent on the same 8 prompts: 7/8 items produced identical outcomes and exactly one flipped, i.e. n=8 noise. The failing items (two-entity selection) fail on the parent too; they are a pre-existing capability gap, not consolidation damage. The safety record of the campaign is clean: no metric shows systematic regression from consolidation.

Forgetting accumulated *across* nights even though each night individually passed: +0.26% → +1.49% → kill. A per-night check would have approved a night-3 that the cumulative slope already doomed; the budget must be managed per campaign, with the tripwire evaluated per checkpoint.

### 4.4 Per-kind analysis of residual misses

Forward misses at step 700 concentrate in occupations: jobs 21/25 vs residences 24/25. Four of five forward misses are job facts, and the missed occupations are rare words — farrier, milliner, potter, mapmaker. The model answers these in perfect trained format with the *wrong* occupation (e.g. "doorman" for the milliner, "archivist" for the farrier): the template landed, the binding did not. Three job facts never landed forward at any of the eight checkpoints; two landed and were later lost. Reverse recall is near-saturated (49/50), so the deficit is direction-specific extraction, not storage. The actionable recipe implication: augmentation count should scale with the object word's corpus rarity rather than being flat per fact.

### 4.5 Secondary discovery: consolidation unteaches abstention toward state recall

Our program's acceptance test is behavioural: a two-turn conversation over the model's streaming (carried-state) path in which turn 2 — sent alone, with only recurrent state connecting it to turn 1 — asks "what did you just say?", with a fresh-state leak control per item. The fork parent `wren1_3@3000` scores 0/6: every reply is a trained abstention ("I don't know"). The consolidated child `wren1_5@700` scores 3/6 with zero leaks on the fresh-state controls — no hallucinated recalls. Consolidation on fact-QA appears to have taught answer-from-what-you-hold over the abstention reflex installed by earlier abstention training, and the gain expresses through the carried state at no hallucination cost. This was unplanned: the consolidation corpus contains no state-recall training at all. It suggests fact-QA sleep generalizes as a disposition ("consult what you hold and answer"), not only as stored content. n=6; we report it as a discovery to be replicated, not a validated result.

## 5. Limitations

- **n=50 facts, one schema.** All facts are single-relation person→attribute pairs from two kinds. Multi-relation facts, revisions of existing facts, and free-form knowledge are untested here (fact revision is a separate pre-registered experiment).
- **One model, one scale.** 200M parameters, one architecture, one fork point. The dose constants (~700-step budget at 5e-6, the sigmoid's location) are measured for this configuration only and should be expected to shift with scale, LR, and mix.
- **Acquisition, not retention.** Every number above is recall immediately after training. Whether the facts survive 7 and 30 days of further model life is exactly what the pre-registered retention quizzes will measure; the program's own falsifier (recall <80% at day 30) is still live. Similarly unknown: whether the +1.50% bpb cost anneals back under subsequent no-fact rehearsal.
- **Synthetic facts and templated augmentation.** Novel-by-construction facts flatter acquisition measurement (no partial prior knowledge) but the 20-template in-house augmenter is a pipeline a deployed system must replace with extraction from real conversation — a harder, noisier source.
- **Single seed.** Per our own reporting rules, sub-5% deltas on one seed are not claims; the headline numbers (0/50→45–49/50) are far outside that band, but the finer structure (e.g. 45 vs 46 forward at 700 vs 800) is within it.
- **The rare-word deficit is diagnosed, not fixed.** Rarity-scaled augmentation is a hypothesis pending its own run.

## 6. Pre-registered next steps

1. **Retention quizzes** at 7 and 30 days (2026-08-27, 2026-09-19): the untouched step-700 checkpoint re-examined closed-book by a frozen tool (`tools/e4_retention_quiz.py`, fixed fact file). Acquisition reference: fwd 45/50, rev 49/50. Falsifier: <80% recall at day 30.
2. **Rarity-scaled augmentation**: scale augmentations/fact with object-word corpus frequency; target the four never/lost job facts.
3. **Budget extension across nights**: per-night LR decay (5e-6 → 2e-6 → 1e-6, mimicking synaptic downscaling) or interleaved rehearsal-only nights, to test whether bpb cost anneals and the campaign ceiling moves.
4. **Trace consolidation (m2)**: the matched arm still unrun — replaying (context, carried-state, prediction) triples and training the weights to reproduce from bare context what the model could only do with state; if it matches fact-SFT recall without a fact-extraction pipeline, it is the cleaner mechanism.
5. **Replay and augmentation sweeps** ({5, 25, 50}% replay; {1, 5, 20} augmentations/fact): the flat-20/25% recipe validated first try, so these sweeps now measure efficiency, not existence.

The operational endpoint is already built: an idle-time sleep controller that scales dose to usage (new exchanges × steps-per-exchange), checkpoints densely, prunes automatically, and carries the same per-checkpoint val-bpb tripwire that ended this campaign. It ships disabled until retention is proven.

## References

Cited by arXiv ID where an ID exists in our research ledgers; otherwise by name only.

- Schlag, Irie, Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers.* arXiv:2102.11174.
- Yang et al. *DeltaNet: parallelizing linear transformers with the delta rule.* arXiv:2406.06484.
- Yang et al. *Gated DeltaNet.* arXiv:2412.06464.
- Behrouz et al. *Titans: Learning to Memorize at Test Time.* arXiv:2501.00663.
- Ibrahim et al. *Simple and Scalable Strategies to Continually Pre-train Large Language Models.* arXiv:2403.08763.
- Berglund et al. *The Reversal Curse: LLMs Trained on "A is B" Fail to Learn "B is A".* arXiv:2309.12288.
- Allen-Zhu, Li. *Physics of Language Models: Part 3.1, Knowledge Storage and Extraction.* arXiv:2309.14316.
- *Gradient-sparse memory-slot fine-tuning for fact injection* (Meta). arXiv:2510.15103.
- Zweiger et al. *SEAL: Self-Adapting Language Models.* arXiv:2506.10943.
- McClelland, McNaughton, O'Reilly. *Why there are complementary learning systems in the hippocampus and neocortex.* (by name)
- Wilson, McNaughton. *Reactivation of hippocampal ensemble memories during sleep.* (by name)
- ROME / MEMIT model-editing line. (by name)
- Kirkpatrick et al., EWC; Zenke et al., SI. (by name)
