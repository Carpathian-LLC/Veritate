# curriculum corpus (developmental child-concept data)

## What it is

A deterministic generator of simple, child-like concept data for tiny models (10M-class), built for IDEA 9 (developmental training). Teaches, in one mixed stream, the primitives a child learns first: naming/categories, properties, spatial relations, and actions. The action stage is the point: every event is stated four ways so roles cannot be read off word position.

## How it works

`veritate_mri/tools/build_curriculum_corpus.py`. Constants at module top hold the vocabulary (objects+categories+articles, colors, sizes, textures, surfaces, spatial relations, animate subjects, action verbs as base/present/past triples). Block builders each emit one short "picture-book page" about a concept with varied phrasing:

- `naming_block` — "A dog is an animal. What is a dog? A dog is an animal." (`_art` picks a/an correctly).
- `property_block` — color/size/texture assignment + a what-color question.
- `spatial_block` — object-surface-relation + a where question.
- `action_block` — the F4 role-binding seed: `"The dog chases the cat. The cat is chased by the dog. Who chases the cat? The dog does. What does the dog chase? The cat."` Active + passive + subject-question + object-question for one event, so subject vs object must be tracked, not guessed from position.
- `narrative_block` — 2-4 sentence mini story combining the above.

`build_stream` loops (seed=0) emitting a mix weighted toward action blocks (role binding is the hard target), shuffles blocks (local coherence kept inside a block, global order randomized), and fills to `--target-mb`. `write_split` holds out the trailing `VAL_FRACTION` (3%) as val. Output is raw UTF-8 bytes (vocab=256); no meta sidecar (rule 37).

Build: `python veritate_mri/tools/build_curriculum_corpus.py --out-train trainers/corpus/concepts_v1_train.bin --out-val trainers/corpus/concepts_v1_val.bin --target-mb 30`

### Stage 2 (`--stage 2`)

Harder material for the IDEA 9 growth experiment, added because a 16M model saturates stage 1. Every inventory is widened (44 objects, 18 animates with pronoun+possessive, 20 action verbs, 9 ditransitives, 16 places, 6 causal connectives) and five structures stage 1 lacks are added:

- `s2_ditransitive_block` — three roles in one event: `"The hunter shows the clock to the painter. The clock is shown to the painter by the hunter. Who shows the clock? ... What does the hunter show? ... Who receives the clock? ..."` The hardest role probe in the corpus (subject vs theme vs recipient).
- `s2_coref_block` — pronoun binding across sentences (`the girl ... she ... her`).
- `s2_negation_block`, `s2_comparative_block`, `s2_count_block`, `s2_multiclause_block` — negation, comparatives with the inverse relation stated, counting, and subordinate clauses with manner/time adverbials.

`build_stream_s2` (seed=SEED+2) emits one object block + one free block + one action block + one ditransitive block per loop. Held-out pairs come from `s2_holdout_verb_map` (Random(SEED+199)) over the 18 animates -> 306 pairs, so `--holdout-frac 0.3` yields **91** held-out role tests (vs 12 at stage 1).

Build: `python veritate_mri/tools/build_curriculum_corpus.py --out-train trainers/corpus/concepts2_ho_train.bin --out-val trainers/corpus/concepts2_ho_val.bin --target-mb 34 --stage 2 --holdout-frac 0.3 --test-out <jsonl>`

`_pp(verb)` supplies the past PARTICIPLE for passives via the `PARTICIPLES` irregular table, falling back to the simple past. Both stages use it.

## Dependencies

None beyond stdlib. Output consumed by any trainer via `--corpus concepts_v1`.

## Pitfalls

- Fixed small vocabulary and no invented entities: role-binding success must be validated on HELD-OUT (subject, verb, object) combinations, not seen ones, or it measures memorization not binding. Build the held-out split with `--holdout-frac 0.3 --test-out <jsonl>`: the chosen (subject, object) pairs are emitted declarative-only with one fixed verb, and their who/what questions become the test set.
- MEASURED 2026-07-25 (`conceptsho_10m`, 2000 steps, val 0.092): this corpus does NOT teach role binding at 10M. Held-out who 17% / what 0%, and CONTROL (pairs seen WITH questions) only 50% / 25%. The model learns to echo the noun from the question ("Who chases the boy?" -> "The boy does.") and to emit a fixed entity for object questions. Longer training made CONTROL worse (who 100% at step 500 -> 50% at step 2000). Treat a low val loss here as memorization of surface statistics, never as evidence of relational competence — score the held-out role eval instead. See `ideas.md` IDEA 9 "STAGE-1 RESULT".
- Random object-surface-relation pairing produces semantically odd but grammatical lines ("the shoe is in the floor"); acceptable for learning relation words, not for factual plausibility.
- Heavy combinatorial repetition drives loss very low on a tiny corpus (near-memorization at many epochs); expected for a concept-drill corpus, but read val as consolidation, not generalization to new vocab.
- **THE BIG ONE — never use this corpus (or any generated corpus) to measure model CAPACITY or growth benefit.** A procedurally-generated corpus has an entropy floor set by the complexity of its GENERATOR, not by its byte count. This generator is a few KB of Python, so once a model has learned the templates plus the inventories the remaining bytes are near-deterministic. Measured 2026-07-25: stage 1 floors at val ~0.0925 (16M, by step 1250) and **stage 2 still floors at ~0.121 despite 3x the block diversity** — widening the inventories did NOT buy proportional headroom. Unique-block percentage (22% -> 63%) is a misleading diversity metric: it counts unique STRINGS, not unique INFORMATION. Capacity/growth experiments belong on real text (`fineweb_edu`, `openwebtext10g`). What a saturating corpus CAN still measure is steps-to-target (approach rate to a shared floor).
- Passives need the past PARTICIPLE, not the simple past. The original stage-1 generator emitted "The cat is saw by the dog" / "is drew" for irregular verbs; fixed 2026-07-25 via `_pp()` + `PARTICIPLES`. The already-trained `conceptsho_10m` / `concepts_10m` were built with the buggy form — it did not affect the F6 role probe (which scores the who/what questions, not the passive) but their corpora do contain ungrammatical passives.
- Relations taking two landmarks ("between") must not be used where only one surface is named; `S2_SPATIAL_ONE` excludes it for the single-landmark blocks.
