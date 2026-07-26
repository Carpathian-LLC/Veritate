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

## Dependencies

None beyond stdlib. Output consumed by any trainer via `--corpus concepts_v1`.

## Pitfalls

- Fixed small vocabulary and no invented entities: role-binding success must be validated on HELD-OUT (subject, verb, object) combinations, not seen ones, or it measures memorization not binding. A held-out action split is the required companion before any F4 claim.
- Random object-surface-relation pairing produces semantically odd but grammatical lines ("the shoe is in the floor"); acceptable for learning relation words, not for factual plausibility.
- Heavy combinatorial repetition drives loss very low on a tiny corpus (near-memorization at many epochs); expected for a concept-drill corpus, but read val as consolidation, not generalization to new vocab.
