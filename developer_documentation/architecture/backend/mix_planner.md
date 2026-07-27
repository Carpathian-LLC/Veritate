# Corpus mix planner

## What it is

Turns a set of selected corpus stems plus a byte target into the weighted mix a trainer consumes. Output is the canonical multicorpus spec string (`stem:0.45,stem:0.30`) that `--corpus` already accepts, plus the per-source arithmetic that produced it. One module owns "what is a good mix": the dashboard, the REST API, and [build_base_corpus.py](../../../veritate_mri/tools/build_base_corpus.py) all call it instead of carrying ratio tables.

## How it works

Module: [mix_planner.py](../../../veritate_mri/training/mix_planner.py). Route: `POST /corpus/mix/plan` ([corpus_routes.py:97](../../../veritate_mri/routes/corpus_routes.py#L97)).

`plan(stems, target_bytes, profile=None, max_epochs=None, model_params=None, weights=None)` ([mix_planner.py:201](../../../veritate_mri/training/mix_planner.py#L201)) returns `{ok, spec, sources[], warnings[], bytes_planned, inputs}`. Each source row carries `stem, label, topic, weight, bytes_drawn, bytes_available, epochs` and the corpus's recommended param band. `inputs` echoes every argument, so a plan is reproducible from its own output.

Four steps:

1. **Source metadata** (`_sources()`, [mix_planner.py:91](../../../veritate_mri/training/mix_planner.py#L91)). Topic, label, and param band come from the corpus library catalog (`corpus_sync.catalog()`); available bytes come from the installed `.bin` when present ([readers/corpus.py](../../../veritate_mri/readers/corpus.py) `resolve_paths`), else the catalog's declared `size_train`, else zero.
2. **Base weights.** Explicit `weights` (`{stem: weight}`) are normalized and used as given. Otherwise the intent profile's target share per topic is renormalized over the topics the selection actually covers, and each topic's share is split across its corpora in proportion to available bytes (`_profile_weights()`, [mix_planner.py:120](../../../veritate_mri/training/mix_planner.py#L120)). With no profile at all (`corpus_mix_default_profile` blank), weights are size-proportional, matching multicorpus's implicit form.
3. **Epoch cap** (`_cap_epochs()`, [mix_planner.py:144](../../../veritate_mri/training/mix_planner.py#L144)). No source may be drawn more than `max_epochs` times over its own bytes. Over-cap sources are clamped and their surplus is water-filled into the sources with headroom, repeating until stable. This is what stops a 1.2 MB corpus from becoming 6% of a 5 GB mix (256 epochs).
4. **Emit.** Weights are rounded to `WEIGHT_DECIMALS` with the residual parked on the heaviest source, so they sum to exactly 1; rows sort by `(-weight, stem)`; zero-weight sources are excluded from the spec but kept in `sources` with a warning.

Warnings never mutate a plan silently: an unavailable source, a source held at the cap, a target beyond total capacity (`bytes_planned` then stops at `sum(available) * max_epochs`), a profile topic no selection covers, and a corpus outside its recommended param band when `model_params` is supplied all produce a warning string.

Profiles are data, not code: [veritate_mri/data/corpus_mix_profiles.json](../../../veritate_mri/data/corpus_mix_profiles.json), or any path in the `corpus_mix_profiles_path` setting. A profile is `{topics: {topic: share}, unlisted_topic_share, stems}`; `stems` is the default source list `profile_stems()` hands callers that supply none.

Determinism: sources are walked in caller order, the water-fill is a fixed-point loop, rows sort with a total order, and there is no RNG. Same inputs give a byte-identical plan.

## Dependencies

- [veritate_core/plugin/multicorpus.py](../../../veritate_core/plugin/multicorpus.py) — spec separators and the parse the emitted string must satisfy.
- [training/sync/corpus_sync.py](../../../veritate_mri/training/sync/corpus_sync.py) — catalog (topic, sizes, param bands).
- [readers/corpus.py](../../../veritate_mri/readers/corpus.py), [readers/paths.py](../../../veritate_mri/readers/paths.py) — stem resolution, profiles path.
- [runtime/settings.py](../../../veritate_mri/runtime/settings.py) — `corpus_mix_max_epochs`, `corpus_mix_default_profile`, `corpus_mix_profiles_path`.
- Tests: [tests/training/test_mix_planner.py](../../../tests/training/test_mix_planner.py).

## Pitfalls

- The planner trusts its callers (rule 112). `POST /corpus/mix/plan` is the only validating layer; a direct Python caller passing a `weights` dict that misses a stem gets a `KeyError`.
- A profile share for a topic no selected corpus covers is redistributed, not honored. Check `warnings` before trusting that a plan reflects the profile's intent.
- `bytes_planned` is `min(target_bytes, sum(available) * max_epochs)`. A plan can be smaller than the target on purpose; the spec weights stay relative and always sum to 1.
- Availability prefers the installed file over the catalog's declared size, so a partially downloaded corpus plans against what is really on disk.
- Catalog stems only carry topics when the catalog says so; a bundled or user-source stem with no topic falls into the profile's `unlisted_topic_share` bucket.
