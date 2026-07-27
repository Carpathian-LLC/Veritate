# eval sets

## What it is

The graded question sets every per-checkpoint smartness score is measured against, and the platform capability that regenerates them. Builders live at [veritate_mri/training/builders/eval/](../../veritate_mri/training/builders/eval/); the package `__init__` owns the rebuild; the route is `GET`/`POST /eval_sets` in [runs_routes.py](../architecture/backend/routes.md).

## How it works

- `BUILDERS` in [__init__.py](../../veritate_mri/training/builders/eval/__init__.py) names the five builders in run order: `build_grade_evals`, `build_comprehension_probe`, `build_grammar_eval`, `build_math_eval`, `build_reasoning_eval`.
- `rebuild_all()` imports each module and calls its `main()`, returning one `{builder, rc}` row per builder. A builder that raises propagates to the caller.
- Outputs land under `veritate_mri/data/eval/grade/`: grade `.bin` bands, `comprehension_<level>.json` and `comprehension_hard_<level>.json`, plus the `grammar/`, `math/`, and `reasoning/` tier `.jsonl` files. All paths come from `readers.paths`.
- Every builder is seeded, so a rebuild with unchanged sources is reproducible.
- `POST /eval_sets` claims a module-level job flag, starts a daemon thread, and returns `{ok, running, known_builders}`; a second POST while one runs returns `409`. `GET /eval_sets` reports `{running, started, finished, error, builders}`. Same claim plus status-poll shape as the eval_deep endpoints in the same module.
- Dashboard control: the **rebuild eval sets** button in the deep-eval panel (`#evalSetsRebuild` in `index.html`, `rebuildEvalSets()` in `index.js`) posts and then polls the status endpoint.

## Dependencies

- `readers.paths` for every output path and for `GRADE_EVAL_SOURCES_ROOT`, the hand-authored passages the grade and comprehension builders read.
- [checkpoint_probe.py](../architecture/backend/checkpoint_probe.md) consumes these files at every checkpoint; the eval sets are its grading key.

## Pitfalls

- A rebuild changes the grading key. Scores from earlier checkpoints stay comparable only while the sets are unchanged; regenerate deliberately, not routinely.
- Adding a probe means adding its builder module name to `BUILDERS`, or the route silently leaves that set stale.
- The comprehension builders sample distractors from the source passages; editing a source passage changes items across every band that uses it.
- Builders print their report to stdout, which is the server log for a route-driven rebuild; the per-builder result is in the status payload.
