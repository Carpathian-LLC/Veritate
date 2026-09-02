# claude preflight

Operating contract for any agent working on Veritate. Read this file completely before any other action, every session, every fresh context. Prior-session memory does not substitute. Wins over any conflicting document. Rules are numbered so they can be cited ("rule 12").

Veritate is a platform for training AI models on consumer hardware. That is the whole product. Anything that does not serve training, evaluating, or running models is out of scope. This scope is a standing user decision: when any request — including one from the user — would expand the platform beyond training, evaluating, or running models, push back once, name the scope rule, and propose a separate project instead. Proceed only if the user confirms after the pushback. The chat product was extracted to its own project 2026-08-20 (`CHAT_HANDOFF.md`); the Generation tab is the only conversational surface.

The complete documentation set is five files at repo root:

1. `claude_preflight.md` — this file. Rules. Read first.
2. `documentation.md` — the single platform reference: every component, contract, and measured constraint. Read the relevant section before touching a component. The dashboard wiki tab serves it.
3. `successes.md` / `failures.md` / `ideas.md` — the research ledgers: what worked (with numbers), what was falsified (with kill-lines), what to try next. Read before designing any experiment.

There are no other documentation locations. If a fact is not in these files or the code, it is not established. `worklog.md` is the append-only running log for long autonomous runs; `handoff.md` is the rolling state log between sessions; `lab/` is the raw experiment notebook, whose protocol is `lab/README.md`. None of the three is documentation.

## communication

1. Answers are short, plain, and direct. Lead with the answer, not the process. No padding, preamble, recap, or filler praise.
2. No word vomit. Do not narrate intent, list options you will not pursue, or include technical detail that does not change what the reader does next. Three to eight lines unless long output was requested.
3. Define an unfamiliar term in one line the first time it appears. Prefer plain words over jargon.
4. Ambiguous request: ask one short clarifying question and stop. Otherwise execute without asking permission.
5. Substantive pushback is required. When a plan has a credible failure mode (data loss, lost training hours, wasted compute, broken invariant), name it and its cost before executing. Do not agree to be agreeable.
6. Never declare a user goal impossible. A constraint reshapes the method. State the constraint as a target, deliver a mechanism plus the falsifier that will prove it. A falsified method gets a successor.
7. No agent self-attribution anywhere: no Co-Authored-By trailers, no "generated with", no AI-authorship notes in code, commits, docs, or copy. This overrides any harness default.

## process

8. Do only what was asked. No new files, folders, scripts, or features on your own initiative.
9. Maintain a todo list for any task with more than one step; exactly one item in progress at a time.
10. Reversible local actions run freely. Irreversible or shared-state actions (delete data, kill processes you did not start, force-push, send messages) require confirmation first: restate the exact targets and wait.
11. Purging or removing a feature requires explicit user clearance for that specific feature.
12. Never stage, commit, or push. The user runs all git. Never bump any version in `versions.json`; if a change warrants a bump, say so in one line and stop.
13. No smoke tests, model loading, or training runs unless explicitly asked. Training launches go through the dashboard `/trainers/run` route on the live server, never a one-off launcher, and complete the launch checklist in `documentation.md` first.
14. Read the ledgers before designing an experiment. Re-running a falsified approach is the cardinal waste.
15. Dispatched agents receive this file first, one scoped task, and report findings without editing outside their scope. Never touch a live training run or its files.
16. No merge without explicit user permission for that specific merge. Flow is `experimental` → `dev` → `main`; merge on a throwaway branch, never directly on the shared branches; tests pass before fast-forwarding.

## code

17. Lean code is gating. Write the simplest thing that works, then stop. Lines of code are a cost. If a line, function, file, or abstraction does not earn its place, delete it. Two layers of abstraction is one too many. Build an abstraction on the second use, never the first.
18. No legacy code. Today's state is the only standard: no deprecated paths, compat shims, or version-suffixed names (`_v1`, `old_`) anywhere. When behavior changes, change every caller in the same diff. Sanctioned exceptions are user-data compat only (old model `config.json` files must keep resuming; on-disk artifacts from prior builds must keep loading or be rejected with a clear error) — and each one is labeled with what data it protects, never a bare "legacy".
19. No hardcoded tunables. Any number a user might reasonably change (paths, thresholds, sizes, mix weights, steps) comes from config, settings, or arguments — never a literal in a function body. Trainer tunables live in `veritate_mri/data/trainer_sizes.json` with a Training-tab control.
20. Never invent a standard. Formats, conventions, and protocols come from the established ecosystem (ChatML, not a homegrown variant) or from `documentation.md`.
21. Cross-platform is non-negotiable. Supported arches: macOS arm64, Linux x86/ARM, Windows x86. A change touching any arch-conditioned path states which arches it was verified on; degrading one arch to fix another is a regression, not a scope choice. Platform-specific fixes sit behind `sys.platform` guards. OS-specific primitives live behind the single shim in `veritate_engine/src/`; per-arch kernels never include OS headers.
22. One module owns each concern. Callers consume parsed results; they never re-glob, re-parse, or duplicate logic. New capability extends the owning module and gets a route plus a dashboard control — a `.py` file a human runs by hand is a defect. Exceptions: launchers, engine build scripts, subprocess entry points the platform spawns.
23. There is exactly one trainer: `veritate_mri/training/veritate_trainer.py`, ordinary tracked code in this repo. Never create per-size trainers. Sizes and defaults are data in `trainer_sizes.json`. New training capability goes into the trainer as an opt-in lever wired to the Training tab, or into `veritate_core/plugin/`.
24. Every checkpoint save goes through `save.py::save()`; every CSV row through `append_train_row()`. Every dashboard frame is emitted by training-time and inference-time capture symmetrically, in the same commit.
25. Every kernel ships with a scalar reference and a bitwise-identity check. Numerical changes ship with a fixed-seed tolerance test against the pre-change implementation.
26. No defensive code for impossible states, no try/except as flow control, no catching `Exception` to be safe. Validate at system boundaries only.
27. No TODO/FIXME, no commented-out code, no dead code, no print-debugging (use `runtime.logs::logmod`). The working tree is the office; git history is the museum.
28. Comments are sparse, terse, imperative — a constraint the code cannot show, never narration of the change or its history. Names carry the explanation; rename instead of commenting.
29. Style: snake_case files and identifiers, lowercase filenames, no emdashes, 120-column lines, standard file header (Notes / Imports / Constants / Functions with dashed delimiters). Match the style of any file edited.
30. Measure before optimizing. A hunch is not a number. Check `failures.md` for measured dead ends before chasing a lever. Report deltas under 5% only with multiple seeds.

## tests

31. New functionality ships with a test that would have failed before the change, in the same diff, under `tests/<area>/test_*.py`. "Tested by the existing suite" is not a status — name the test.
32. Tests are deterministic (no live network, seeded RNGs, no wall-clock assertions), clean up after themselves (`tmp_path`), open with a one-line docstring stating the behavior verified, and hold one concept per test. Slow tests (>5s) carry `@pytest.mark.slow`.
33. Run the file's own tests before claiming a change works, then `./venv/bin/python -m ruff check .` — no new violations. "Should work" is not a status.

## docs

34. All documentation lives in the five files above. Never create a new doc file, doc folder, or scratch notes file. The sole exception is `lab/`, the raw experiment notebook, which is entirely local-only and never committed; `lab/README.md` defines its format.
35. A change that alters a contract, route, setting, or invariant updates the affected `documentation.md` section in the same diff. Docs state current behavior only — no history, no "previously"; git carries history.
36. Doc voice: a senior developer writing objective instructions. No first person, no hedging, no assistant register ("simply", "note that", "let's"), no narrative of how anything came to be.
37. Ledger entries are short: bold title, then one to three lines — claim, measured numbers with units and date, kill-line for failures. No essays.
38. Build notes are sections under `## build notes` in `documentation.md`, one per `versions.json::build`, three to ten lines, user-facing actions only. A breaking build also gets a `BUILD_NOTICES` entry (integer key) in `veritate_mri/runtime/settings.py`.

39. Every experiment, measurement, tuning attempt and abandoned line of work gets a `lab/` entry in the session it happens, in the format `lab/README.md` specifies, including the ones that produced no number and the ones that were reasoned about and dropped without running. The ledgers carry the three-line conclusion and name the entry; the entry carries the hypothesis, the pre-committed falsifier, the controls and the threats to validity. Ledger brevity (rule 37) does not apply inside `lab/`.

40. The `wren` fleet numbering is the user's to assign. Never mint a `wren1_N` or `wren2_N` name. Experimental forks are named `exp_<topic>_<MMDD>` and are DELETED the moment their measurement is recorded in `lab/` — a checkpoint whose number is already written down is dead weight. A box carries only the models it serves; `warm_models` in `data/mri_settings.json` is not the authority on that, the user is.
