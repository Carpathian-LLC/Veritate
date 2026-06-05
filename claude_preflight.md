<!-- markdownlint-disable MD029 -->
# claude preflight

Contract for how Claude operates on this project. If any other doc conflicts with this one, this wins. Rules hold until the user explicitly revokes one in writing. Rule numbers are continuous across sections so individual rules can be cited as "rule 18", etc.

## behavior

0. Short, concise answers. No padding, no preamble, no recap. Bullet-form when listing. Never assume the user knows the internals: define unfamiliar terms in one line, prefer plain words over jargon, lead with what a thing does before how. Intuitive and short.
1. Do only what the user explicitly asks. No suggestions, no "want me to", no follow-up offers, no questions about next steps.
2. If a request is ambiguous, ask one short clarifying question and stop. Otherwise execute.
3. No new files, folders, agents, scripts, worktrees, or memory entries without explicit instruction.
4. Never push back on framing. Answer the question; do not interrogate it.
5. Always maintain a todo list (TodoWrite) for any task with more than one sub-step. Update status in real time.
6. Always break work into sub-tasks before executing. Mark exactly one in_progress at a time.
7. Keep responses short. Three to eight lines unless the user asked for a long output.
8. State facts. No meta description, no narration of intent, no opening praise, no apology padding.
9. Engage as a peer. No sycophancy, no patronizing, no harshness. Do not agree with a plan to make the user feel good. State the technical case honestly even when it contradicts the user's framing.
9a. **Substantive pushback is required, not optional.** When a user-proposed plan has a credible failure mode (data loss, OOM, lost training hours, irreversible state, wasted compute, broken invariant), name the failure mode and its specific cost before executing. Quantify when possible (e.g. "9 hours of training at risk"). Do not soften, hedge, or rephrase as a gentle suggestion. Agreement that costs the user hours is worse than friction that costs them seconds. Rule 4 still holds: do not interrogate the *question*; rule 9a applies to the *plan*.
9b. No filler praise of feedback or instructions ("great point", "good question", "I appreciate that"). Acknowledge in one short line if at all, then proceed. The acknowledgement itself is not the work.
9c. **Read this file (and `documentation/agents/coding_roe.md`) at the start of every session before non-trivial work. Always. No exceptions.** Memory from a prior context does not substitute. If you find yourself about to edit code, dispatch an agent, run a smoke, or change a config without having re-read both files in the current context, STOP and read first. Cite preflight rule numbers in your reasoning so it's visible you did.

## code

10. Never use emdashes anywhere in code or copy. Use a colon, a period, or a comma.
11. Never hardcode vars. Paths, dimensions, sizes, thresholds, scales, prompts, file extensions, magic numbers all come from a constants module or function arguments. No literals in function bodies.
11a. **Cross-model compatibility is gating.** Inference / decode code never branches on the *specific model variant* (`hasattr(model, "mtp")`, `isinstance(model, Veritate800M)`, etc.). If two model variants have different output projections, the *model class* exposes a shared contract method (e.g. `model.project_byte0(residual)`); inference calls the contract blindly. Adding a new model variant must never require touching the decoder, the agent loop, the speculative draft, or the engine wiring. Same rule applies to RoPE-vs-pos-emb dispatch and to MTP-vs-no-MTP detection: the model knows what it is; the consumer does not.
12. snake_case for files, functions, variables. Lowercase filenames including docs.
13. Every source file begins with the standard structure below. Sections are mandatory and separated by the dashed delimiter line. Order: Notes, Imports, Constants, Functions.
14. The `/docs/` dir is private and not available in the repo. DO NOT REFERENCE IT!
```
# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - this is what the file contains and what it does. Special considerations.
# file/path
# ------------------------------------------------------------------------------------
# Imports:


# ------------------------------------------------------------------------------------
# Constants

# ------------------------------------------------------------------------------------
# Functions
```
15. Inline comments are sparse, terse, imperative. No articles, no rationale, no "why". Rationale lives in `documentation/`.
16. No TODO, FIXME, or commented-out code. Dead code is deleted.
17. No PR refs, ticket numbers, or fix tags in source.
18. Match the existing style of any file edited. Default to the leanest version that compiles.
19. Anti-overengineering is gating. If a feature can be removed without breaking the goal, remove it. Two layers of abstraction is one too many.
20. No ad-hoc code anywhere. One module owns each concern. Callers consume parsed results; they do not build paths, glob, parse, or duplicate logic. New capability extends the owning module.
20a. **Coding ROE is gating.** Any agent writing or editing code in this repo must read `documentation/agents/coding_roe.md` before producing the diff. Rules 100-128 there are enforced like rules in this file. Lean code, no bloat, measure before optimizing, no defensive code for impossible states. Cite by number when rejecting a change (e.g., "rule 105: one-caller abstraction, inlining").

## training pipeline

21. Every checkpoint save goes through `veritate_mri/training/save.py::save()`. Every per-step CSV write goes through `append_train_row()` in the same module. No trainer writes `.pt`, runs `dump_*`, or appends CSV rows directly.
22. `save()` runs the full dump suite on every call into `models/<name>/hooks/step_<N>/` with canonical filenames.
23. Field symmetry: every per-token frame the dashboard renders is emitted by both training-time and inference-time. Adding a field touches both in the same commit.
24. Every kernel produces bitwise-identical output to its scalar reference before shipping.

## docs

25. All documentation lives under `documentation/` (current platform contracts, in repo) and `docs/` (papers, plans, results, notes; gitignored, repo-local scratch). Lowercase filenames, snake_case. Layout under `documentation/`:
    - `architecture/frontend/` — one file per dashboard tab, panel, or standalone module.
    - `architecture/backend/` — one file per Flask app, runtime, training, engine, or inference component.
    - `agents/` — agent-facing rule files (coding_roe, claude_merge, agent_roe).
    - `addons/`, `corpus/`, `engine/`, `hooks/`, `kernels/`, `multimind/`, `platform/`, `plugins/`, `trainers/`, `training/` — domain references.
25a. **Voice for shipped artifact docs.** Anything under `documentation/` (READMEs, contract docs, engine specs, format specs, anything alongside source) is written developer-to-developer about the artifact. No "the user", no "I/we/my", no narrative of how the artifact came to be. State what it is, how to use it, what it guarantees. Past-tense narration of decisions and references to "the user wanted X" read like an AI talking to a reader and are forbidden. Ledger entries, ROE files, design diaries, and `docs/` scratch notes are exempt; the multi-agent voice is correct in those.
25b. **Per-component docs are mandatory.** Every component (frontend tab, frontend panel, backend module, trainer plugin, kernel, engine subsystem) has a doc under `documentation/`. The doc is one file per item — never a multi-component monolith. Standard sections: what it is, how it works (with file:line references), dependencies, pitfalls. Keep each file short and digestible.
25c. **When you change a component, update its doc in the same change.** If the component is new, add a new file. If the component is removed, delete the file. The doc reflects current state, not history; git carries the history. If a component you touch has no doc yet, write one before finishing the change.
25d. **No new top-level documentation hierarchies.** New per-component docs go inside the existing `documentation/architecture/frontend/`, `documentation/architecture/backend/`, or the appropriate domain folder. Do not add a new top-level docs folder (e.g., `dev_documentation/`, `docs2/`) — fold the content into `documentation/` instead.
26. `documentation/hooks/contract.md` is the API reference for the dump artifacts. Updated whenever a hook changes.
27. Run output goes to `models/<model_name>/` (gitignored). Layout is fixed and documented at `documentation/training/storage.md`.
28. No CSVs in `documentation/` or `docs/`. CSVs live in the model dir.
29. Active work tracking lives in `docs/` files marked `*_tracking.md`. Check these first when resuming work. They record what is done, what works, what is broken, and who did what. Do not create parallel tracking — maintain the canonical tracking doc, one entry per completed task.
30. Agents: Read any active `*_tracking.md` file at the start of each session. One task at a time. Test before declaring done. Document blockers, don't work around them. Keep implementations simple: two layers of abstraction is one too many. If a feature can be removed without breaking the goal, remove it.

## execution

31. Reversible local actions: run freely (file edits, file moves, py_compile, node --check, ls/grep).
32. Irreversible or shared-state actions: confirm first (delete data, drop tables, force-push, send messages, change CI, kill processes the user did not start).
33. No smoke tests, model loading, or training runs unless explicitly asked.

## portability

34. OS-specific primitives (aligned allocation, threading, time, file I/O) live behind a single shim in `veritate_engine/src/`. Per-arch kernels under `veritate_engine/kernels/<arch>/` never include OS headers (`windows.h`, `pthread.h`) directly and never call OS APIs directly. Kernels see one portable surface; the shim picks Win32 vs POSIX vs other at compile time. This keeps each arch kernel a pure compute unit and lets cross-platform support extend without duplicating OS plumbing per arch.

## layout

34a. **`trainers/` is a synced checkout from an upstream canonical repo.** See `trainers/.sync_state.json`. The upstream is the source of truth; local lives behind it. Platform code (`veritate_core/`, `veritate_mri/`, root files) is fully editable locally — that's this repo.
34a-i. **Local trainer edits ARE allowed, but only when the edit will be mirrored upstream.** Either (a) you're staging a change the user will push to the canonical trainers repo, or (b) the user explicitly accepts the local edit and will sync it. Silent local-only edits are forbidden — `/trainers/git/sync` will overwrite them and the team's other machines never see them.
34a-ii. **Announce trainer edits before making them.** State that the change is intended for upstream, name the file(s), and quote the diff in your reply so the user can mirror it. Don't slip trainer changes into a multi-edit batch without explicit acknowledgement.
34b. **No new trainers without explicit user permission.** The existing trainer set is the canonical product. Do NOT add `trainers/<new_name>/` directories on your own initiative. New training capabilities go into the EXISTING trainers via the upstream trainers repo, or into platform helpers under `veritate_core/plugin/` that any trainer can call.
35. Trainers are self-contained by default. A trainer owns every file it uses. If only one trainer needs a helper, the helper lives inside that trainer's folder, not at any shared scope.
36. `trainers/common/` is the only escape hatch for shared helpers. A file is moved there only when two or more trainers genuinely need it. Until that moment, it stays local to its single trainer.
37. `trainers/corpus/` holds training data only: `.bin` files. Never code, never build scripts, never config, never JSON manifests describing what the bins are.
38. Build scripts live with their consumer. A builder used by one trainer lives in that trainer. A builder used by many lives in `trainers/common/`. Output `.bin` files always land in `trainers/corpus/` (shared) or `trainers/<trainer>/corpus/` (bundled).
39. Trainers do not import from `veritate_mri/` or `veritate_engine/` directly. The only platform surface they may reach is `veritate_core.plugin`, specified in `documentation/plugins/contract.md`. `sys.path` injection into platform internals is forbidden.
40. Every code file lives in exactly one of: a trainer folder, `trainers/common/`, `veritate_core/`, `veritate_mri/`, or `veritate_engine/`. No file fits two; no file fits none. If a file does not fit any, the rule is wrong or the file is wrong. Stop and ask.

40a. **SUCCESSES / FAILURES / IDEAS are research logs, not changelogs.** Only validated research findings and PoCs go in `SUCCESSES.md` / `FAILURES.md` (with numbers + falsifier). Bug fixes, refactors, plumbing, infra wiring, and routine engineering do NOT go there — they live in git history. If a fix implies a durable invariant (e.g., "exporter only handles canonical-Veritate trunks"), the invariant gets added to this preflight or `NOTES.md`; the fix itself is not an S/F entry. Default to NOT writing a doc entry; require yourself to justify why it isn't just a commit.

40b. **Export format invariant (v9/v11).** The `.bin` engine format expects a canonical Veritate trunk: learned `pos_emb`, single `lm_head`, no MTP. RoPE-based models (Veritate800M, anything with `rope_*` buffers or no `pos_emb`), MTP-head models, and other variants are NOT exportable today and will not be until a v12 engine format ships. The exporter raises `ValueError` early with a clear variant name. Same guard exists in both `export_checkpoint` and `export_checkpoint_ternary`.

## versioning and build notes

41. `versions.json` at the repo root is the version ledger. It tracks one global `build` counter plus per-component versions: `engine`, `mri`, `format` (the on-disk schema for models, settings, and config files), and `plugins`. Read it before answering any version question.
42. **Never bump a version without explicit user permission.** This includes the `build` counter and every component string. If a change feels like it should warrant a bump, propose it in plain words and stop. The user decides.
43. When a major change to a *format* lands (model `.bin` layout, settings schema, trainer manifest, hook artifact contract), a build note **MUST** accompany the change. The note explains, in user-facing terms, what the user has to do to keep working: which files to delete, which to rebuild, which commands to rerun. No internals-only language.
44. Build notes live at `veritate_mri/wiki/build_notes/build_<N>.md` where `<N>` matches `versions.json::build`. One note per build. Format spec at `docs/build_notes_format.md` (gitignored scratch). Frontmatter is required.
44a. **Build notes are SUPER concise.** Very high-level overview only. What changed, what the user has to do, the version line. Three to ten lines of body, plus the version table. No deep dives, no design rationale, no per-component novella. Internals belong in `documentation/`, not in build notes.

## merging

45. Branch merges are governed by [documentation/agents/claude_merge.md](documentation/agents/claude_merge.md). Read it before any merge. Its rule 1 is absolute: no merge without explicit user permission for that specific merge.

## tests

46. Every test function starts with a one-line docstring that states exactly what behavior it verifies. Stupid concise. No "why", no "how", no "this test ensures…" padding. Just: "GET /endpoint returns 200 + JSON."
47. One assertion per concept. A test with five unrelated asserts is five tests. Split it.
48. Tests are deterministic. No live network calls (mock them). No real wall-clock timing assertions. Seed every RNG.
49. Tests clean up after themselves. Use `tmp_path` or explicit fixture teardown for any file or directory created during the test.
50. Slow tests (> 5 seconds) carry `@pytest.mark.slow`. Default `pytest` runs them; the marker exists so devs can filter with `-m "not slow"` during fast local iteration.
51. Tests live under `tests/<area>/test_*.py`. The folder structure mirrors the platform area being tested (`engine/`, `export/`, `mri/`, `plugin_contract/`, etc.).
52. When functionality is added, a test that would have failed before the change lands in the same commit. No new feature ships untested.
