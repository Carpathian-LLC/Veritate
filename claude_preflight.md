<!-- markdownlint-disable MD029 -->
# claude preflight

Contract for how Claude operates on this project. Wins over any conflicting doc. Rules hold until the user revokes one in writing. Numbers are continuous across sections so a rule can be cited as "rule 18".

## behavior

0. Short, concise, intuitive answers. No padding, preamble, or recap. Bullets when listing. Never assume the user knows the internals: define unfamiliar terms in one line, prefer plain words over jargon, say what a thing does before how.
1. Do only what the user explicitly asks. No suggestions, "want me to", follow-up offers, or next-step questions.
2. Ambiguous request: ask one short clarifying question and stop. Otherwise execute.
3. No new files, folders, agents, scripts, worktrees, or memory entries without explicit instruction.
4. Never push back on framing. Answer the question; don't interrogate it.
5. Maintain a TodoWrite list for any task with more than one sub-step; update status in real time.
6. Break work into sub-tasks before executing; mark exactly one in_progress at a time.
7. Responses are three to eight lines unless the user asked for long output.
8. State facts. No meta narration of intent, opening praise, or apology padding.
9. Engage as a peer: no sycophancy, patronizing, or harshness. Don't agree to make the user feel good; state the technical case honestly even against the user's framing.
9a. **Substantive pushback is required, not optional.** When a user plan has a credible failure mode (data loss, OOM, lost training hours, irreversible state, wasted compute, broken invariant), name the failure mode and its specific cost before executing, quantified when possible ("9 hours of training at risk"). Don't soften or hedge it: friction that costs seconds beats agreement that costs hours. This applies to the *plan*; rule 4 (don't interrogate the *question*) still holds.
9b. No filler praise of feedback or instructions ("great point", "good question", "I appreciate that"). One short acknowledgement at most, then proceed.
9c. **Read this file and [`developer_documentation/agents/coding_roe.md`](developer_documentation/agents/coding_roe.md) at the start of every session before any non-trivial work. Always, no exceptions.** Prior-context memory does not substitute. Before editing code, dispatching an agent, running a smoke, or changing config, confirm you've re-read both in this context. Cite rule numbers in your reasoning so it's visible you did.

## code

10. No emdashes anywhere in code or copy. Use a colon, period, or comma.
11. No hardcoded vars. Paths, dimensions, sizes, thresholds, scales, prompts, file extensions, magic numbers all come from a constants module or function arguments. No literals in function bodies.
11a. **Cross-model compatibility is gating.** Inference/decode code never branches on the *specific model variant* (`hasattr(model, "mtp")`, `isinstance(model, Veritate800M)`). Variants with different output projections expose a shared contract method on the model class (e.g. `model.project_byte0(residual)`); the consumer calls it blindly. Adding a variant must never require touching the decoder, agent loop, speculative draft, or engine wiring. Same for RoPE-vs-pos-emb and MTP-vs-no-MTP dispatch: the model knows what it is, the consumer doesn't.
12. snake_case for files, functions, variables. Lowercase filenames, including docs.
13. Every source file uses the standard structure below: mandatory sections separated by the dashed delimiter, in order Notes, Imports, Constants, Functions.
14. The `/docs/` dir is private and not in the repo. DO NOT REFERENCE IT.
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
20. No ad-hoc code anywhere. One module owns each concern; callers consume parsed results, they don't build paths, glob, parse, or duplicate logic. New capability extends the owning module.
20a. **Coding ROE is gating.** Any agent writing/editing code reads [`developer_documentation/agents/coding_roe.md`](developer_documentation/agents/coding_roe.md) (rules 100-128) before producing the diff: lean code, no bloat, measure before optimizing, no defensive code for impossible states. Cite by number when rejecting a change.

## training pipeline

21. Every checkpoint save goes through `veritate_mri/training/save.py::save()`; every per-step CSV write through `append_train_row()` in the same module. No trainer writes `.pt`, runs `dump_*`, or appends CSV rows directly.
22. `save()` runs the full dump suite on every call into `models/<name>/hooks/step_<N>/` with canonical filenames.
23. Field symmetry: every per-token frame the dashboard renders is emitted by both training-time and inference-time. Adding a field touches both in the same commit.
24. Every kernel produces bitwise-identical output to its scalar reference before shipping.

## docs

25. Documentation lives in three places: [`documentation/`](documentation/) (public, extension-author facing: [`api/rest_api.md`](documentation/api/rest_api.md) REST reference, [`extensions/`](documentation/extensions/) authoring), [`developer_documentation/`](developer_documentation/) (internal platform + component contracts, in repo), and `docs/` (papers, plans, results, notes; gitignored scratch). Lowercase, snake_case. Layout under [`developer_documentation/`](developer_documentation/):
    - `architecture/frontend/` — one file per dashboard tab, panel, or standalone module.
    - `architecture/backend/` — one file per Flask app, runtime, training, engine, or inference component.
    - `agents/` — agent rule files (coding_roe, claude_merge, agent_roe).
    - `addons/`, `corpus/`, `engine/`, `hooks/`, `kernels/`, `market/`, `multimind/`, `platform/`, `plugins/`, `trainers/`, `training/` — domain references.
25a. **Voice for shipped artifact docs.** Anything under `documentation/` or `developer_documentation/` (READMEs, contract/engine/format specs, anything alongside source) is written developer-to-developer about the artifact: no "the user", no "I/we/my", no narrative of how it came to be. State what it is, how to use it, what it guarantees. Past-tense decision narration and "the user wanted X" are forbidden. Ledger entries, ROE files, design diaries, and `docs/` scratch are exempt.
25b. **Per-component docs are mandatory.** Every component (frontend tab/panel, backend module, trainer plugin, kernel, engine subsystem) has one file under `developer_documentation/`, never a multi-component monolith. Sections: what it is, how it works (with file:line refs), dependencies, pitfalls. Keep each short.
25c. **Update a component's doc in the same change.** New component → new file; removed → delete the file. Docs reflect current state, not history (git carries history). If a component you touch has no doc, write one before finishing.
25d. **The two canonical doc trees are `documentation/` (public) and `developer_documentation/` (internal); no others.** New per-component docs go in the existing `developer_documentation/architecture/frontend/`, `architecture/backend/`, or appropriate domain folder, not a new top-level folder (`docs2/`, etc.).
25e. **Keep the external extension docs current.** When platform code the external API surface or extension contract depends on changes (a route an extension calls, the model-loading path, the experimental gate, the isolation boundary), update [`documentation/extensions/authoring.md`](documentation/extensions/authoring.md) and the affected endpoints in [`documentation/api/rest_api.md`](documentation/api/rest_api.md) in the same change (as 25c). The external "extension" is distinct from the internal trainer "plugin"; never relabel one as the other.
26. [`developer_documentation/hooks/contract.md`](developer_documentation/hooks/contract.md) is the API reference for dump artifacts. Update it whenever a hook changes.
27. Run output goes to `models/<model_name>/` (gitignored). Layout is fixed and documented at [`developer_documentation/training/storage.md`](developer_documentation/training/storage.md).
28. No CSVs in `documentation/` or `docs/`. CSVs live in the model dir.
29. Active work tracking lives in `docs/` files marked `*_tracking.md`. Check these first when resuming. They record what's done, works, is broken, and who did what. Maintain the canonical tracking doc (one entry per completed task); no parallel tracking.
30. Agents: read any active `*_tracking.md` at session start. One task at a time. Test before declaring done. Document blockers, don't work around them. Keep implementations lean (rule 19).

## execution

31. Reversible local actions run freely: file edits, file moves, py_compile, node --check, ls/grep.
32. Irreversible or shared-state actions confirm first: delete data, drop tables, force-push, send messages, change CI, kill processes the user didn't start.
33. No smoke tests, model loading, or training runs unless explicitly asked.

## portability

34. OS-specific primitives (aligned allocation, threading, time, file I/O) live behind a single shim in `veritate_engine/src/`. Per-arch kernels under `veritate_engine/kernels/<arch>/` never include OS headers (`windows.h`, `pthread.h`) or call OS APIs directly; they see one portable surface, and the shim picks Win32/POSIX/other at compile time. Each arch kernel stays a pure compute unit, and cross-platform support extends without per-arch OS plumbing.

## layout

34a. **`trainers/` is a synced checkout from an upstream canonical repo** (see `trainers/.sync_state.json`); upstream is source of truth, local lives behind it. Platform code (`veritate_core/`, `veritate_mri/`, root files) is fully editable locally — that's this repo.
34a-i. **Local trainer edits are allowed only when mirrored upstream:** either (a) staging a change the user will push to the canonical trainers repo, or (b) the user explicitly accepts the local edit and will sync it. Silent local-only edits are forbidden — `/trainers/git/sync` overwrites them and other machines never see them.
34a-ii. **Announce trainer edits before making them:** state the change is for upstream, name the file(s), quote the diff so the user can mirror it. Don't slip trainer changes into a multi-edit batch without acknowledgement.
34b. **No new trainers without explicit user permission.** The existing trainer set is the canonical product; don't add `trainers/<new_name>/` on your own initiative. New training capabilities go into the existing trainers via the upstream repo, or into platform helpers under `veritate_core/plugin/` that any trainer can call.
35. Trainers are self-contained by default: a trainer owns every file it uses. A helper only one trainer needs lives in that trainer's folder, not at any shared scope.
36. `trainers/common/` is the only escape hatch for shared helpers. A file moves there only when two or more trainers genuinely need it; until then it stays local to its single trainer.
37. `trainers/corpus/` holds training data only (`.bin` files). No code, build scripts, config, or JSON manifests describing what the bins are.
38. Build scripts live with their consumer: a one-trainer builder in that trainer, a many-trainer builder in `trainers/common/`. Output `.bin` files always land in `trainers/corpus/` (shared) or `trainers/<trainer>/corpus/` (bundled).
39. Trainers do not import from `veritate_mri/` or `veritate_engine/` directly. Their only platform surface is `veritate_core.plugin` (specified in [`developer_documentation/plugins/contract.md`](developer_documentation/plugins/contract.md)). `sys.path` injection into platform internals is forbidden.
40. Every code file lives in exactly one of: a trainer folder, `trainers/common/`, `veritate_core/`, `veritate_mri/`, `veritate_engine/`. No file fits two; no file fits none. If one fits none, the rule or the file is wrong — stop and ask.
40b. **Export format invariant (v9/v11).** The `.bin` engine format expects a canonical Veritate trunk: learned `pos_emb`, single `lm_head`, no MTP. RoPE-based models (Veritate800M, anything with `rope_*` buffers or no `pos_emb`), MTP-head models, and other variants are not exportable until a v12 engine format ships. The exporter raises `ValueError` early with the variant name, in both `export_checkpoint` and `export_checkpoint_ternary`.

## versioning and build notes

41. `versions.json` at the repo root is the version ledger: one global `build` counter plus per-component versions `engine`, `mri`, `format` (on-disk schema for models, settings, config files), `plugins`. Read it before any version question.
42. **Never bump a version without explicit user permission** — the `build` counter and every component string. If a change seems to warrant a bump, propose it in plain words and stop. The user decides.
43. When a major *format* change lands (model `.bin` layout, settings schema, trainer manifest, hook artifact contract), a build note MUST accompany it, explaining in user-facing terms what the user must do: which files to delete, rebuild, or rerun. No internals-only language.
44. Build notes live at `veritate_mri/wiki/build_notes/build_<N>.md` where `<N>` matches `versions.json::build`. One note per build. Format spec at `docs/build_notes_format.md` (gitignored scratch). Frontmatter required.
44a. **Build notes are super concise:** what changed, what the user must do, the version line. Three to ten body lines plus the version table. No deep dives or design rationale — internals belong in `documentation/`.

## merging

45. Branch merges are governed by [`developer_documentation/agents/claude_merge.md`](developer_documentation/agents/claude_merge.md) — read it before any merge. Its rule 1 is absolute: no merge without explicit user permission for that specific merge.

## tests

46. Every test function opens with a one-line docstring stating exactly what behavior it verifies. Concise, no "why"/"how"/padding. Just "GET /endpoint returns 200 + JSON."
47. One assertion per concept. Five unrelated asserts are five tests — split them.
48. Tests are deterministic: no live network calls (mock them), no real wall-clock timing assertions, seed every RNG.
49. Tests clean up after themselves: `tmp_path` or explicit fixture teardown for any file or directory created.
50. Slow tests (> 5s) carry `@pytest.mark.slow`. Default `pytest` runs them; the marker lets devs filter with `-m "not slow"` during fast local iteration.
51. Tests live under `tests/<area>/test_*.py`, mirroring the platform area tested (`engine/`, `export/`, `mri/`, `plugin_contract/`).
52. When functionality is added, a test that would have failed before the change lands in the same commit. No new feature ships untested.
