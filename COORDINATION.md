# coordination

## agent-1 (review + fixes)

Platform audit running. Fixing as found.

**My lane:** routes/, inference/, training/ (not teacher), runtime/, readers/, tools/, veritate_core/ (not `plugin/__init__.py`), documentation/ (not plugins/contract.md), tests/ (not tests/teacher/), CLAUDE.md.

**Your lane (untouched):** teacher/, data/seeds/, web/, settings.py teacher_* keys, `veritate_core/plugin/__init__.py`, documentation/plugins/contract.md, app.py register line.

## agent-2 (teacher model) — DONE

Shipped: `veritate_mri/teacher/` (provider-agnostic client + synth pipeline), `routes/teacher_routes.py`, seeded `data/seeds/` (7 jsonl + catalog, 250 entries), Settings panel + Training tab gating in `web/index.{html,js,css}`, `get_teacher_client()` on plugin surface, settings keys `teacher_*` + validation, `.gitignore` += `/synth_jobs/`. 49 tests in `tests/teacher/`. User declined version bump.

UX audit items from agent-1: NOT applied here. They touch pre-existing emdashes/CLI strings/jargon outside the teacher feature, so they belong in agent-1's review lane.
