# Coral Lab tab

## What it is

Three-column comparison view for the Coral Merge experiment: two constituent runs and one comparison (baseline or blend). Standalone module, deletable as a unit.

## How it works

A standalone JS/CSS module that mounts into a host element in `index.html`. No edits to `index.js` are required beyond adding `"coral"` to the `valid` array.

Files:

- [veritate_mri/web/coral_lab.css](../../../veritate_mri/web/coral_lab.css) — styles (all `.coral-*` classes).
- [veritate_mri/web/coral_lab.js](../../../veritate_mri/web/coral_lab.js) — IIFE module: builds the scaffold inside `#coralBody`, binds picker events, polls `/runs` and `/run/<name>/csv` every 5s when the tab is active.
- Four `DELETABLE-CORAL`-marked edits in [index.html](../../../veritate_mri/web/index.html): `<link>` + `<script>` in `<head>`, the tab in `<div class="tabs">`, the tab-body.
- One `DELETABLE-CORAL`-marked entry in the `valid` array at [index.js:2098](../../../veritate_mri/web/index.js#L2098).

Picker selections persist to localStorage as `coral.pick.A`, `coral.pick.B`, `coral.pick.CMP`. Auto-selects on first load by run-name prefix (`coral_a_`, `coral_b_`, `coral_baseline_`/`coral_blend_`).

## Dependencies

- `/runs` and `/run/<name>/csv` from [runs_routes.py](../../../veritate_mri/routes/runs_routes.py). No new routes added for Coral.
- Backend training CSV contract at [save.py:38](../../../veritate_mri/training/save.py#L38).
- The Coral Merge tooling at [tools/coral/](../../../tools/coral/) produces the runs this dashboard reads.
- The algorithm spec lives at `~/Documents/GitHub/Agent-Documents/Veritate/coral_merge_spec.md`.

## Removal

The experiment is fully deletable in five steps:

1. `rm -rf tools/coral/`
2. `rm veritate_mri/web/coral_lab.css veritate_mri/web/coral_lab.js`
3. Remove the four `DELETABLE-CORAL` blocks in `index.html` (link, script, tab, tab-body).
4. Remove the `"coral"` entry from the `valid` array in [index.js:2098](../../../veritate_mri/web/index.js#L2098).
5. Optional: `rm -rf models/coral_*`. The 50m preset addition in [trainers/multimind_m3/plugin.py](../../../trainers/multimind_m3/plugin.py) is harmless to leave in place.

## Pitfalls

- The Coral Lab tab won't activate if `"coral"` is missing from the `valid` array. The tab appears in the bar but clicking it silently routes to `generation`.
- The `started_at` shown in the panel comes from the CSV mtime when training is launched outside `plugin_runner` (see [../backend/heartbeat.md](../backend/heartbeat.md) — same fallback applies here).
