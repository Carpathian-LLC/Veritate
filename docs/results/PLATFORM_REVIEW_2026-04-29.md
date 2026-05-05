# Platform review — 2026-04-29

Reviewer: Claude (Opus 4.7, 1M context). Scope: full end-to-end validation of
the running MRI server (`http://127.0.0.1:8001`) against
`docs/GLASS_MODEL_ROE.md` rules, plus frontend / backend / filesystem
smoke tests. The user is frustrated with the cycle of "ship feature - break
something - discover later"; this review establishes a baseline.

Server state at start of review:
- exe `veritate_test.exe`, model `tinystories-80m-int8-qat2-curriculumB`
- `/meta` reports INT8-percol, qat2_curriculum, gelu, bin_version=5
- backfill jobs in flight (PIDs 44484, 56752 — not touched).

## Pass/fail matrix

| Test                                                          | Status | Notes                                                                                                |
| ------------------------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------- |
| 1. `GET /meta`                                                | PASS   | `c_model_dir`=curriculumB, precision=INT8-percol, training=qat2_curriculum                           |
| 2. `GET /c-models`                                            | PASS   | 7 entries; B/C have full description+stage; older entries have `description=""` (acceptable)         |
| 3. `GET /c-engines`                                           | PASS-W | Lists 2 engines (`veritate.exe`, `veritate_v1.0.0.exe`); active engine `veritate_test.exe` not in list |
| 4. `GET /runs`                                                | PASS   | 4 entries (B, C, mamba2 + txfm experiments). B+C present                                             |
| 5a. `GET /run/<B>/csv`                                        | PASS   | 200, 28929 B, 440 rows                                                                               |
| 5b. `GET /run/<C>/csv`                                        | PASS   | 200, 21619 B, 330 rows                                                                               |
| 6a. `GET /run/<B>/probes`                                     | PASS   | 10 steps (2k–20k @ 2k), each with probe + lens                                                       |
| 6b. `GET /run/<C>/probes`                                     | PASS   | 8 steps (2k–14k @ 2k, plus 15k final)                                                                |
| 7a. `GET /run/<B>/classroom`                                  | PASS   | 10 classroom + 9 grades + 9 concepts (18k grades/concepts pending — backfill in flight)              |
| 7b. `GET /run/<C>/classroom`                                  | PASS   | 8 classroom + 8 grades + 8 concepts                                                                  |
| 8a. `GET /run/<B>/classroom/classroom_step_2000.json`         | PASS   | 200                                                                                                  |
| 8b. `GET /run/<C>/classroom/classroom_step_4000.json`         | PASS   | 200                                                                                                  |
| 9a. `GET /run/<B>/config`                                     | PASS   | 200, full schema, `stage_description` set                                                            |
| 9b. `GET /run/<C>/config`                                     | PASS   | 200, full schema, `stage_description` set                                                            |
| 10. `GET /timelines`                                          | PASS   | 3 entries: curriculumB, curriculumC, default                                                         |
| 11a. `GET /timeline/<B>/timeline.json`                        | PASS   | 200, synthesized from probes                                                                         |
| 11b. `GET /timeline/<C>/timeline.json`                        | PASS   | 200                                                                                                  |
| 11c. `GET /timeline/default/timeline.json`                    | PASS   | 200, manifest source                                                                                 |
| 12. `GET /timeline/<B>/probe_step_2000.json`                  | PASS   | 200                                                                                                  |
| 13. `GET /train.csv`                                          | PASS   | 200                                                                                                  |
| 14. `POST /c-config` valid path                               | PASS   | swaps model on disk with body `{"model": "<abs path>"}`. Note body key is `model`, NOT `path`        |
| 15. C engine generation, max_new=10, full v7 frame            | PASS   | 10 token frames + done. Confidence/margin/entropy/lens_consistency/residual_stab all present         |
| 16. PyTorch generation, max_new=10                            | PASS   | streams meta + tokens                                                                                |
| 17. Path traversal blocked                                    | PASS-W | Blocked, but with HTTP 404 ("run not found") rather than 400 ("bad name"). Either way, no leakage    |
| FE-A. `mri/static/conversation.html` script parses cleanly    | PASS   | 1 inline script, 125k bytes; `node --check` clean                                                    |
| FE-B. Element IDs present                                     | PARTIAL | Found: runPicker, cModel, timelinePicker, learnSizeMeter, learnNeuronBio, cConfEvoL, learnLensDrift, cLossT, trainPlateau, trainLatest, cConfBar/cConfTrend (confidence panel). MISSING: `tabLearning`, `tabTraining` — code uses `data-tab="learning"` and `data-tab="training"` selectors instead |
| PY-A. `import mri.server.app`                                 | PASS   | imports cleanly                                                                                      |
| PY-B. `import mri.server.brain`                               | PASS   |                                                                                                      |
| PY-C. `import mri.server.c_engine`, FRAME_PAYLOAD_BYTES > 100k| PASS   | 125748 bytes (v7 size)                                                                               |
| PY-D. `import training.checkpoint_probe`, len(CONCEPTS) == 50 | PASS   |                                                                                                      |
| PY-E. `training.qat_v2_finetune --description` in help        | PASS   |                                                                                                      |
| FS-1. `data/models/<dir>/config.json` exists                  | PARTIAL | 7 of 11 model dirs have config.json. Missing: all 4 `mamba2-*` experimental dirs                     |
| FS-2. Both training_runs `train.csv` exist                    | PASS   | B + C                                                                                                |
| FS-3. `data/corpus/grade_eval_manifest.json` lists 7 grades   | PASS   | prek/k/elem/middle/hs/college/phd (phd is stub, flagged in manifest)                                 |
| FS-4. 7 grade_*_eval.bin                                      | PASS   |                                                                                                      |
| FS-5. B has 10 probe + 10 lens                                | PASS   |                                                                                                      |
| FS-6. B has 10 classroom (post-backfill)                      | PASS   | grades+concepts at 9/10 (18k pending)                                                                |
| FS-7. C has 8 probe + 8 lens                                  | PASS   |                                                                                                      |
| ROE-1. Naming `<corpus>-<size>-<precision>[-<variant>]`       | PASS   | All 11 dirs conform                                                                                  |
| ROE-2. Every config has `stage_description`                    | FAIL   | 4 dirs lack it: tinystories-80m-fp32, tinystories-80m-int8-qat2, tinystories-80m-int4-quarot, tinystories-5m-int8-draft, tinystories-200m-fp32. Legacy pre-rule dirs |
| ROE-3. No `docs/train_*.csv` other than `docs/train.csv`      | PASS   | git status mentions `docs/train_taskA_cont.csv` but file is not on disk                              |
| ROE-4. Five mandatory artifacts per run dir                    | PASS-W | B+C complete (B grades/concepts 9/10 = backfill in flight, acceptable). Older runs lack the new 3 — flagged but acceptable per task brief |
| ROE-9. No hardcoded model names in app.py / training/* / .h   | FAIL   | Hardcoded paths in `mri/server/diff.py:206-207` (defaults: `data/models/tinystories-80m/...`) and `training/export_quarot_int4.py:228-229`. `engine/src/veritate.h` is clean. `mri/server/app.py` is clean of model-name literals |

PASS = 30. PASS-W (pass with warning) = 3. PARTIAL = 2. FAIL = 2. Total = 37.

## Issues found

### BLOCK — none

No blocking issues. Every primary workflow (model swap, C generation, PyTorch
generation, run/timeline browsing, classroom dashboards) returns valid data.

### WARN-1 — Hardcoded model names in `mri/server/diff.py` (ROE Rule 9)

Location: `mri/server/diff.py:206-207`.

```
ap.add_argument("--bin",       default="data/models/tinystories-80m/veritate.bin")
ap.add_argument("--checkpoint",default="data/models/tinystories-80m/checkpoints/step_45000.pt")
```

`data/models/tinystories-80m/` does not exist (was renamed to `tinystories-80m-fp32`).
Anyone running `diff.py` with no flags will get FileNotFoundError. Rule 9
forbids string-literal model-name defaults; should derive freshest matching
dir or require explicit `--bin`/`--checkpoint`.

Suggested fix: drop the defaults, mark `required=True`, or implement a
`_freshest_model_dir()` discovery helper. Not applied — touches a tool that
imports into the running server's process namespace.

### WARN-2 — Hardcoded model names in `training/export_quarot_int4.py` (ROE Rule 9)

Location: `training/export_quarot_int4.py:228-229`.

```
ap.add_argument("--checkpoint", default="data/models/tinystories-80m/checkpoints/step_45000.pt")
ap.add_argument("--out",        default="data/models/tinystories-80m-quarot-int4/veritate-int4.bin")
```

Out-path also references a non-existent `tinystories-80m-quarot-int4` dir
(actual dir is `tinystories-80m-int4-quarot`). Both are stale post-rename.

Suggested fix: same as WARN-1. Not applied per task brief ("training scripts:
report only").

### WARN-3 — `mamba2-*` experimental dirs lack `config.json` (ROE Rule 1/2)

Locations:
- `data/models/mamba2-7m-fp32-baseline/`
- `data/models/mamba2-7m-fp32-test/`
- `data/models/mamba2-20m-fp32/`
- `data/models/mamba2-20m-fp32-txfm-baseline/`

These dirs hold `*.pt` checkpoints and a `summary.json`/`sample_text.txt`,
but no `config.json`. ROE Rule 2 says every model dir MUST have one with
`stage_description`. They surface in `/c-models` only because the scanner
also looks for `summary.json`-only dirs (those are filtered out as
`status: training`). Recommend: either move these out to
`experiments/26_mamba2_prototype/models/` (where the train CSVs already
live), or create stub config.json files documenting they are mamba-2
prototype runs.

Not applied — moving / writing files for non-active experiments is out of
scope, and these existed before the ROE was finalized.

### WARN-4 — Legacy configs missing `stage_description` (ROE Rule 2)

5 dirs: `tinystories-200m-fp32`, `tinystories-80m-fp32`, `tinystories-80m-int8-qat2`,
`tinystories-80m-int4-quarot`, `tinystories-5m-int8-draft`. All have valid
`config.json` but no `stage_description` field. Pre-date Rule 2's gate.
Acceptable as legacy per the task brief, but the trainer description gate
(Rule 6) protects forward — these will never get worse, just won't show
descriptions in the model picker tooltip.

Suggested fix: backfill `stage_description: "legacy <stage> baseline, no
description recorded"` to make the picker tooltip non-empty. Low risk.
Could be applied — these are static files, no live process owns them.
**Holding off** because the task brief excluded "training scripts, or
running services" and the running server reads these on every `/c-models`
hit.

### NIT-1 — `_safe_name` returns 404 instead of 400 on traversal

Location: `mri/server/app.py:285,298,321,353,367,454`.

When you URL-encode `..%2F`, Werkzeug normalizes the path and the route
either fails to match (404) or `_safe_name` returns False and emits 400.
In practice the test traversal URLs all returned 404 ("run not found"),
not 400. Functionally safe — no path traversal succeeds — but the documented
contract was "400 on bad name". Not a real bug.

### NIT-2 — Test-list ID mismatch: `tabLearning` / `tabTraining`

Location: `mri/static/conversation.html`.

The review test list expected `id="tabLearning"` and `id="tabTraining"`.
Actual HTML uses `<div class="tab" data-tab="learning">` and
`<div class="tab" data-tab="training">` — class+data-attr selectors, not
IDs. Functionally fine; the JS uses `document.querySelector('[data-tab="X"]')`
which works. Just an outdated test spec.

### NIT-3 — `/c-engines` does not list active `veritate_test.exe`

Location: `mri/server/app.py` (engine scanner).

Active engine name (per `/meta`) is `veritate_test.exe`, but the scanner
only returns `veritate.exe` and `veritate_v1.0.0.exe`. Nothing breaks (the
running engine works fine), but a future user clicking "current" in the
engine picker would actually swap to a different binary than the one in
flight. Suggest including the currently-active engine in the listing
even if it doesn't match the canonical glob.

## Fixes applied this round

None. All four code-touching fixes (WARN-1, WARN-2, WARN-3 stubs, WARN-4
backfill, NIT-3 scanner) involve files that are either training scripts,
imported into the running server, or modify shared state. Per the task
brief: "Anything that touches engine, training scripts, or running services:
report only — don't apply." All other findings are spec drift in the
review checklist, not bugs.

The model selection was temporarily swapped from curriculumB to curriculumC
(via `POST /c-config`) to verify the swap path, then immediately swapped
back. End state matches the start state.

## Outstanding

Human attention needed on:

1. **WARN-1 / WARN-2** (high priority): hardcoded post-rename model paths
   in `mri/server/diff.py` and `training/export_quarot_int4.py`. Anyone who
   runs these tools with default args today will hit FileNotFoundError.
2. **WARN-3** (med): mamba2 experimental dirs need either `config.json` or
   relocation to `experiments/`. They currently appear in some scans but
   lack the metadata to drive any panel.
3. **WARN-4** (low): backfill `stage_description` on the 5 legacy configs
   so model-picker tooltips render content for them.
4. **NIT-3** (low): `/c-engines` scanner should include the live exe even
   when it doesn't match the canonical name pattern, or `veritate_test.exe`
   should be renamed to `veritate.exe` once it stabilises.

The actively-running grades/concepts backfill (B step_18000 in flight) will
close PASS-W on test 7a once it lands; no human action needed there.

End-to-end status: all primary workflows are green. The "ship feature →
break something → discover later" cycle this review was meant to break
shows two real artifacts (WARN-1 / WARN-2 — both stale defaults from the
2026-04-26 rename) that would silently break the next user. Everything else
is cosmetic.
