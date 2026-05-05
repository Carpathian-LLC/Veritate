# MRI dashboard audit — 2026-04-29

End-to-end audit of every fetch + DOM hookup in `mri/static/conversation.html`
against `mri/server/app.py`, with the running server (`veritate_test.exe`,
`tinystories-80m-int8-qat2-curriculumC`) responding on `http://127.0.0.1:8001`.

## Endpoint reachability (server)

| endpoint                                         | status | shape OK |
| ------------------------------------------------ | ------ | -------- |
| `GET /meta`                                      | 200    | yes      |
| `GET /timelines`                                 | 200    | 3 entries: curriculumC, curriculumB, default — confirmed |
| `GET /timeline/<name>/timeline.json`             | 200    | probe-source synthesized; checkpoints carry `step,file,n_frames,output_text` |
| `GET /timeline/<name>/step_<N>.json`             | 200    | `{meta, frames}`, 80 frames, full Rule-5 v7 field set |
| `GET /timeline/<name>/probe_step_<N>.json`       | 200    | `{step, prompt, top_k, layers}` |
| `GET /timeline/<name>/lens_step_<N>.npz`         | 200    | npz, ~7 KB, `lens_logits`+`residual_norms` |
| `GET /runs`                                      | 200    | 4 runs (curriculumB+C + 2 mamba2 experiments) |
| `GET /run/<name>/csv`                            | 200    | matches CSV at `data/training_runs/<name>/train.csv` |
| `GET /run/<name>/probes`                         | 200    | curriculumB:10 steps, curriculumC:8 steps |
| `GET /run/<name>/config`                         | 200    | curriculum runs only; mamba2 experiments → 404 (graceful) |
| `GET /generate?backend=c&...` (SSE)              | 200    | full v7 frame fields per token |
| `GET /generate?backend=pytorch&...` (SSE)        | 200    | mirrors v7 fields |

## Pass/fail matrix

Legend: PASS = renders correctly with current data; DEG = degrades gracefully
(empty state with a sensible "no data" caption); FIX = was broken, fixed in this
audit; OPEN = still broken after fix.

### Generation tab (live SSE drives `render(frame)`)

| panel                 | DOM id              | data field                              | status |
| --------------------- | ------------------- | --------------------------------------- | ------ |
| model meta            | `modelMeta`         | `/meta` + SSE meta                      | PASS   |
| telemetry             | `cTel`              | `frames[*].entropy_bits/surprise_bits/fwd_ms` | PASS |
| FFN brain             | `cFfn`              | `frame.ffn_full`                        | PASS   |
| attention map         | `cAttn`             | `frame.attn`, `frame.T`                 | PASS   |
| logit lens            | `lens`              | `frame.lens`, `frame.byte`              | PASS   |
| candidates            | `cand`              | `frame.cand`                            | PASS   |
| residual depth        | `res`               | `frame.res`                             | PASS   |
| per-layer contrib     | `contrib`           | `frame.contrib`                         | PASS   |
| decision trace picked | `dlaPickedTable`    | `frame.dla_picked`                      | PASS   |
| decision trace argmax | `dlaArgmaxTable`    | `frame.dla_argmax`                      | PASS   |
| confidence (big bar)  | `cConfBar`          | `frame.confidence`                      | PASS   |
| confidence components | `confComponents`    | `frame.margin/entropy/lens_consistency/residual_stab` | PASS |
| confidence trend      | `cConfTrend`        | per-frame `confidence`                  | PASS   |
| top neurons           | `cTop`              | `frame.ffn_top`                         | PASS   |
| decisiveness          | `cDecisive`         | `frame.decisiveness`                    | PASS   |
| info-flow chart       | `cFlow`             | `frame.info_flow`                       | PASS   |
| memory fingerprint    | `memory`            | `frame.memory`                          | DEG (always `[]` from C path; brain.py path populates) |
| letter ms             | `cLetMs`            | derived from frames[]                   | PASS   |
| letter stats          | `letterStats`       | derived                                 | PASS   |
| ASCII reference       | `asciiRef`          | static                                  | PASS   |

### Learning tab (timeline picker → `renderLearning()` + 4 classroom panels)

| panel                  | DOM id              | data dependency                                          | status                    |
| ---------------------- | ------------------- | -------------------------------------------------------- | ------------------------- |
| status                 | `learningStatus`    | `timeline.json` `prompt`, `checkpoints.length`, `max_new` | PASS                      |
| output evolution grid  | `ckptOutputs`       | `meta.checkpoints[].output_text`, fallback `precision="FP32"` | PASS                      |
| quant KL trajectory    | `cQuantKlL`         | `checkpoint.quant_kl_bits`                               | DEG (probe-source synth has none → "not present" caption) |
| ckpt scrubber          | `ckptSlider`        | `meta.checkpoints.length`                                | PASS                      |
| ckpt label             | `ckptLabel`         | `c.precision/c.step`                                     | PASS                      |
| response text          | `responseL`         | `frame.byte` accumulated                                 | PASS                      |
| token scrubber         | `scrubL`            | frames length                                            | PASS                      |
| telemetry              | `cTelL`             | `frames[*].entropy_bits/...`                             | PASS                      |
| FFN brain              | `cFfnL`             | `frame.ffn_full`                                         | PASS                      |
| saturation             | `cSatL`             | `frame.saturation`                                       | PASS (training dump emits it) |
| attention map          | `cAttnL`            | `frame.attn`                                             | PASS                      |
| lens                   | `lensL`             | `frame.lens`                                             | PASS                      |
| candidates             | `candL`             | `frame.cand`                                             | PASS                      |
| residual depth         | `resL`              | `frame.res`                                              | PASS                      |
| contrib                | `contribL`          | `frame.contrib`                                          | PASS                      |
| decision trace L       | `dlaPickedTableL`   | `frame.dla_picked`                                       | PASS                      |
| decision trace L arg.  | `dlaArgmaxTableL`   | `frame.dla_argmax`                                       | PASS                      |
| top neurons            | `cTopL`             | `frame.ffn_top`                                          | PASS                      |
| decisiveness           | `cDecisiveL`        | `frame.decisiveness`                                     | PASS                      |
| memory fingerprint     | `memoryL`           | `frame.memory`                                           | DEG (training dump empties array) |
| classroom: size meter  | `learnSizeMeter`    | `/run/<tl>/config` `shape` field                         | FIX (was: never loaded on first activation; now loads from `ensureLearningLoaded`) |
| classroom: neuron bio  | `learnNeuronBio`    | `/run/<tl>/probes` + each `probe_step_*.json`            | FIX (same)                |
| classroom: conf evo    | `cConfEvoL`         | `/run/<tl>/probes` + each `lens_step_*.npz`              | FIX (same)                |
| classroom: lens drift  | `learnLensDrift`    | `/run/<tl>/probes` + `lens_step_*.npz`                   | FIX (same)                |

### Live Training tab (`/runs` + `/run/<n>/csv` + 4 classroom panels)

| panel                 | DOM id          | data dependency                              | status |
| --------------------- | --------------- | -------------------------------------------- | ------ |
| status                | `trainStatus`   | run name + last load time                    | PASS   |
| latest snapshot       | `trainLatest`   | last CSV row                                 | PASS   |
| plateau verdict       | `trainPlateau`  | val series last 4 evals                      | PASS   |
| loss chart            | `cLossT`        | train + val from CSV                         | PASS   |
| lr chart              | `cLrT`          | `lr` column                                  | PASS   |
| tok/s chart           | `cTpsT`         | `tok_per_s` column (mamba2 csvs lack it → blank chart, expected) | PASS / DEG |
| grad-norm chart       | `cGnT`          | `grad_norm` column                           | PASS   |
| recent rows           | `trainRecent`   | last 30 CSV rows                             | PASS   |
| classroom: size meter | `trainSizeMeter`| `/run/<run>/config`                          | PASS   |
| classroom: neuron bio | `trainNeuronBio`| `/run/<run>/probes` + probe jsons            | PASS   |
| classroom: conf evo   | `cConfEvoT`     | lens npzs                                    | PASS   |
| classroom: lens drift | `trainLensDrift`| lens npzs                                    | PASS   |

## Root causes of the failing panels (Learning tab, before fix)

1. The `activateTab("learning")` handler guarded the classroom load on
   `learningTimelineName` being non-null, but `ensureLearningLoaded` is async
   and sets `learningTimelineName` AFTER `activateTab`'s synchronous guard
   already ran. First-paint result: the four classroom panels stayed in their
   "loading…" or "no timeline selected" / "no probe data" placeholder state
   forever, even though `/timelines` returned valid entries. The fix triggers
   `loadClassroomForLearning(learningTimelineName)` from the bottom of
   `ensureLearningLoaded`, after the manifest has been parsed and the variable
   is populated.

2. The `timelineRefresh` button no-op'd when the picked timeline name was
   unchanged. Now it forces both `learningState` and `classroomStateL` to clear
   so freshly written probe/lens/step files surface without a tab toggle.

## Files changed

- `mri/static/conversation.html`
  - L2702-L2710 — added classroom-mirror kick-off at the bottom of
    `ensureLearningLoaded` once `learningTimelineName` is bound.
  - L3199-L3210 — `timelineRefresh` button forces a state reset on same-name
    refresh.

## Outstanding (not breaking anything; informational)

- Probe-source synthesized timelines do not carry `precision` /
  `effective_step` / `quant_kl_bits` per checkpoint, so the
  `outputEvolution` grid lumps everything under "FP32" and the `cQuantKlL`
  chart shows its empty-state caption. Reasonable suggested fix:
  `_scan_timelines` could enrich each synthesized checkpoint with `precision`
  drawn from the run's `config.json` `training` field (e.g.
  `qat2_curriculum` → "QAT2"). 5 lines in `app.py::_scan_timelines`. Not done
  now since it's cosmetic.
- Field-symmetry mandate (Rule 4 vs Rule 5): training-time dumps include
  `saturation` per layer; the C engine does not emit it in the TFRM v7 frame.
  Currently rendered only on the Learning tab so chat doesn't go blank, but it
  is a Rule 4/5 lockstep gap that should be closed (the dashboard has a
  `cSatL` panel that would also fire on the chat tab if `saturation` were in
  the SSE frame). Suggested fix: add `saturation` to `_build_c_mri_frame` from
  the FFN neuron magnitudes (count fraction with abs-int8 ≥ ~127*0.5).
- `memory` is always `[]` in C-engine SSE; the Brain (PyTorch path) populates
  it via stored neuron-fingerprint stories. Visible as the "memorization
  fingerprint" panel showing the empty-state on C generations. Working as
  designed today; flagged for awareness.
- The `26_mamba2_prototype/*` runs in the Training tab dropdown have CSVs
  with only 6 columns (no `tok_per_s`); the `cTpsT` chart blanks for them.
  Acceptable (these are out-of-band experiments).

## Restart

The MRI is a Flask process that serves `mri/static/conversation.html` at
runtime — no Jinja templating, no Python code referenced from the HTML.
The fix is HTML-only; **no MRI restart is required**. A browser hard-refresh
(`Ctrl+Shift+R`) is enough.
