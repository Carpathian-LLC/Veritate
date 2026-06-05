# Training tab

## What it is

Live + historical view of training runs. Shows loss curves, learning rate, throughput, grad norm, plateau health, latest stats, plus a real-time SSE feed of per-step brain frames.

## How it works

Markup at [index.html:901–1043](../../../veritate_mri/web/index.html#L901).

- Run picker at `#runPicker` populated by `loadRunsList()` ([index.js:3418](../../../veritate_mri/web/index.js#L3418)) which fetches `/runs`.
- `loadTrainCsv()` ([index.js:3485](../../../veritate_mri/web/index.js#L3485)) fetches the selected run's `train.csv` and parses train/val rows.
- Charts: `cLossT` (loss), `cLrT` (lr schedule), `cTpsT` (throughput), `cGnT` (grad norm), plus confidence-evolution and reading-grade panels.
- `#trainPlateau` div displays one of six health states from the plateau detector (IMPROVING, PLATEAU, REGRESSING, SLOWING, BOUNCING, WARMING).
- `#trainLatest` shows the last row's metrics.

Polling starts when the tab activates ([index.js:2122–2123](../../../veritate_mri/web/index.js#L2122)):

- `startTrainPolling()` ([index.js:6237](../../../veritate_mri/web/index.js#L6237)) — three intervals (CSV 5s, runs 30s, classroom 30s).
- `trainStreamStart()` ([index.js:11447](../../../veritate_mri/web/index.js#L11447)) — opens `/train_stream` SSE.

Polling stops on tab switch.

## Dependencies

- `/runs` and `/run/<name>/csv` from [runs_routes.py:256](../../../veritate_mri/routes/runs_routes.py#L256).
- `/train_stream` SSE from [train_routes.py:50](../../../veritate_mri/routes/train_routes.py#L50).
- Backend training CSV contract at [save.py:38](../../../veritate_mri/training/save.py#L38) — `step,split,loss,lr,grad_norm,tok_per_s,wall_s,seed`. Any change to that header breaks this tab.
- [canvas_rendering.md](canvas_rendering.md) for chart helpers.

## Pitfalls

- A run with no `train.csv` doesn't appear in `/runs`. Trainer skeletons need to write at least the header at startup for visibility.
- SSE reconnects are silent. If the backend dies and restarts, the stream stops but the polled CSV keeps the tab usable.
- The plateau detector uses smoothed differences; very short runs (under ~50 steps) sit in WARMING the whole time.
