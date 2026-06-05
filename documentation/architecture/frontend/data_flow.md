# Data flow

## What it is

How the frontend gets data from the Flask backend: JSON polling for most things, Server-Sent Events (SSE) for streaming, no WebSockets.

## Routes the frontend calls

| Route                          | Method | Purpose                                                |
| ------------------------------ | ------ | ------------------------------------------------------ |
| `/`                            | GET    | Serves `index.html`                                    |
| `/meta`                        | GET    | Current model metadata (layers, params, version)       |
| `/runs`                        | GET    | All training runs (name, mtime, size, n_rows, caps)    |
| `/run/<name>/csv`              | GET    | Raw `train.csv` for a run                              |
| `/run/<name>/<timeline>`       | GET    | Learning tab: timeline data for one checkpoint         |
| `/heartbeat/status`            | GET    | Heartbeat state and last-send result                   |
| `/sys/specs`                   | GET    | CPU, GPU, RAM detection snapshot                       |
| `/logs/snapshot`               | GET    | Initial log ring contents                              |
| `/logs/stream`                 | SSE    | Streaming log lines                                    |
| `/train_stream`                | SSE    | Live training frames (TFRM-lite)                       |
| `/generate`                    | POST   | Inference request, token-by-token SSE response         |

Route definitions live under [veritate_mri/routes/](../../../veritate_mri/routes/); each module exports a `register(app)` function called from [app.py:128–150](../../../veritate_mri/app.py#L128).

## Polling cadence

`startTrainPolling()` at [index.js:6237](../../../veritate_mri/web/index.js#L6237) starts three intervals when the training tab activates:

- `loadTrainCsv()` — every 5s, refreshes the CSV for the selected run.
- `loadRunsList()` — every 30s, refreshes the dropdown of available runs.
- Classroom/confidence — every 30s.

`stopTrainPolling()` clears all three. Polling only runs while the training tab is active so background tabs don't burn requests.

Coral Lab uses the same `/runs` + `/run/<name>/csv` routes with its own 5s interval in [coral_lab.js](../../../veritate_mri/web/coral_lab.js). No new routes were added for Coral.

## SSE streams

Two SSE feeds:

- `/logs/stream` — opened at [index.js:8706](../../../veritate_mri/web/index.js#L8706). Each `event.data` is a JSON log entry; appended to the log ring view.
- `/train_stream` — opened at [index.js:11447](../../../veritate_mri/web/index.js#L11447). Each frame is a JSON training payload (step, loss, neuron telemetry, etc.) pushed by [train_stream.py:33](../../../veritate_mri/training/train_stream.py#L33).

`EventSource` doesn't expose HTTP status codes on failure. The logs view re-polls `/logs/snapshot` if the stream stalls to detect backend disconnects.

## Error handling

`_backendErrMsg(e)` at [index.js:58–67](../../../veritate_mri/web/index.js#L58) translates network errors into messages like "backend offline. relaunch via start.bat / start.command." Used by every fetch wrapper to surface failures without dumping stack traces into the UI.

## Dependencies

- [tab_system.md](tab_system.md) — polling starts/stops with tab activation.
- Backend route modules at [veritate_mri/routes/](../../../veritate_mri/routes/).
- Backend reader modules at [veritate_mri/readers/](../../../veritate_mri/readers/) — all disk reads route through these.

## Pitfalls

- Polling pauses when the tab is hidden. Switching back has a 5s stale window before the next refresh.
- SSE connections silently re-open on transient failures. Surfacing connection state to the UI requires explicit `onerror` handlers.
- Long-running fetches (e.g., `/runs` over a slow disk) can pile up if the poll interval is shorter than the response time. Watch the network tab if requests overlap.
