# Logs tab

## What it is

Engine build status, system metrics snapshot (CPU, memory, GPU, temperatures), and a live tail of the in-memory log ring.

## How it works

Markup at [index.html:1219–1265](../../../veritate_mri/web/index.html#L1219).

- `loadEngineStatus()` fetches `/engine/status` to display build state (built? for which platform? latest version?).
- `_renderSysmetrics(snap)` ([index.js:8960](../../../veritate_mri/web/index.js#L8960)) renders the latest system snapshot (CPU usage, memory pressure, GPU activity, temperatures).
- Log ring rendered live via `/logs/stream` SSE. Initial state populated by `/logs/snapshot` on tab activation.

The log ring is in-memory only (no disk persistence). Server restart resets it. Lines are bounded to a fixed ring size to keep dashboard memory usage flat.

## Dependencies

- `/engine/status`, `/logs/snapshot`, `/logs/stream` routes.
- [sys_metrics.md](../backend/sys_metrics.md) backend for the periodic snapshots.
- Heartbeat reads error counts from the same ring (see [../backend/heartbeat.md](../backend/heartbeat.md)).

## Pitfalls

- If `/logs/stream` stalls, the tab silently stops updating. The view doesn't show a connection-state indicator yet.
- Log lines are JSON; very long messages (e.g., full tracebacks) are truncated by the ring to keep the wire payload bounded.
