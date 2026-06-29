# train_stream

## What it is

In-process publish/subscribe for live training payloads at [veritate_mri/training/train_stream.py](../../../veritate_mri/training/train_stream.py). Trainers call `publish(payload)`; the Flask `/train_stream` SSE route subscribes and forwards events to connected dashboards.

## How it works

`publish(payload)` at [train_stream.py:33](../../../veritate_mri/training/train_stream.py#L33) accepts any dict; serializes to JSON and pushes to every active subscriber queue.

`/train_stream` route at [train_routes.py:50](../../../veritate_mri/routes/train_routes.py#L50) opens a generator that yields SSE-formatted lines from a per-connection queue. Disconnects clean up the queue.

The training tab opens the SSE via `EventSource` ([index.js:11447](../../../veritate_mri/web/index.js#L11447)).

## Schema

`train_stream` does not enforce a schema. The payload format is whatever the trainer publishes; dashboards consume what they understand and ignore the rest. Conventional fields used by the Training tab:

- `step`, `loss`, `lr`, `tok_per_s` — same names as the CSV contract.
- `layer_acts`, `neuron_top`, `lens` — per-token brain-stream telemetry.

Trainers must opt in by calling `publish()`. Most current trainers do.

## Dependencies

- [train_routes.py](../../../veritate_mri/routes/train_routes.py) — exposes the SSE endpoint.
- Frontend [data_flow.md](../frontend/data_flow.md) — SSE consumption.

## Pitfalls

- This is in-process. Trainer subprocesses publish back via... they don't, actually. The publisher is the dashboard process; trainer subprocesses can't directly call into this. Live brain-stream events from a subprocess require an IPC bridge (currently absent — the dashboard polls `train.csv` instead).
- Queues are per-connection. A slow client doesn't block publishers, but a stalled subscriber can grow its queue indefinitely if not bounded. Bounded queues are recommended for new subscribers.
