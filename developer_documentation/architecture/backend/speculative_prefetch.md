# backend: speculative prefetch

Generates a reply for a draft prompt while the client is still typing, so the
real request flushes bytes that already exist instead of paying prefill. Time to
first byte on the C engine is prefill-bound (27 ms per prompt byte on the target
box, [c_engine.py:344](../../../veritate_mri/inference/backends/c_engine.py)),
and this moves that cost into the seconds a user spends typing.

Owned by [speculate.py](../../../veritate_mri/inference/speculate.py). The route
surface and the flush live in
[backends_routes.py](../../../veritate_mri/routes/backends_routes.py).

## what it is

One job process-wide, holding a draft prompt, the sampling params it was
speculated at, the subprocess that produced it, and a byte buffer. A client posts
the draft it would submit; the job runs ahead; the matching request takes the
buffer.

Three settings own it, all in
[settings.py](../../../veritate_mri/runtime/settings.py) `DEFAULTS`:

- `speculative_enabled` : off by default. The Generation tab and Settings both
  toggle this key.
- `speculative_bytes` : how far ahead a draft may run.
- `speculative_chunk_bytes` : engine turn size.

## how it works

- `start()` supersedes any current job and spawns the runner thread. Re-posting
  the same draft is a no-op, so a client may post freely.
- A job records why it stopped in `reason` (`running` / `done` / `budget` / `busy` /
  `error`), surfaced as `state` in the status payload. The rail names it, so a draft
  that stops is explainable rather than looking stuck.
- The runner issues one engine turn per chunk over `prompt + bytes so far`. The
  engine state cache ([state_cache.md](../../engine/state_cache.md)) restores the
  previous turn's snapshot, so a turn re-steps only the last chunk.
- Cancellation is "do not issue the next chunk". A live engine turn is never
  abandoned, so the pipe always reaches TEND and the subprocess is never
  respawned (a respawn would discard the in-memory prefix state this feature
  exists to build).
- **A real request always wins the engine.** `take()` cancels the live job on every
  call, hit or miss, and `_c_engine_stream` calls `stand_down()` for the callers
  that never take. A job left running is not merely wasted work: the runner
  reclaims the subprocess lock between chunks, so it can starve the request
  waiting on that lock for the whole draft budget while the user watches a blank
  panel.
- Between chunks the runner tests `sub.lock.locked()`. A held lock means a real
  request owns the engine, so the runner WAITS (up to `BUSY_WAIT_MAX_S`) and resumes
  when it frees, rather than abandoning the draft: giving up on the first contention
  left the rail stalled on a half-written reply. The lock is the live answer to "is the
  engine busy", and unlike an in-flight counter it cannot leak: an abandoned SSE
  generator that never runs its cleanup would pin a counter above zero and silently
  disable speculation forever. Worst case a real request waits one chunk for a turn
  already in flight.
- The dashboard posts a draft on a pause in typing, where a pause is measured
  against that typist's own median keystroke gap rather than a fixed timeout.
  Terminal punctuation fires almost immediately, deleting waits longer, and a draft
  under two words does not speculate at all. The client half is
  [../frontend/generation_tab.md](../frontend/generation_tab.md).
- Each draft carries an id. `take()` requires that id plus the same subprocess and
  the same prompt, and consumes the job either way, so a buffer is flushed once or
  never. The id is the client asserting nothing it would send has changed since
  the draft: the server cannot compare a request it has not received yet, and
  comparing knob-by-knob only moves the bug to whichever side serializes a value
  differently.
- `_c_engine_stream(prefetched=...)` emits a `kind:"prefetch"` frame carrying the
  count, then yields the buffered frames as they are, then continues the engine from
  `prompt + prefetched` with the remaining `max_new`. A buffer that already covers
  `max_new` returns without touching the engine.

## the trade

Speculation spends compute on drafts that may be discarded, so
`status()["stats"]` reports `served_bytes` against `spent_bytes`. Below roughly
50% the feature is buying latency with a doubling of energy per served answer.
The Settings panel renders that ratio and warns under 50%.

## pitfalls

- A speculative turn runs traced (`do_trace=True`) and buffers assembled MRI
  frames, so flushed bytes reach the dashboard with the same telemetry a live byte
  carries. The cost is that speculation is as expensive per byte as a real turn.
- The speculative prompt must be a strict prefix of the real one. A traced
  restore caps its scan at `n-1`, so a snapshot stored at exactly the real
  prompt's length is not reused. Chat framing gives this for free: the wire
  prompt ends on the assistant header, and the real request appends the reply.
- Sampling is not reproducible across a flush boundary: buffered bytes were
  sampled in the speculative turn. They are a valid sample at the same params,
  not the same sample a fresh run would draw.
- Only `/generate` claims a buffer. `/v1/chat/completions` and the chat page
  ignore it.

## dependencies

- [c_engine.py](../../../veritate_mri/inference/backends/c_engine.py) :
  `CTracedSubprocess.stream`, which serializes the one stateful subprocess.
- [state_cache.md](../../engine/state_cache.md) : makes chunked resume cheap.
- Tests: [tests/mri/test_speculate.py](../../../tests/mri/test_speculate.py).
