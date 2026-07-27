# backend: read-ahead

Reads the prompt into the engine while it is still being typed, so the request that
carries it skips the prefill. Predicts nothing.

Owned by [speculate.py](../../../veritate_mri/inference/speculate.py) (`read_ahead`,
`read_stand_down`, `read_status`). Route surface in
[backends_routes.py](../../../veritate_mri/routes/backends_routes.py).

## What it is

Answering has two costs: reading the prompt, then writing the reply. Reading is
prefill, measured at ~1.8 ms per prompt byte on the hybrid trunk, and no reply byte can
be written until it finishes. Read-ahead moves that cost into the seconds someone spends
typing.

The client posts the text so far **without the closing chat scaffold**, so it is a
strict prefix of the wire prompt the same text will produce on submit. The engine's
state cache ([state_cache.md](../../engine/state_cache.md)) restores the longest matching
prefix, so a request whose prompt begins with what was read resumes from there.

A chat client posts `messages` instead and the box renders the prefix through
`render_local_open` in [hybrid_routes.py](../../../veritate_mri/routes/hybrid_routes.py),
the same function `_render_local` builds the wire prompt from. Framing stays in the one
module that owns it, so a client cannot get the scaffold wrong.

## Why it needs no prediction

This is the distinction from [speculative_prefetch.md](speculative_prefetch.md), which
generates the *reply* ahead. That one has to know the user has finished typing, and a
wrong guess discards a whole generation. Read-ahead does the work the real request must
do regardless, so:

- **A miss costs nothing.** A diverging prompt restores a shorter prefix; the bytes
  already read were bytes the request needed read.
- **An edit costs only the tail.** The prefix before the edit stays warm. Measured: a
  mid-message edit still returned 601 ms against 940 ms cold.
- **There is nothing to tune per user.** Typing speed changes how often it runs, not
  whether it is correct.

Measured on `chat_200m` (M3 Ultra, 2026-07-27), a 502-character message, 552-byte wire
prompt, read in 12 steps as it was typed:

| | first byte |
|---|---|
| no read-ahead | 857 ms |
| read ahead while typing | 136 ms (6.3x) |

Engine time spent reading ahead was ~1 s spread across ~106 s of typing: a 0.9% duty
cycle on work that is not wasted.

## How it works

- `read_ahead(sub, prompt, params)` supersedes any read in flight; re-posting the same
  prefix is a no-op, so a client may post freely on every pause.
- The runner issues ONE untraced byte (`READ_MAX_NEW`). The engine writes its state
  cache at step 0 of a generation, so a single byte is the cheapest way to make it read
  the prompt; that byte is discarded.
- It never runs behind a held `sub.lock`: a held lock means a real request owns the
  engine. `_c_engine_stream` also calls `read_stand_down()` so no caller can be starved.
- `POST /prefill` with `prompt` or `messages` starts a read; omitting both stands down.
  `GET /backends` reports `c.read_ahead`.
- Gated by `_ahead_allowed`, which splits the dashboard from programmatic callers. A
  caller presenting a bearer token is programmatic (the dashboard never sends one), so
  `api_read_ahead_enabled` / `api_generate_ahead_enabled` govern them and
  `read_ahead_enabled` / `speculative_enabled` govern this box's own UI. Both read-ahead
  keys default on: unlike generating ahead, there is no case where reading costs more
  than it saves. The split exists because the box runs a single stateful engine, so a
  client working ahead is a client holding it.

## Pitfalls

- **Send the open prefix, not the wire prompt.** The closing scaffold
  (`<|im_end|>\n<|im_start|>assistant\n`) sits AFTER the text, so a closed form moves it
  on every keystroke and matches nothing in the cache. `wrapChatOpen` in
  [index.js](../../../veritate_mri/web/index.js) builds the open form; `wrapChat` is
  that plus the scaffold, which keeps the two in step by construction.
- A traced restore caps its scan at `n-1`, so a snapshot stored at exactly the request's
  prompt length is not reused. Read-ahead prefixes are shorter than the final prompt, so
  this is free.
- Only the C engine has a state cache. PyTorch requests ignore read-ahead.

## Dependencies

- [c_engine.py](../../../veritate_mri/inference/backends/c_engine.py) : the one stateful
  subprocess and its lock.
- [state_cache.md](../../engine/state_cache.md) : what makes a prefix reusable.
- Tests: [tests/mri/test_read_ahead.py](../../../tests/mri/test_read_ahead.py).
