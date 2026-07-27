# backend: typing samples

Store for recorded typing sessions: the per-keystroke evidence the speculative-draft
trigger is tuned against.

Owned by [typing_samples.py](../../../veritate_mri/runtime/typing_samples.py). Routes
live in [settings_routes.py](../../../veritate_mri/routes/settings_routes.py); the
recorder that produces a session is the Settings panel described in
[../frontend/settings_tab.md](../frontend/settings_tab.md).

## What it is

One JSON file per session under `data/typing_samples/`, machine-local and gitignored
with the rest of `data/`. A session carries `keys`, the per-keystroke records, plus the
threshold in force when it was recorded and the loaded model's sequence length.

A keystroke is `{t, gap, ch, len, ctx, word, done, still_after}`. `done` marks the
keystroke the typist was finished on, which labels every other keystroke as a known
not-done and makes a candidate rule scorable against real typing.

## Routes

- `POST /typing/samples` : store one session. `name` must be alphanumeric with `-` or
  `_`, so a name can never escape the store. An empty session is a 400.
- `GET /typing/samples` : name, keystroke count, labelled-question count, newest first.
- `GET /typing/samples/<name>` : the raw session.

## Pitfalls

- **Raw only.** No median, no percentile, no recommended threshold. A gap varies
  systematically with where in the text it falls, and a summary computed here would
  destroy the structure the store exists to preserve.
- `MAX_KEYSTROKES` is a bug backstop, not a session-length policy: a session past it
  means the recorder is looping, not that someone typed for a long time.
- A malformed file is skipped by `listing()` rather than failing the whole listing.

## Dependencies

- [readers/paths.py](../../../veritate_mri/readers/paths.py) : `REPO_ROOT`.
- Tests: [tests/mri/test_typing_samples.py](../../../tests/mri/test_typing_samples.py).
