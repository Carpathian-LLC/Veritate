# tests

Platform regression tests. Run before every merge to `dev`. CI runs the same suite.

## run

```bash
pip install -r requirements.txt    # pytest is bundled
pytest tests/                      # all (from repo root)
pytest tests/engine/               # one category
pytest tests/ -k engine_load       # by name
pytest tests/ -v                   # verbose
pytest tests/ -m "not slow"        # skip slow tests
pytest tests/ -m slow              # run only slow tests
```

`tests/pytest.ini` is the suite's config root. Pytest discovers it when invoked as `pytest tests/` from the repo root or as `pytest` from inside `tests/`. The whole suite (config + cases + fixtures) is self-contained under `tests/`.

## what's covered

| folder | what it verifies |
|---|---|
| [`engine/`](engine/) | engine builds, loads each `.bin` in `models/`, rejects malformed binaries with a clean error. |
| [`export/`](export/) | export round-trip: synthetic checkpoint -> `.bin` -> header bytes match expected v9 / v11 layout. |
| [`mri/`](mri/) | dashboard endpoints respond 200 + correct JSON shape via the Flask test client (no live server). |
| [`plugin_contract/`](plugin_contract/) | `veritate.plugin` public surface (save / paths / model) hasn't drifted. The plugins repo depends on this. |

## rules for adding a test

1. **One assertion per concept.** A test that asserts five unrelated things is five tests.
2. **Heavy on the docstring.** Every test function starts with a one-line docstring stating exactly what it verifies. No "why", no "how" -- just the behavior. Stupid concise.
3. **Self-contained.** No shared global state. Use pytest fixtures from `conftest.py` for setup.
4. **Deterministic.** No network calls (mock them). No real wall-clock timing assertions. Seed every RNG.
5. **Cleans up.** A test that writes files cleans them up via `tmp_path` or `addfinalizer`.
6. **Marks slow tests.** Anything that takes > 5 seconds gets `@pytest.mark.slow`.
7. **When functionality is added, a test is added in the same commit.** No new feature lands without a test that would have failed before it.

## file structure

```
tests/
|-- README.md           # this file
|-- conftest.py         # shared fixtures + pytest configuration
|-- engine/
|-- export/
|-- mri/
`-- plugin_contract/
```

Each subfolder is a Python package only if it needs `__init__.py`. Pytest discovers `test_*.py` files automatically.
