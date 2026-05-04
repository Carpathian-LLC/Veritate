# veritate_mri

The MRI server. Hosts the dashboard, runs the two inference backends, and is the only entry point users invoke.

## what runs where

```
veritate_mri/
  app.py              flask app. routes only. zero on-disk reads outside readers/.
  backends/
    pytorch.py        full-hook backend. loads a checkpoint into PyTorch and exposes Brain.stream().
    c_engine.py       fast backend. spawns veritate.exe chat_traced as a subprocess and parses TFRM frames.
  readers/            single ingestion entrypoint per artifact. nothing else opens files.
    paths.py          the only place path strings exist.
    models.py         list / validate model directories.
    config.py         load config.json.
    train_csv.py      parse train.csv.
    checkpoints.py    list step_<N>.pt / load one.
    hooks.py          per-step hook artifacts (8 dump types). single ingestion point.
    engine.py         engine_versions.json registry.
    bin.py            veritate.bin header (precision + bin format version).
  tools/              invokable utilities, not part of the request path.
    diff.py           PyTorch vs C-engine logit divergence harness.
    perf_trace.py     per-stage timing capture.
    build_memory.py   neuron-memory probe builder.
  static/
    index.html        the dashboard. three tabs: Generation, Learning, Live Training.
  logs/               in-memory build and runtime status. moved here from repo root.
  readme.md           this file.
```

## data flow

A dashboard request lands at `app.py`. The route hands the model name to the matching reader. The reader resolves the path via `readers/paths.py`, parses the artifact, returns a Python dict. The route serializes JSON back to the frontend. No glob, no path string, no file open lives outside `readers/`.

The two backends are wired into `app.py` once at startup:

- `BRAIN` is a PyTorch `Brain` instance loaded from the model the user picked at launch (`models/<name>/checkpoints/step_<N>.pt`). It serves `/generate?backend=pytorch` and `/neuron/...`.
- `C_SUBPROCESS` is a `CTracedSubprocess` running `veritate.exe chat_traced` against `models/<name>/veritate.bin`. It serves `/generate?backend=c`. If the binary for the current OS+arch is absent the subprocess is `None` and the C backend is disabled in the dashboard.

## adding a new artifact

Per the project rule "no ad-hoc code anywhere": every artifact has exactly one reader.

1. Add the file name and parser kind to `readers/paths.py::HOOK_ARTIFACTS` (or write a sibling reader if it lives outside `hooks/`).
2. Update `documentation/hooks/contract.md` to list the new field set.
3. Add a render path in the dashboard.

No route, backend, or tool touches paths directly. Layout changes are localized to `readers/paths.py`.

## adding a new route

Routes are thin. Read inputs from `request`, ask one or more readers for parsed data, return JSON. No file system, no parsing. About 5 to 20 lines per route.
