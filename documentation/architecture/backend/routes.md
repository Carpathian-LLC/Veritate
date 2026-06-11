# Routes

## What it is

The Flask routing layer lives under [veritate_mri/routes/](../../../veritate_mri/routes/). Each module owns one concern, exports a single `register(app)` function, and is registered from [app.py:140](../../../veritate_mri/app.py#L140). `auth_routes.register` runs first so its `before_request` guard precedes every other route. The page routes `/`, `/app`, `/chat` are defined directly on the app at [app.py:70](../../../veritate_mri/app.py#L70).

## The pattern

```python
# veritate_mri/routes/<name>_routes.py
def register(app):
    @app.route("/<name>/...")
    def handler():
        ...
        return {"ok": True, ...}
```

Decorating happens inside `register` so the same module can be loaded but not registered (used by tests). Every handler returns JSON via Flask's auto-jsonification of dicts. Disk reads go through [readers](readers.md), never `open()` directly.

## Module inventory

| Module                                                                   | Concern                                                                 |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| [atlas_routes.py](../../../veritate_mri/routes/atlas_routes.py)          | Atlas (prompt index) operations                                         |
| [auth_routes.py](../../../veritate_mri/routes/auth_routes.py)            | Optional password gate; `/login`, `/logout`. See [auth.md](auth.md)     |
| [backends_routes.py](../../../veritate_mri/routes/backends_routes.py)    | PyTorch inference brain: `/generate`, `/meta`, neuron lookups           |
| [corpus_routes.py](../../../veritate_mri/routes/corpus_routes.py)        | Corpus discovery and usage stats                                        |
| [engine_routes.py](../../../veritate_mri/routes/engine_routes.py)        | C inference engine control: build status, start, stop                   |
| [rag_routes.py](../../../veritate_mri/routes/rag_routes.py)    | RAG corpus build + SFT jobs, `/rag/stop` (`/rag/*`)  |
| [hybrid_routes.py](../../../veritate_mri/routes/hybrid_routes.py)        | RAG chat: `/hybrid/chat`, `/hybrid/health`. See [hybrid_chat.md](hybrid_chat.md) |
| [lifecycle_routes.py](../../../veritate_mri/routes/lifecycle_routes.py)  | Flask app restart                                                       |
| [logs_routes.py](../../../veritate_mri/routes/logs_routes.py)            | `/logs/snapshot` and `/logs/stream` SSE                                 |
| [mesh_routes.py](../../../veritate_mri/routes/mesh_routes.py)            | Federation peer discovery                                               |
| [models_routes.py](../../../veritate_mri/routes/models_routes.py)        | Model listing, config, checkpoints                                      |
| [pruning_routes.py](../../../veritate_mri/routes/pruning_routes.py)      | Structured neuron-pruning analysis + plugin generation                  |
| [runs_routes.py](../../../veritate_mri/routes/runs_routes.py)            | `/runs`, `/run/<name>/csv`, timeline endpoints                          |
| [settings_routes.py](../../../veritate_mri/routes/settings_routes.py)    | `/settings` GET/POST against `mri_settings.json`                        |
| [sys_routes.py](../../../veritate_mri/routes/sys_routes.py)              | System metrics, `/sys/detect`, `/versions`, `/heartbeat/status`         |
| [teacher_routes.py](../../../veritate_mri/routes/teacher_routes.py)      | Teacher config, `/teacher/models`, synth start/status/stop/samples/build_corpus |
| [train_routes.py](../../../veritate_mri/routes/train_routes.py)          | `/train/discovery`, `/train_stream` SSE                                 |
| [trainers_routes.py](../../../veritate_mri/routes/trainers_routes.py)    | Plugin listing, manifest, start/stop, `/trainers/tune_defaults` (auto tune write-back), repo sync |
| [wiki_routes.py](../../../veritate_mri/routes/wiki_routes.py)            | Wiki index and entry retrieval                                          |

## Dependencies

- [app_py.md](app_py.md) — owns registration order.
- [readers.md](readers.md) — every route reads through these.
- Per-module concern files (e.g., training, engine, runtime) — each route is a thin wrapper around domain logic.

## Pitfalls

- Route order matters only when paths overlap. Currently all paths are disjoint; don't introduce overlapping paths without explicit ordering rules.
- Never `return Response(html, ...)` from a route. The global exception handler returns JSON; mixing HTML breaks the frontend's `r.json()` parsing.
- Adding a new route module requires importing and calling `register(app)` in [app.py](../../../veritate_mri/app.py).
