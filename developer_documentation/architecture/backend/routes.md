# Routes

## What it is

The Flask routing layer lives under [veritate_mri/routes/](../../../veritate_mri/routes/). Each module owns one concern, exports a single `register(app)` function, and is registered from [app.py:140](../../../veritate_mri/app.py#L140). `auth_routes.register` runs first so its `before_request` guard precedes every other route. The page routes `/`, `/chat`, `/app` are defined directly on the app at [app.py:70](../../../veritate_mri/app.py#L70).

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
| [api_auth_routes.py](../../../veritate_mri/routes/api_auth_routes.py)    | Optional Bearer-key gate on `/generate`, `/agent/stream`, `/v1/*`. See [api_auth.md](api_auth.md) |
| [atlas_routes.py](../../../veritate_mri/routes/atlas_routes.py)          | Atlas (prompt index) operations                                         |
| [auth_routes.py](../../../veritate_mri/routes/auth_routes.py)            | Optional password gate; `/login`, `/logout`. See [auth.md](auth.md)     |
| [backends_routes.py](../../../veritate_mri/routes/backends_routes.py)    | PyTorch inference brain: `/generate`, `/meta`, neuron lookups           |
| [corpus_routes.py](../../../veritate_mri/routes/corpus_routes.py)        | Corpus discovery and usage stats; `/corpus/library/*`, `/corpus/mix/plan` |
| [engine_routes.py](../../../veritate_mri/routes/engine_routes.py)        | C inference engine control: build status, start, stop                   |
| [extensions_routes.py](../../../veritate_mri/routes/extensions_routes.py)| Extension catalog, install/uninstall, per-extension data (`/extensions/*`) |
| [hallucination_routes.py](../../../veritate_mri/routes/hallucination_routes.py) | `/hallucination/analyze`. See [hallucination_detector.md](hallucination_detector.md) |
| [rag_routes.py](../../../veritate_mri/routes/rag_routes.py)              | RAG corpus build + SFT jobs, `/rag/stop` (`/rag/*`)                     |
| [hybrid_routes.py](../../../veritate_mri/routes/hybrid_routes.py)        | RAG chat: `/hybrid/chat`, `/hybrid/health`. See [hybrid_chat.md](hybrid_chat.md) |
| [lifecycle_routes.py](../../../veritate_mri/routes/lifecycle_routes.py)  | Flask app restart, soft reload, kill. See [lifecycle.md](lifecycle.md)  |
| [logs_routes.py](../../../veritate_mri/routes/logs_routes.py)            | `/logs/snapshot` and `/logs/stream` SSE                                 |
| [mesh_routes.py](../../../veritate_mri/routes/mesh_routes.py)            | Federation peer discovery                                               |
| [models_routes.py](../../../veritate_mri/routes/models_routes.py)        | Model listing, config, checkpoints                                      |
| [pruning_routes.py](../../../veritate_mri/routes/pruning_routes.py)      | Structured neuron-pruning analysis + plugin generation                  |
| [runs_routes.py](../../../veritate_mri/routes/runs_routes.py)            | `/runs`, `/run/<name>/csv`, timeline endpoints, `/eval_sets` rebuild    |
| [settings_routes.py](../../../veritate_mri/routes/settings_routes.py)    | `/settings` GET/POST against `mri_settings.json`                        |
| [sys_routes.py](../../../veritate_mri/routes/sys_routes.py)              | System metrics, `/sys/detect`, `/versions`, `/heartbeat/status`         |
| [teacher_routes.py](../../../veritate_mri/routes/teacher_routes.py)      | Teacher config, `/teacher/models`, synth start/status/stop/samples/build_corpus |
| [train_routes.py](../../../veritate_mri/routes/train_routes.py)          | `/train/discovery`, `/train_stream` SSE                                 |
| [trainers_routes.py](../../../veritate_mri/routes/trainers_routes.py)    | Plugin listing, manifest, start/stop, `/trainers/tune_defaults` (auto tune write-back), repo sync |
| [wiki_routes.py](../../../veritate_mri/routes/wiki_routes.py)            | Wiki index and entry retrieval                                          |

Two underscore-prefixed modules in the same folder are shared helpers, not route modules and not registered: [_brain.py](../../../veritate_mri/routes/_brain.py) (PyTorch / C-engine model resolution and load helpers, shared with `app.main()`) and [_common.py](../../../veritate_mri/routes/_common.py) (`safe_route`, `safe_name`, `auto_thread_count`, `user_error`, `is_loopback`, `open_folder`).

## Dependencies

- [app_py.md](app_py.md): owns registration order.
- [readers.md](readers.md): every route reads through these.
- Per-module concern files (e.g., training, engine, runtime): each route is a thin wrapper around domain logic.

## Pitfalls

- Route order matters only when paths overlap. Currently all paths are disjoint; don't introduce overlapping paths without explicit ordering rules.
- Never `return Response(html, ...)` from a route. The global exception handler returns JSON; mixing HTML breaks the frontend's `r.json()` parsing.
- Adding a new route module requires importing and calling `register(app)` in [app.py](../../../veritate_mri/app.py).
- **Filesystem-path query params are loopback-gated.** `app.run` binds `0.0.0.0` and auth is off unless `VERITATE_DASHBOARD_PASSWORD` is set (see [auth.md](auth.md)), so any caller-chosen path param is an arbitrary-read surface for a LAN client. `/generate?rag=` and `/agent/stream?corpus=`/`?fs_root=` therefore reject non-loopback requests with `403` via `is_loopback(request.remote_addr)` ([_common.py:75](../../../veritate_mri/routes/_common.py#L75)); a same-machine (loopback) caller keeps full local-path access. Trade-off: a password-gated remote admin cannot pass a server-side path over the LAN, and must run RAG/agent from the server box (loopback). The in-process `fs_read` sandbox jails reads to its root but does not choose the root; this gate does.
- **`/generate` numeric params are guarded and bounded.** `temperature`/`top_k`/`max_new`/`ablate_*` parse inside a try that returns a `kind:"error"` SSE frame + `done` on a bad value (a bare `500` reaches the EventSource as an unparseable non-stream body). `top_k` clamps to `1..BYTE_VOCAB` (torch.topk raises outside that) and `max_new` caps at `MAX_NEW_CAP` (an uncapped `for _ in range(max_new)` holds the brain/C lock: DoS). Constants at the module top of [backends_routes.py](../../../veritate_mri/routes/backends_routes.py).
