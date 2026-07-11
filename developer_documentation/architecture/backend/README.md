# Backend components

Server-side architecture: Flask app, training pipeline, runtime, readers, engine, inference brain.

## Top-level layout

- [veritate_core_overview.md](veritate_core_overview.md) — training-side model + QAT
- [veritate_mri_overview.md](veritate_mri_overview.md) — Flask app + runtime + readers + routes
- [veritate_engine.md](veritate_engine.md) — C inference engine (`veritate_engine/v1/`)
- [veritate_shim.md](veritate_shim.md) — backwards-compat `veritate/` package
- [trainer_plugins.md](trainer_plugins.md) — `trainers/<id>/` plugin contract

## Flask app

- [app_py.md](app_py.md) — startup sequence, route registration, exception handling
- [routes.md](routes.md) — the route-module pattern; per-module summary
- [auth.md](auth.md): optional dashboard password gate
- [readers.md](readers.md) — data layer (all disk I/O routes through readers)

## Training pipeline

- [trainer_runner.md](trainer_runner.md) — subprocess management, PID file, global lock
- [save.md](save.md) — CSV contract + checkpoint save + dump suite
- [checkpoint_probe.md](checkpoint_probe.md) — what's in `hooks/step_<N>/` artifacts
- [train_stream.md](train_stream.md) — SSE pub/sub for live training payloads
- [export.md](export.md) — PyTorch `.pt` → engine `.bin` conversion
- [build_runner.md](build_runner.md) — engine rebuild orchestration
- [native_trainer.md](native_trainer.md) — low-level training loop

## Runtime

- [heartbeat.md](heartbeat.md) — Carpathian webhook integration
- [settings.md](settings.md) — `mri_settings.json` store
- [lifecycle.md](lifecycle.md) — Flask restart
- [sys_metrics.md](sys_metrics.md) — CPU/GPU/RAM detection

## Inference

- [inference_brain.md](inference_brain.md) — PyTorch inference (Generation tab backend)
- [veritate_engine.md](veritate_engine.md) — C engine (fast byte-level decode)
- [hybrid_chat.md](hybrid_chat.md): RAG chat endpoint (`/chat`, `/hybrid/*`)
- [knowledge_base.md](knowledge_base.md): `kb_build.py` index pipeline for the hybrid chat

## Federation

- [mesh.md](mesh.md) — optional inter-device federation (`veritate_mesh/`)
