# Backend components

Server-side architecture: Flask app, training pipeline, runtime, readers, engine, inference brain.

## Top-level layout

- [veritate_core_overview.md](veritate_core_overview.md): training-side model + QAT
- [veritate_mri_overview.md](veritate_mri_overview.md): Flask app + runtime + readers + routes
- [veritate_engine.md](veritate_engine.md): C inference engine (`veritate_engine/`)
- [trainer_plugins.md](trainer_plugins.md): `trainers/<id>/` plugin contract

## Flask app

- [app_py.md](app_py.md): startup sequence, route registration, exception handling
- [routes.md](routes.md): the route-module pattern; per-module summary
- [auth.md](auth.md): optional dashboard password gate
- [api_auth.md](api_auth.md): optional Bearer-key gate on the programmatic API surface
- [readers.md](readers.md): data layer (every disk read routes through a reader)

## Training pipeline

- [trainer_runner.md](trainer_runner.md): subprocess management, PID file, global lock
- [trainer_tuning.md](trainer_tuning.md): machine-local per-trainer setting overrides
- [save.md](save.md): CSV contract + checkpoint save + dump suite
- [checkpoint_probe.md](checkpoint_probe.md): what's in `hooks/step_<N>/` artifacts
- [train_stream.md](train_stream.md): SSE pub/sub for live training payloads
- [export.md](export.md): PyTorch `.pt` to engine `.bin` conversion
- [build_runner.md](build_runner.md): engine rebuild orchestration
- [native_trainer.md](native_trainer.md): low-level training loop
- [mix_planner.md](mix_planner.md): corpus mix weighting and spec emission
- [corpus_library.md](corpus_library.md): corpus catalog and installer
- [model_capabilities.md](model_capabilities.md): per-model generation-mode tiers

## Runtime

- [heartbeat.md](heartbeat.md): Carpathian webhook integration
- [net.md](net.md): shared HTTPS SSL context (certifi-backed)
- [settings.md](settings.md): `mri_settings.json` store
- [lifecycle.md](lifecycle.md): Flask restart
- [sys_metrics.md](sys_metrics.md): CPU/GPU/RAM detection
- [sysprobe.md](sysprobe.md): cross-platform hardware benchmark suite
- [deps.md](deps.md): dependency auto-install and torch wheel repair

## Inference

- [inference_brain.md](inference_brain.md): PyTorch inference (Generation tab backend)
- [hybrid_chat.md](hybrid_chat.md): RAG chat endpoint (`/chat`, `/hybrid/*`)
- [hallucination_detector.md](hallucination_detector.md): span-level hallucination grading
- [warm_models.md](warm_models.md): resident C-engine warm pool
- [speculative_prefetch.md](speculative_prefetch.md): generate ahead while a client types (`/prefetch`)

## Federation

- [mesh.md](mesh.md): optional inter-device federation (`veritate_mesh/`)
