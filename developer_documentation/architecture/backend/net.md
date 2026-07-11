# net

## What it is

One helper, [veritate_mri/runtime/net.py](../../../veritate_mri/runtime/net.py), that builds the SSL context every outbound HTTPS call in the platform shares.

## How it works

- `ssl_context()` returns `ssl.create_default_context(cafile=certifi.where())` when `certifi` is importable, else a plain `ssl.create_default_context()`. certifi carries a CA bundle so verification works on framework Python builds (macOS) whose system store is empty until "Install Certificates.command" is run.

## Dependencies

- `certifi` (optional, imported lazily inside the function).

## Consumers

Every platform module that opens an `https://` URL passes `context=` from this helper: `runtime/heartbeat.py`, `runtime/ai_assist.py`, `teacher/client.py`, and the four `training/sync/{trainers,corpus,models,app}_sync.py` updaters. Modules that need a context at import time bind `_SSL_CTX = net.ssl_context()` once; `app_sync.py` calls `net.ssl_context()` per request.

## Pitfalls

- Extensions live behind an isolation boundary and do not import platform runtime modules; `extensions/data.py` keeps its own context.
