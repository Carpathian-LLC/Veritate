# Veritate documentation

Public reference for building and using extensions on top of the Veritate platform.

## REST API

- **[api/external_api.md](api/external_api.md)** — the box's programmatic API for outside
  clients: the OpenAI-compatible endpoints plus `/generate` and `/agent/stream`, and the
  optional key gate. Start here to integrate this box from another service or app.
- **[api/internal_api.md](api/internal_api.md)** — the complete HTTP contract every extension
  codes against. Base URL, auth, error convention, SSE, and every endpoint with its
  params and response shape.

## Building an extension

- **[extensions/authoring.md](extensions/authoring.md)** — how to structure a
  self-contained extension that runs beside the dashboard, calls the API, loads a
  trained model, stays isolated, and how to surface and publish it. What exists today
  vs. what is not yet implemented.

## Platform internals

Architecture, runtime, training, and agent notes for developing the platform itself
live in [developer_documentation/](../developer_documentation/).
