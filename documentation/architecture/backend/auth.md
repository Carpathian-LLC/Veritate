# auth (dashboard password gate)

## What it is

Optional session-based password gate for the dashboard. Lives at [veritate_mri/routes/auth_routes.py](../../../veritate_mri/routes/auth_routes.py). Registered first in [app.py:140](../../../veritate_mri/app.py#L140) so its `before_request` guard runs ahead of every other route.

Auth is OFF unless the password env var is set, so a fresh install is never locked out.

## How it works

- `enabled()` returns true only when `VERITATE_DASHBOARD_PASSWORD` is set ([auth_routes.py:37](../../../veritate_mri/routes/auth_routes.py#L37)).
- `register(app)` sets `app.secret_key` from `VERITATE_SECRET_KEY`, or a random 32-byte key when unset ([auth_routes.py:46](../../../veritate_mri/routes/auth_routes.py#L46)). A random key means sessions do not survive a server restart.
- The `before_request` guard ([auth_routes.py:49](../../../veritate_mri/routes/auth_routes.py#L49)) allows the request when auth is disabled, the path is public, or `session["authed"]` is set. Otherwise:
  - `GET` redirects to `/login`.
  - any other method returns `{"ok": false, "error": "authentication required"}` with HTTP 401.
- `POST /login` compares the submitted password with `hmac.compare_digest` and, on match, sets `session["authed"]` and redirects to `/app`. A miss redirects to `/login?e=1` ([auth_routes.py:57](../../../veritate_mri/routes/auth_routes.py#L57)). Login form served from `veritate_mri/web/login.html`.
- `/logout` clears the session and redirects to `/` ([auth_routes.py:67](../../../veritate_mri/routes/auth_routes.py#L67)).

## Public surface

Open without a session even when auth is enabled ([auth_routes.py:26](../../../veritate_mri/routes/auth_routes.py#L26)):

- Exact paths: `/` (chat), `/login`, `/logout`, `/favicon.ico`.
- Prefixes: `/static`, `/chat` (hybrid page), `/hybrid` (hybrid chat API).

Everything else, including `/app` (the dashboard) and all management / training APIs, requires a session login.

## Environment

| Var | Effect |
| --- | ------ |
| `VERITATE_DASHBOARD_PASSWORD` | Sets the password and enables the gate. Unset = no auth. |
| `VERITATE_SECRET_KEY` | Flask session signing key. Unset = random per process (sessions reset on restart). |

## Pitfalls

- Without `VERITATE_SECRET_KEY`, every server restart invalidates active sessions.
- The gate protects the management surface only; the public chat page (`/`, `/chat`) and `/hybrid` API stay reachable by design. Do not assume `/hybrid/chat` is gated.
- Single shared password, no users or roles. It is a deployment lock, not an identity system.
