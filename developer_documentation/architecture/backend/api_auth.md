# api auth (programmatic API-key gate)

## What it is

Optional Bearer-key gate for this box's programmatic API surface. Lives at [veritate_mri/routes/api_auth_routes.py](../../../veritate_mri/routes/api_auth_routes.py). Independent of the dashboard password gate ([auth.md](auth.md)); the two are separate `before_request` hooks. Registered right after `auth_routes` in [app.py:141](../../../veritate_mri/app.py#L141).

Off by default: on a trusted LAN the API stays open. Operators exposing the box beyond the LAN set a key.

## How it works

- The gated surface: exact paths `/generate`, `/agent/stream`; prefix `/v1/` (`/v1/models`, `/v1/chat/completions`) ([api_auth_routes.py:31](../../../veritate_mri/routes/api_auth_routes.py#L31)).
- The `before_request` guard reads `settings.get()["api_key"]`. When empty, or when the path is not protected, it returns `None` (open). Otherwise it requires `Authorization: Bearer <key>`, compared with `hmac.compare_digest` (constant-time).
- On a match it calls `settings.record_api_key_use()` (increments `api_key_request_count`, sets `api_key_last_used_at`) and allows the request. On a miss it returns `{"ok": false, "error": "invalid or missing api key"}` with HTTP 401.

## Key storage + lifecycle

The key and its counters are three settings keys (`api_key`, `api_key_request_count`, `api_key_last_used_at`) in `data/mri_settings.json` (see [settings.md](settings.md)). No new store.

- Mint/rotate: `settings.rotate_api_key()` sets `api_key` to `vrt_` + `secrets.token_urlsafe(32)` and resets the counters.
- Clear/disable: `settings.clear_api_key()` blanks `api_key` and resets the counters, reopening the gate.
- Both are driven from the dashboard Settings tab via POST `/settings/api-key` (`{"action": "rotate"|"clear"}`).

## Never gated here

Dashboard pages (`/`, `/app`), `/chat`, `/hybrid/*`, `/static`, heartbeat, and every management/training route. This gate covers only the four programmatic endpoints above.

## Operator + Carpathian

The minted key must be given to Carpathian: it goes in the AIModel's per-region `api_key` field so Carpathian sends `Authorization: Bearer <key>` when reaching this box.

## Pitfalls

- Independent of the dashboard password gate; enabling one does not enable the other.
- `/generate` and `/agent/stream` are also independently loopback-restricted when a caller passes a filesystem path (see [routes.md](routes.md)); that guard is orthogonal to this key gate.
- Each authed request rewrites `mri_settings.json` (counter update). Fine for LAN traffic volumes; not a high-QPS design.
