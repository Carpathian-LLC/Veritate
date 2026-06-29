# mesh (federation)

## What it is

Optional inter-device federation at [veritate_mesh/](../../../veritate_mesh/). Lets multiple Veritate installs discover each other and share work (capability lookups, training delegation). Off by default.

## How it works

Roles configured via `mesh_role` in [settings.py](settings.md):

- `off` — no federation; the local install runs standalone.
- `hub` — runs a registry that other nodes connect to.
- `node` — connects to a hub and advertises its capabilities.
- `both` — runs both sides.

Modules:

- [client.py](../../../veritate_mesh/client.py) — node-side client; sends heartbeats to the hub.
- [hub.py](../../../veritate_mesh/hub.py) — registry server.
- [registry.py](../../../veritate_mesh/registry.py) — in-memory peer table.
- [capabilities.py](../../../veritate_mesh/capabilities.py) — what each node advertises.
- [protocol.py](../../../veritate_mesh/protocol.py) — message envelope and version.
- [node.py](../../../veritate_mesh/node.py) — node-side daemon.

Registered conditionally at [app.py:152](../../../veritate_mri/app.py#L152) only when `mesh_role` is non-off.

## Dependencies

- [routes/mesh_routes.py](../../../veritate_mri/routes/mesh_routes.py) — exposes peer listing + status.
- Settings keys: `mesh_role`, `mesh_hub_address`, `mesh_auth_token`.

## Pitfalls

- Federation adds an attack surface. The default is off; turn on only on trusted networks.
- Auth token is shared between hub and nodes via `mesh_auth_token` in settings — keep secret, don't commit.
- Capability advertisement is voluntary; a malicious node can lie about what it has. Treat federated results as advisory, not authoritative.
