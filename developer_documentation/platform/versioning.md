# Versioning

## What it is

The single standard for how every upgradable part of Veritate is numbered, when each number moves,
and what an upgrade obliges the platform to announce. The ledger is `versions.json` at the repo
root; the dashboard reads it through `GET /versions` ([sys_routes.py:128](../../veritate_mri/routes/sys_routes.py#L128)).

## The ledger

```json
{
  "channel": "stable",
  "build":    1,
  "engine":  "v1.0.0",
  "mri":     "v1.0.0",
  "format":  "v1.0.0",
  "trainers":"v1.0.0"
}
```

| key | covers | moves when |
|---|---|---|
| `channel` | release track: `stable` or `dev` | the box is switched between tracks |
| `build` | one global monotonic counter across the whole platform | every shipped change set, without exception |
| `engine` | `veritate_engine/` C runtime and per-arch kernels | engine source, kernel, or wire behavior changes |
| `mri` | `veritate_mri/` dashboard, routes, readers, training, inference | any platform Python or dashboard change |
| `format` | on-disk schemas: model `.bin`, settings, `config.json`, trainer manifest, hook artifacts | a persisted layout changes shape |
| `trainers` | the `trainers/` contract that upstream trainers compile against | the trainer-facing surface of `veritate_core.plugin` changes |

`build` is the number quoted in a bug report. The four component strings are semver and answer
"can this artifact still be read by that code".

## Semver, applied

For `engine`, `mri`, `format`, and `trainers`:

- **Major** (`v1.x.x` to `v2.0.0`): existing artifacts or callers stop working. A `.bin` written by the
  old version cannot be read. A route an extension calls is removed or changes meaning. Requires a
  build note and a migration path.
- **Minor** (`v1.0.x` to `v1.1.0`): capability is added, everything old still works. A new route, a new
  optional config key, a new kernel, a new dump artifact.
- **Patch** (`v1.0.0` to `v1.0.1`): behavior fixed with no surface change. Bug fix, performance work,
  a corrected number.

Additive is minor; subtractive or reinterpreted is major. A field gaining an optional key is minor; a
field changing units, type, or default is major.

## Compatibility rules

- `format` is the gate on user data. A major `format` bump means models, settings files, or corpora on
  disk need rebuilding, and the build note says exactly which and how.
- The `.bin` engine format is versioned inside the file. `veritate_mri/readers/bin.py` mirrors the C
  engine's accepted version set; both move in the same change set, and the reader rejects a version it
  cannot handle rather than misparsing it.
- `engine` and `mri` bump independently. A `.bin` written under one `format` is readable by every
  `engine` that declares support for it, regardless of the `engine` number.
- `trainers` is a contract number, not a code number. Upstream trainers pin against it. Bumping it major
  breaks every trainer on every box, so it moves only with a migration for the whole trainer set.

## What an upgrade obliges

1. `build` increments. Every change set, no exception.
2. Each component whose surface moved gets its semver bump under the rules above.
3. If a `format` major landed, a build note is mandatory
   ([build notes](#build-notes)) and states in user-facing words what to delete, rebuild, or rerun.
4. If a route or extension contract moved, the wiki `api/` and `extensions/` entries move in the same
   change set.
5. If a component was added or removed, its `developer_documentation/` file is created or deleted in the
   same change set.

## Build notes

One note per build, at `veritate_mri/data/wiki/build_notes/build_<N>.md`, where `<N>` matches
`versions.json::build`. Served by the wiki tab.

Required frontmatter: `title`, `date`, `tags`, `summary`.

Body is three to ten lines plus the version table: what changed, what action is required, the versions.
No design rationale, no internals. Rationale belongs in the component's `developer_documentation/` file.

A build that requires no action still gets a note; it says so in one line.

## Breaking-change notices

`BUILD_NOTICES` in [settings.py](../../veritate_mri/runtime/settings.py) maps a build number to a modal
the dashboard raises until it is acknowledged. Add an entry only when action is required. The map
is empty at 1.0.0 because there is no pre-1.0 state to migrate.

`pending_notices()` returns every notice above `last_acknowledged_build`, so a box that skipped several
builds sees all of them in order.

## Dependencies

- `versions.json` (repo root), read by [sys_routes.py](../../veritate_mri/routes/sys_routes.py).
- [settings.py](../../veritate_mri/runtime/settings.py) for `BUILD_NOTICES` and `last_acknowledged_build`.
- The wiki reader ([wiki.py](../../veritate_mri/readers/wiki.py)) for build-note rendering.

## Pitfalls

- Version strings carry a leading `v` in `versions.json`; `build` is a bare integer. Comparing them as
  strings sorts `v1.10.0` below `v1.9.0`. Parse before comparing.
- `BUILD_NOTICES` keys are integers, not strings. A JSON round trip turns them into strings and the
  modal silently stops firing.
- A build note whose `<N>` does not match a real `build` value is unreachable from the dashboard: the
  wiki lists it, but no notice ever points at it.
