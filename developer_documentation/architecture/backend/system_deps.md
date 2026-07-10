# system_deps

## What it is

Registry of Veritate's system-level (OS package) dependencies plus a secure installer, at [veritate_mri/training/system_deps.py](../../../veritate_mri/training/system_deps.py). Turns "bare-bones machine is missing X" into a detected, one-click-installable state in the dashboard. Companion to [build_runner.md](build_runner.md), which runs the actual compile.

## Scope

OS packages only, installed through the distro package manager. Currently:

- **clang** (`required`) — builds the inference engine.
- **git** (`required: false`) — git-based model and trainer sync; the built-in tarball updater ([app_sync](../../../veritate_mri/training/sync/app_sync.py)) works without it, and [git_runner](../../../veritate_mri/training/sync/git_runner.py) degrades to `RC_NOT_FOUND`, so it is recommended, not required.

Python itself and pip packages are **not** here: they are the pre-server bootstrap (the dashboard cannot run without them), so there is nothing for the running UI to install. Extend the set by adding a `DEPS` entry, not by touching call sites.

## How it works

- `status()` returns `{os, arch, package_manager, deps:[...], all_present, missing_required, can_auto_install}`. Each dep view: `{key, label, present, required, purpose, install_command, can_auto_install}`. `present` is a `shutil.which` probe; `can_auto_install` is true only when the dep is missing AND a supported package manager (`apt-get`/`dnf`/`pacman`) is present. macOS reports the `xcode-select --install` command but never auto-installs (interactive GUI dialog).
- `install(keys=None)` installs the missing deps (all, or the subset named in `keys`) in one package-manager call, streams output to the log ring, and returns `{ok, exit_code, installed, error, status}`. `keys` are validated against the registry; the package names come from the registry, never from the caller.
- `/engine/status` embeds `status()` under a `deps` key so the build-settle poll already carries it; `GET /engine/deps` returns it standalone; `POST /engine/deps/install` (body `{keys?:[...]}`) runs `install()`.

## Security model

The dashboard must not become a privilege-escalation surface (`app.run` binds `0.0.0.0`, reachable on the LAN). Three constraints hold together:

- **Loopback-only.** `POST /engine/deps/install` returns `403` unless `is_loopback(request.remote_addr)` (routes/[_common.py](../../../veritate_mri/routes/_common.py)). `X-Forwarded-For` is ignored (spoofable).
- **Fixed packages.** Nothing from the request reaches a shell as a package name; `install` runs the package manager via an argv list (no shell) built from the registry, with `keys` filtered to known entries.
- **Non-interactive sudo.** The command runs under `sudo -n` (skipped when already root). The server holds no credentials and never prompts. Escalation succeeds only if the OS already grants passwordless sudo, the operator's explicit choice. Absent that, install fails fast and the UI falls back to the `install_command` for the user to paste.

## Dependencies

- [readers/paths.py](../../../veritate_mri/readers/paths.py) — `current_os()`, `current_arch()`.
- [routes/_common.py](../../../veritate_mri/routes/_common.py) — `is_loopback()` route guard.
- Frontend: `_maybeOfferDeps` / `_renderDepsRemedy` (rebuild-failed remedy) and `_renderSystemDeps` / `_installDeps` (Settings > System dependencies panel) in [web/index.js](../../../veritate_mri/web/index.js).

## Pitfalls

- `install` does not run `apt-get update` first (matching the standalone [setup.sh](../../../veritate_engine/v1/build/setup.sh)); on a box with a stale/empty package index the install can fail, and the UI then shows the manual command. Run an index refresh in a terminal if that happens.
- macOS `install()` is a no-op that returns the `xcode-select` command: that installer is an interactive GUI dialog, not scriptable.
- `status()` runs `shutil.which` per dep on every `/engine/status` poll. Cheap (PATH stat), not cached: a package installed out-of-band shows up on the next poll, which is intended.
- `setup.sh` (the pre-server, Python-free `run setup.sh first` bootstrap for clang only) and this module both know clang's package name. They serve different contexts (CLI bootstrap vs in-app), so the small overlap is intentional; keep them in sync when adding a package manager.
