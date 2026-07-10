# heartbeat (Carpathian webhook)

## What it is

Background daemon that posts presence and diagnostics payloads to `https://api.carpathian.ai/webhook/veritate-heartbeat`. Lives at [veritate_mri/runtime/heartbeat.py](../../../veritate_mri/runtime/heartbeat.py). Started at dashboard launch from [app.py:234](../../../veritate_mri/app.py#L234).

## How it works

Two payload kinds, one daemon thread:

- **Presence** (`kind="presence"`) — every 5 minutes idle, every 60 seconds while training. Minimal envelope: machine_id, device_id, ts, uptime, restarts, error count, optional `training` block.
- **Diagnostics** (`kind="diagnostics"`) — every 5 minutes when `diagnostics_logs_enabled`. Heavier payload: hardware specs, log tails, plugin run tail.

Presence is always on: there is no off switch. `heartbeat_enabled` defaults `True` and the daemon starts unconditionally at launch; the first ping leaves ~5s in. The first-load consent modal is informational (`allowDecline: false`) and does not gate sending.

`device_id` is the user-facing device name. It defaults to a friendly auto-generated name (`brave-otter-07`) created once on first setup by `settings._random_device_name()` and stored as `device_name`; the user can rename it (≤15 chars). Only if `device_name` is blank does `_effective_device_id()` fall back to the first 8 chars of `machine_id`.

State persisted at `data/heartbeat_state.json` ([line 71](../../../veritate_mri/runtime/heartbeat.py#L71)):

- `machine_id` — 16 chars, `sha256("machine|" + fingerprint)`, bound to this box.
- `machine_fingerprint` — 16-char hash of `sys_metrics.stable_machine_key()` (Linux `/etc/machine-id`, macOS `IOPlatformUUID`, Windows `MachineGuid`) plus OS/arch.
- `host_token` — random per-install token (avoids shipping the macOS hostname).
- `restarts`, `total_runtime_secs`, `errors_pending` — counters.
- `last_send_ts`, `last_send_status`, `last_send_error` — last attempt outcome.

**Auto-deconfliction.** `_ensure_identity()` reconciles the persisted id with the current fingerprint on every read. A fresh install, or state copied to another box (fingerprint present but different), regenerates `machine_id` + `host_token` so two machines never heartbeat under the same id. An existing id with no fingerprint (pre-fingerprint state) is grandfathered — kept as-is and stamped with this machine's fingerprint — so upgrades don't churn established ids, while any later clone of that stamped state mismatches and regenerates. `reconcile_identity()` is called by `POST /sys/detect` so "detect system" repairs a cloned id on demand. `data/` is machine-local and must not be copied between installs (excluded from any deploy sync; the HTTP updater already preserves it).

## Training detection (two paths)

The provider callback `_TRAINING_FN` is set at [app.py:233](../../../veritate_mri/app.py#L233) to `_heartbeat_training`, which:

1. **Primary** — calls `plugin_runner.state()` ([app.py:264+](../../../veritate_mri/app.py#L264)). Returns `{plugin_id, started_at, model_name, n_params, shape}` when `STATUS_RUNNING`.
2. **Fallback** — `_detect_csv_based_training()` ([app.py:223](../../../veritate_mri/app.py#L223)) scans `models/<name>/train.csv` mtimes. Any CSV touched within 120s = active training. Catches direct-script trainers that bypass `plugin_runner`.

Without the fallback, presence pings falsely report idle during direct-script training, and the Carpathian dashboard flips the device offline mid-run.

## Tiers

The `analytics_advanced_enabled` setting gates which fields ship:

- Off — only `training_active: true` in the presence payload.
- On — full block: `plugin_id`, `started_at`, `model_name`, `n_params`, `shape`.

`heartbeat_send_errors` gates whether per-error detail (source + message) accompanies the count.

## Dependencies

- [training/trainer_runner.py](../../../veritate_mri/training/trainer_runner.py) — primary detection path.
- `models/<name>/train.csv` — fallback detection path.
- [runtime/settings.py](../../../veritate_mri/runtime/settings.py) — `heartbeat_enabled`, `analytics_advanced_enabled`, `diagnostics_logs_enabled`, `device_name`.
- [runtime/sys_metrics.py](../../../veritate_mri/runtime/sys_metrics.py) — hardware block in the analytics tier.

## Pitfalls

- Code changes to `_heartbeat_training` require a dashboard restart — the function is captured by closure at startup. Direct-script trainers won't be detected until the dashboard restarts after a heartbeat fix.
- The webhook URL is hardcoded. If Carpathian-side endpoint changes, every deployed install needs an update.
- 413 (payload too large) from the diagnostics endpoint silently drops the payload. The presence ping is always small enough to fit; only diagnostics can blow the budget.
