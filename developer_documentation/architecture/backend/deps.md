# deps

## What it is

Auto-installer for missing Python packages, torch wheel repair, and native OS helpers, at [veritate_core/plugin/deps.py](../../../veritate_core/plugin/deps.py). Backs the Auto tune modal's missing-dep recovery, the Detect Hardware modal's torch-build repair, and the HUD sensor row's auto-install of temperature helpers, so the user doesn't bounce to a terminal to unblock a benchmark or unlock the temperature HUD. Kept in sync with the one-shot launcher install in [veritate.py](../../../veritate.py), which handles the first-boot bootstrap.

## How it works

### Python packages

- [`is_installed(pkg)`](../../../veritate_core/plugin/deps.py#L42) — trivial `importlib.import_module` check. `pkg` is treated as the import name (with `-` → `_`). Callers whose dist name differs from the import name (`opencv-python` → `cv2`) must check via `importlib` directly.
- [`ensure(pkg, index_url=None)`](../../../veritate_core/plugin/deps.py#L129) — idempotent wrapper. When already importable, returns `{ok: True, method: "present"}` without invoking pip.
- [`install(pkg, index_url=None)`](../../../veritate_core/plugin/deps.py#L75) — escalation ladder, stops at the first success:
    1. `pip install --user <pkg>` (no elevation).
    2. `pip install <pkg>` (site-packages when the user has write access).
    3. Windows: [`ShellExecuteW(runas)`](../../../veritate_core/plugin/deps.py#L60) triggers a UAC prompt. Automated in the sense that no CLI-flag confirmation is needed; the OS-level admin consent is unavoidable.
    4. POSIX: `sudo -n pip install <pkg>`. `-n` fails immediately if sudo would prompt, so the request thread never hangs.
- Returns `{ok, method, stdout, stderr, needs_elevation}`. `needs_elevation` is True only when every automated path failed and interactive elevation is the only remaining option (rare on Windows because step 3 already triggers UAC).

### Torch build detection & repair

The single most common bad-install case: a Windows or Linux-x86 box with an NVIDIA GPU ends up running the CPU torch wheel (pip fell back to PyPI, or the launcher shipped before the two-phase fix landed). The runtime helpers below let the Detect Hardware modal see the mismatch and repair it in-place without a reinstall from scratch.

- Constants [`TORCH_CUDA_INDEX`](../../../veritate_core/plugin/deps.py#L35) (`https://download.pytorch.org/whl/cu128`) and [`TORCH_CPU_INDEX`](../../../veritate_core/plugin/deps.py#L36) (`https://download.pytorch.org/whl/cpu`) — the wheel indices used by `install_torch()`. Duplicated (not imported) from the launcher so a runtime repair still works when the launcher module isn't on `sys.path`.
- [`has_nvidia_gpu()`](../../../veritate_core/plugin/deps.py#L150) — probes for a usable NVIDIA GPU **without** importing torch (used to *decide* whether to reinstall torch). Cross-checks signals so a working GPU + driver still routes to the CUDA build when `nvidia-smi` isn't on PATH:
    - `nvidia-smi -L` listing a device (ground truth when reachable).
    - Windows: `nvcuda.dll` loadable via `ctypes.WinDLL`, or `Win32_VideoController` names an NVIDIA / GeForce / RTX / GTX / Quadro / Tesla adapter via PowerShell + CIM.
    - Linux: `/dev/nvidia0` present, or `libcuda.so.1` loadable via ctypes, or `/sys/class/drm/card*/device/vendor == 0x10de`.
- [`torch_state()`](../../../veritate_core/plugin/deps.py#L220) — introspects the running interpreter's torch. Returns `{installed, version, cuda_build, cuda_runtime, cuda_available, device_count}`. Import failure counts as `installed: False`.
- [`install_torch(force_cuda=None)`](../../../veritate_core/plugin/deps.py#L261) — reinstalls torch from the correct wheel index for the running box:
    - macOS → PyPI unchanged (arm64 gets MPS from the standard wheel, Intel gets CPU).
    - Linux ARM → PyPI (no CUDA wheels published for ARM).
    - Linux/Windows x86 → `TORCH_CUDA_INDEX` when an NVIDIA GPU is detected, else `TORCH_CPU_INDEX`.
    `force_cuda=True/False` overrides detection. Always passes `--index-url` (never `--extra-index-url` — see Pitfalls) and `--force-reinstall` because a stale wheel already in site-packages would be skipped by pip's normal resolver. Same escalation ladder as `install()`. Returns the install result plus `wheel_index` (naming what was installed) and `restart_required` (True on success — the running dashboard is still holding the old torch).
- [`status_snapshot()`](../../../veritate_core/plugin/deps.py#L312) — one-shot dict for the modal so a single `POST /sys/detect` produces everything the frontend needs to decide whether to auto-open the installer popup:
    - `torch` — `torch_state()` output.
    - `has_nvidia_gpu` — fresh probe, independent of the `sys_metrics` cache.
    - `needs_torch_cuda` — the specific "GPU present but torch is CPU" case (True when auto-repair should fire).
    - `torch_cuda_index` — the constant, so the frontend can pass it through to `/system/install_dep`.
    - `bitsandbytes` — `{installed: bool}`; only relevant when the CUDA torch path is OK.
    - `helpers.temp_sensor` — the per-arch helper id from `_temp_sensor_helper_id()`, or `None`.
- [`_temp_sensor_helper_id()`](../../../veritate_core/plugin/deps.py#L343) — picks the temperature-sensor helper matching the running arch: `mac_temp_arm` on Apple Silicon, `mac_temp_intel` on Intel Mac, `linux_lm_sensors` on any Linux with `apt-get`, else `None`. Windows returns `None` because LibreHardwareMonitor ships as a GUI-only installer — surfaced via a link in Settings, not this auto-install path.

### Native OS helpers

Some HUD features (CPU/GPU temperature) depend on OS-level tools pip can't install. [`HELPERS`](../../../veritate_core/plugin/deps.py#L359) is a keyed registry so a caller doesn't reason about brew/apt package names:

| id                 | os     | manager | pkg           | purpose                                                    |
| ------------------ | ------ | ------- | ------------- | ---------------------------------------------------------- |
| `mac_temp_arm`     | darwin | brew    | `macmon`      | Apple Silicon CPU + GPU temp/load; sudoless.               |
| `mac_temp_intel`   | darwin | brew    | `osx-cpu-temp`| Intel Mac CPU temp only.                                   |
| `linux_lm_sensors` | linux  | apt     | `lm-sensors`  | CPU temp via `psutil.sensors_temperatures()` on Linux.     |

[`install_helper(helper_id)`](../../../veritate_core/plugin/deps.py#L366) dispatches to `_run_brew` or `_run_apt`. Guards against wrong-OS invocation (returns `unsupported: True`), missing package manager (brew not on PATH → tells the user to install Homebrew), and passwordless-sudo prompts (`-n` on apt). Windows LibreHardwareMonitor is not auto-installable because it ships as a GUI-only installer; the Settings tab renders the manual instructions for that case.

### The launcher install

First-boot bootstrap lives in [veritate.py](../../../veritate.py) and runs once per venv (guarded by a requirements hash sentinel). It creates a venv, then installs deps from [requirements.txt](../../../requirements.txt) via the two-phase pattern in [`_install_torch_then_rest(py, index)`](../../../veritate.py#L531):

- **Phase 1:** `pip install --index-url <pytorch-index> torch`. This is the load-bearing choice — `--index-url` (not `--extra-index-url`) locks pip to the pytorch wheel index for this call only, so the resolver can't pick the same-versioned CPU wheel from PyPI. The original bug was exactly this: with `--extra-index-url`, boxes with NVIDIA GPUs kept landing on CPU torch because PyPI was still eligible.
- **Phase 2:** `pip install -r requirements.txt` from PyPI. Torch is already satisfied, so pip won't try to re-resolve it — the CUDA wheel just installed stays put.
- **Phase 3 (CUDA path only):** [`_verify_torch_cuda(py)`](../../../veritate.py#L431) runs `import torch; torch.cuda.is_available()` in a subprocess (the launcher's own interpreter may not have torch on `sys.path` yet). False here means the wheel is CPU-only, or the driver is too old, or a stale CPU torch is shadowing the venv — a clear WARNING is printed so the user isn't silently handed a CPU-only stack. The runtime `install_torch()` above can repair this case from the Detect Hardware modal without a full rebuild.

The launcher's [`_has_nvidia_gpu()`](../../../veritate.py#L345) mirrors `deps.has_nvidia_gpu()`: same nvidia-smi + `nvcuda`/WMI (Windows) + `/dev/nvidia0` / `libcuda.so.1` / DRM vendor id (Linux) cross-check. macOS is untouched — it rides PyPI (arm64 → MPS, Intel → CPU).

## Consumers

- [`POST /system/install_dep`](../../../veritate_mri/routes/trainers_routes.py#L116) — Auto tune modal's missing-import auto-heal. Body: `{pkg, index_url?}`. Calls `ensure()`.
- [`POST /system/install_helper`](../../../veritate_mri/routes/trainers_routes.py#L100) — HUD sensor row's auto-install and Settings-tab install-now button. Body: `{helper: <id>}`.
- [`POST /sys/detect`](../../../veritate_mri/routes/sys_routes.py#L66) — embeds `status_snapshot()` under `result["deps"]` so the Detect Hardware click can drive the "installing missing dependencies" popup and the HUD sensor auto-install without a second round-trip.

## Pitfalls

- **`--extra-index-url` is the CPU-torch-on-a-GPU-box footgun.** Both the launcher and `install_torch()` pass `--index-url` for torch specifically, precisely because pip's resolver treats extra-index as equally eligible: on Windows especially, PyPI's `torch` wheel is CPU-only and often wins the coin flip. Do not "helpfully" change these callsites to `--extra-index-url`.
- **`--user` on a system Python can put the install on a `PYTHONPATH` the running dashboard doesn't include.** The retry works because the Auto tune modal re-invokes `/trainers/run`, which spawns a fresh trainer subprocess that re-reads the user-site path. In-process code that fails to import after `install(--user)` still needs a dashboard restart because `sys.path` is only computed at interpreter start.
- **`install_torch()` returns `restart_required: True` on success.** The dashboard is still holding the old torch in memory; a fresh subprocess (bench, sysprobe) picks up the new wheel, but any code path importing torch in the request thread will still see the pre-repair build until the app is restarted.
- **The UAC prompt on Windows is asynchronous.** `ShellExecuteW` returns as soon as it dispatches; success/failure of the actual install is observed by the pipe on the elevated process, which this shim does not capture. The current code only knows whether the launch itself succeeded (`rc > 32`). If pip fails inside the elevated shell, the client sees `ok: True` — the modal's retry will re-hit the ImportError, which is the honest signal.
- **The HUD sensor row auto-installs instead of showing a button.** The helper set is small, per-arch, and unambiguous (macmon on arm Mac, lm-sensors on apt Linux), so a click adds friction without adding information. Windows still shows the manual LibreHardwareMonitor link because no auto-installable Windows helper exists (LibreHardwareMonitor is a GUI installer).
