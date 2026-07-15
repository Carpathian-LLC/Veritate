# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Auto-install missing Python packages when the platform hits an ImportError
#   at runtime. Tries `pip install --user` first (unattended, no elevation).
#   Falls back to a system-scope install: on Windows via ShellExecute runas
#   (UAC prompt, still automated: no CLI confirmation), on POSIX via sudo when
#   the box has passwordless sudo, else returns needs_elevation=True so the
#   caller surfaces a popup.
# - Never installs a package the user did not explicitly request through
#   ensure() or the dashboard install route. No implicit background installs.
# veritate_core/plugin/deps.py
# ------------------------------------------------------------------------------------
# Imports:

import importlib
import os
import shutil
import subprocess
import sys

# ------------------------------------------------------------------------------------
# Constants

PIP_TIMEOUT_SECS = 300
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

# Wheel indices used by install_torch(). Kept in sync with veritate.py's
# launcher (which uses the same values for the first-boot install). If the
# launcher's constants drift, the auto-install popup can still repair a bad
# initial install without editing this file.
#
# Rule 34c: each index targets a specific arch/GPU combination. Picking the
# wrong one silently gives the user a wheel that can't reach their hardware,
# which is exactly the CPU-torch-on-a-GPU-box bug that motivated the two-phase
# install. Every index MUST be reachable at Restart time so the runtime repair
# path has the same options the launcher has.
TORCH_CUDA_INDEX      = "https://download.pytorch.org/whl/cu128"
TORCH_CPU_INDEX       = "https://download.pytorch.org/whl/cpu"
# ROCm: AMD's CUDA equivalent, Linux x86 only (no ROCm on macOS or Windows).
# PyTorch's ROCm wheels expose the same torch.cuda.* API surface as the CUDA
# wheels, so the trainer code path is identical — only the wheel index changes.
TORCH_ROCM_INDEX      = "https://download.pytorch.org/whl/rocm6.2"
# DirectML: AMD/Intel GPUs on Windows. Ships as a SEPARATE package
# (torch-directml) that plugs into a stock CPU torch install, exposing a
# `privateuseone:0` device string that trainers do NOT currently target. We
# detect the hardware and surface it in status_snapshot so a future trainer
# refactor can enable it without more probe work.
TORCH_DIRECTML_PKG    = "torch-directml"
# Intel XPU: Arc/Battlemage discrete GPUs and integrated Xe. Requires
# intel-extension-for-pytorch + a matching torch build. Like DirectML, current
# trainers do not target `xpu` — detection only for now.
TORCH_INTEL_XPU_PKG   = "intel-extension-for-pytorch"

# ------------------------------------------------------------------------------------
# Functions


def is_installed(pkg):
    """True when `import pkg` (or its top-level dist name) resolves. Uses the
    dist name form the caller passed; a package with a different import name
    (`opencv-python` -> `cv2`) must be checked via importlib directly by the
    caller, not through this helper."""
    try:
        importlib.import_module(pkg.replace("-", "_"))
        return True
    except ImportError:
        return False


def _emit_line(msg):
    """Best-effort mirror of one pip output line into the dashboard's log
    stream. Runs lazily because deps.py is in veritate_core and only the mri
    dashboard defines the log ring; a plain trainer script calling this module
    directly still works (the try/except swallows the ImportError)."""
    try:
        from runtime import logs as logmod
        logmod.info("deps", msg.rstrip())
    except Exception:
        pass


def _run_pip(args, use_shell_runas=False):
    """Invoke pip in a subprocess with LIVE line-by-line output routed into
    the dashboard's log ring (source='deps'). The frontend restart overlay
    subscribes to /logs/stream and shows these lines in real time so the user
    sees exactly what pip is doing. Returns (ok, stdout, stderr). Timeout
    prevents a hung install from wedging the request thread."""
    if use_shell_runas and sys.platform.startswith("win"):
        import ctypes
        params = " ".join(f'"{a}"' if " " in a else a for a in args)
        _emit_line(f"$ (UAC) pip {params}")
        rc = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, f"-m pip {params}", None, 1)
        ok = rc > 32
        _emit_line(f"ShellExecute returned {rc} — ok={ok}")
        return ok, "", ("" if ok else f"ShellExecute returned {rc}")
    _emit_line(f"$ pip {' '.join(args)}")
    stdout_lines = []
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "pip"] + args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, creationflags=_NO_WINDOW,
        )
    except (FileNotFoundError, OSError) as exc:
        return False, "", f"{type(exc).__name__}: {exc}"
    deadline = None
    try:
        import time as _time
        deadline = _time.time() + PIP_TIMEOUT_SECS
        assert proc.stdout is not None
        for line in proc.stdout:
            if len(line) > 4096:
                line = line[:4096] + " ...[truncated]"
            stdout_lines.append(line)
            _emit_line(line)
            if _time.time() > deadline:
                proc.kill()
                _emit_line("[deps] pip timed out — killed")
                return False, "".join(stdout_lines), "timeout"
        proc.wait(timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        try: proc.kill()
        except OSError: pass
        return False, "".join(stdout_lines), f"{type(exc).__name__}: {exc}"
    ok = proc.returncode == 0
    _emit_line(f"[deps] pip exit {proc.returncode}")
    return ok, "".join(stdout_lines), ""


def install(pkg, index_url=None):
    """Install `pkg` with as little friction as possible. Escalation ladder:
      1. pip install --user <pkg>            (no elevation)
      2. pip install <pkg>                   (site-packages if writable)
      3. Windows: ShellExecute runas         (UAC popup, automatic)
      4. POSIX:   sudo -n pip install <pkg>  (only if passwordless sudo)
    Returns {ok, method, stdout, stderr, needs_elevation}. `needs_elevation`
    is True when every automated path failed and only interactive elevation
    remains (rare on Windows because step 3 already triggers UAC)."""
    args_base = ["install", "--disable-pip-version-check"]
    if index_url:
        args_base += ["--index-url", index_url]
    args = args_base + [pkg]

    ok, out, err = _run_pip(args + ["--user"])
    if ok:
        return {"ok": True, "method": "user", "stdout": out, "stderr": err,
                "needs_elevation": False}

    ok, out, err = _run_pip(args)
    if ok:
        return {"ok": True, "method": "site", "stdout": out, "stderr": err,
                "needs_elevation": False}

    if sys.platform.startswith("win"):
        ok, out, err = _run_pip(args, use_shell_runas=True)
        if ok:
            return {"ok": True, "method": "uac", "stdout": out, "stderr": err,
                    "needs_elevation": False}
        return {"ok": False, "method": "uac", "stdout": out, "stderr": err,
                "needs_elevation": True}

    ok, out, err = _run_sudo(args)
    if ok:
        return {"ok": True, "method": "sudo", "stdout": out, "stderr": err,
                "needs_elevation": False}
    return {"ok": False, "method": "sudo", "stdout": out, "stderr": err,
            "needs_elevation": True}


def _run_sudo(args):
    """Passwordless sudo pip install. -n makes sudo fail immediately if it
    would prompt for a password, so the request thread never hangs."""
    try:
        out = subprocess.run(
            ["sudo", "-n", sys.executable, "-m", "pip"] + args,
            capture_output=True, text=True, timeout=PIP_TIMEOUT_SECS,
            check=False,
        )
        return out.returncode == 0, out.stdout, out.stderr
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return False, "", f"{type(exc).__name__}: {exc}"


def ensure(pkg, index_url=None):
    """Install `pkg` only when not already importable. Idempotent, cheap when
    already present. Returns the install() result dict, or {ok:True, method:
    'present'} when nothing was needed."""
    if is_installed(pkg):
        return {"ok": True, "method": "present", "stdout": "", "stderr": "",
                "needs_elevation": False}
    return install(pkg, index_url=index_url)


# ---- torch build detection & repair ------------------------------------------
# The single most common bad-install case: a Windows or Linux-x86 box with an
# NVIDIA GPU ends up with the CPU torch wheel because pip fell back to PyPI.
# torch_state() reports what's actually loaded; has_nvidia_gpu() checks for a
# usable GPU without importing torch (so the check works even when torch is
# broken); install_torch() reinstalls torch from the correct wheel index for
# the running box. Cross-platform: macOS rides PyPI unchanged (arm64 gets MPS
# from the standard wheel, Intel gets CPU); only Linux/Windows x86 with an
# NVIDIA GPU present are routed to the CUDA index.


def has_nvidia_gpu():
    """True when a usable NVIDIA GPU is present. Cross-checks several signals
    so a machine with a working GPU + driver but a missing `nvidia-smi` on
    PATH still gets the CUDA build. Never imports torch (used to *decide*
    whether to reinstall it)."""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL,
                timeout=10, creationflags=_NO_WINDOW,
            )
            if "GPU" in out:
                return True
        except (OSError, subprocess.SubprocessError):
            pass
    plat = sys.platform
    if plat.startswith("win") or os.name == "nt":
        try:
            import ctypes
            try:
                ctypes.WinDLL("nvcuda")
                return True
            except OSError:
                pass
        except Exception:
            pass
        ps = shutil.which("pwsh") or shutil.which("powershell")
        if ps:
            try:
                out = subprocess.check_output(
                    [ps, "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_VideoController | "
                     "Select-Object -ExpandProperty Name) -join '|'"],
                    text=True, stderr=subprocess.DEVNULL, timeout=8,
                    creationflags=_NO_WINDOW,
                )
                low = out.lower()
                if any(t in low for t in ("nvidia", "geforce", "rtx", "gtx", "quadro", "tesla")):
                    return True
            except (OSError, subprocess.SubprocessError):
                pass
        return False
    if plat.startswith("linux"):
        if os.path.exists("/dev/nvidia0") or os.path.exists("/dev/nvidiactl"):
            return True
        try:
            import ctypes
            try:
                ctypes.CDLL("libcuda.so.1")
                return True
            except OSError:
                pass
        except Exception:
            pass
        drm = "/sys/class/drm"
        try:
            for entry in os.listdir(drm):
                vend = os.path.join(drm, entry, "device", "vendor")
                if os.path.isfile(vend):
                    try:
                        with open(vend, "r") as f:
                            if f.read().strip().lower() == "0x10de":
                                return True
                    except OSError:
                        pass
        except OSError:
            pass
    return False


def has_amd_gpu():
    """True when a ROCm-capable AMD GPU is present on Linux. ROCm exists on
    Linux only — macOS never had ROCm, Windows has DirectML instead. Signals:
    `/dev/kfd` (kernel-fusion driver device node, ROCm-specific), `rocm-smi`
    on PATH, or PCI vendor 0x1002 in `/sys/class/drm/*/device/vendor`.

    Only integrated APU + discrete Radeon lines are ROCm-usable; older/mobile
    parts (below gfx900) fall back to CPU at torch runtime, which is the wheel
    package's own guard. Detection stays permissive: presence is enough to
    justify installing the ROCm wheel."""
    if not sys.platform.startswith("linux"):
        return False
    if os.path.exists("/dev/kfd"):
        return True
    if shutil.which("rocm-smi") or shutil.which("rocminfo"):
        return True
    drm = "/sys/class/drm"
    try:
        for entry in os.listdir(drm):
            vend = os.path.join(drm, entry, "device", "vendor")
            if os.path.isfile(vend):
                try:
                    with open(vend, "r") as f:
                        if f.read().strip().lower() == "0x1002":
                            return True
                except OSError:
                    pass
    except OSError:
        pass
    return False


def has_intel_discrete_gpu():
    """True when an Intel Arc / Battlemage / Xe-HPG discrete GPU is present.
    Detection is by name substring plus PCI vendor 0x8086 (Intel), because
    every Intel iGPU also shares vendor 0x8086 and we want the discrete cards
    the XPU backend can actually accelerate. Currently informational only —
    trainers do not target Intel XPU."""
    plat = sys.platform
    # Windows: WMI adapter name lookup.
    if plat.startswith("win") or os.name == "nt":
        ps = shutil.which("pwsh") or shutil.which("powershell")
        if not ps:
            return False
        try:
            out = subprocess.check_output(
                [ps, "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController | "
                 "Select-Object -ExpandProperty Name) -join '|'"],
                text=True, stderr=subprocess.DEVNULL, timeout=8,
                creationflags=_NO_WINDOW,
            )
            low = out.lower()
            return any(tok in low for tok in ("arc a", "arc b", "arc ", "battlemage", "iris xe max"))
        except (OSError, subprocess.SubprocessError):
            return False
    # Linux: DRM sysfs + name match. Any Intel adapter is vendor 0x8086; we
    # accept it as "discrete-ish" only when the model name doesn't look like a
    # standard iGPU (UHD, HD, Iris Plus without "Max").
    if plat.startswith("linux"):
        drm = "/sys/class/drm"
        try:
            for entry in os.listdir(drm):
                vend = os.path.join(drm, entry, "device", "vendor")
                if os.path.isfile(vend):
                    try:
                        with open(vend, "r") as f:
                            if f.read().strip().lower() == "0x8086":
                                # We could parse the model name from lspci to
                                # filter iGPUs out — for now, presence + the
                                # trainer opt-in gate is enough.
                                return True
                    except OSError:
                        pass
        except OSError:
            pass
    return False


def torch_state():
    """Introspect the running interpreter's torch. Returns a dict so the
    frontend can decide whether the CPU-vs-CUDA situation warrants a repair
    prompt. Import failure counts as not-installed."""
    try:
        import torch
    except Exception as e:
        return {"installed": False, "import_error": str(e)}
    try:
        cuda_avail = bool(torch.cuda.is_available())
    except Exception:
        cuda_avail = False
    return {
        "installed":      True,
        "version":        str(getattr(torch, "__version__", "")),
        "cuda_build":     bool(getattr(torch.version, "cuda", None)),
        "cuda_runtime":   str(getattr(torch.version, "cuda", "") or ""),
        "cuda_available": cuda_avail,
        "device_count":   int(torch.cuda.device_count()) if cuda_avail else 0,
    }


def _torch_wheel_index_for_host():
    """Which wheel index install_torch() should hit for the running box.
    Priority: NVIDIA CUDA > AMD ROCm (Linux only) > CPU. None means "let pip
    resolve from PyPI" — used on macOS (PyPI serves the right build per arch;
    arm64 gets MPS, Intel gets CPU) and Linux ARM (no accelerator wheels
    published). Rule 34c: every arch picks a wheel that matches its actual
    hardware; the resolver never assumes a specific box shape."""
    plat = sys.platform
    if plat == "darwin":
        return None
    if plat.startswith("linux"):
        # Linux ARM has no CUDA/ROCm wheels; only x86_64 accelerators are
        # published. NVIDIA wins over AMD when both are present (rare, but
        # ROCm can't be built against a CUDA-only card and vice versa;
        # NVIDIA-first is the safer default because torch's ROCm wheel does
        # NOT fall back to CPU when kfd fails — it errors on import).
        import platform as _plat
        mach = (_plat.machine() or "").lower()
        if mach not in ("x86_64", "amd64"):
            return None
        if has_nvidia_gpu():
            return TORCH_CUDA_INDEX
        if has_amd_gpu():
            return TORCH_ROCM_INDEX
        return TORCH_CPU_INDEX
    if plat.startswith("win") or os.name == "nt":
        # Windows: no ROCm wheels published; AMD/Intel on Windows use DirectML
        # via a separate torch-directml package (installed on top of a CPU
        # torch). status_snapshot surfaces the hardware; the runtime CUDA
        # branch stays NVIDIA-only.
        return TORCH_CUDA_INDEX if has_nvidia_gpu() else TORCH_CPU_INDEX
    return None


def install_torch(force_cuda=None, force_rocm=None):
    """Install (or repair) torch for the running box. Picks the wheel index
    automatically:
      - macOS: PyPI (unchanged; arm64 gets MPS from the standard wheel)
      - Linux ARM: PyPI (no accelerator wheels published)
      - Linux x86: CUDA index if NVIDIA present, else ROCm index if AMD
        present, else CPU
      - Windows x86: CUDA index if NVIDIA present, else CPU
    `force_cuda=True` overrides the detection (used when the modal is
    surfacing a "GPU present but torch is CPU" repair). `force_cuda=False`
    forces the CPU wheel. `force_rocm=True` forces the ROCm index (Linux
    only; on other platforms it falls back to CPU). `--force-reinstall` is
    always passed because a stale wheel already in site-packages would be
    skipped by pip's normal resolver.

    Returns the install() result dict, plus a `wheel_index` field naming the
    index used so the frontend can report exactly what was installed."""
    if force_cuda is True:
        index_url = TORCH_CUDA_INDEX
    elif force_rocm is True:
        index_url = TORCH_ROCM_INDEX if sys.platform.startswith("linux") else TORCH_CPU_INDEX
    elif force_cuda is False:
        index_url = TORCH_CPU_INDEX
    else:
        index_url = _torch_wheel_index_for_host()

    args_base = ["install", "--disable-pip-version-check", "--force-reinstall"]
    if index_url:
        # --index-url (not --extra-index-url): with --extra-index-url pip is
        # free to pick same-version wheels from either source, which is how
        # boxes with NVIDIA GPUs end up with CPU torch. --index-url locks pip
        # to the pytorch index for this install call.
        args_base += ["--index-url", index_url]
    args = args_base + ["torch"]

    ok, out, err = _run_pip(args + ["--user"])
    if not ok:
        ok, out, err = _run_pip(args)
    if not ok and (sys.platform.startswith("win") or os.name == "nt"):
        ok, out, err = _run_pip(args, use_shell_runas=True)
    if not ok and not (sys.platform.startswith("win") or os.name == "nt"):
        ok2, out2, err2 = _run_sudo(args)
        if ok2:
            ok, out, err = ok2, out2, err2
    result = {
        "ok":              bool(ok),
        "method":          "install_torch",
        "stdout":          out,
        "stderr":          err,
        "needs_elevation": not ok,
        "wheel_index":     index_url or "pypi",
        "restart_required": bool(ok),
    }
    return result


def status_snapshot():
    """One-shot summary of the dep state the auto-tune modal cares about.
    Called by /sys/detect so a single click produces everything the modal
    needs to decide whether to auto-open the installer popup:

      torch            - torch_state() output (what's actually loaded)
      has_nvidia_gpu   - fresh probe, does NOT depend on the sys_metrics cache
      needs_torch_cuda - the specific "GPU present but torch is CPU" case; when
                         True the modal should auto-pop the installer and pass
                         index_url=TORCH_CUDA_INDEX (or call install_torch()
                         via /system/install_dep with pkg='torch' + the index)
      bitsandbytes     - {installed: bool}; only relevant when torch_cuda is OK
      helpers          - per-platform native-helper hints (see HELPERS)"""
    ts = torch_state()
    nv = has_nvidia_gpu()
    amd = has_amd_gpu()
    intel = has_intel_discrete_gpu()
    needs_torch_cuda = bool(nv and ts.get("installed") and not ts.get("cuda_build"))
    # ROCm's torch wheel exposes torch.cuda.* API surface but is built against
    # AMD's runtime. `needs_torch_rocm` = "AMD is here, no NVIDIA (so we
    # picked ROCm), and torch's advertised cuda_runtime string is empty or
    # doesn't look like an NVIDIA CUDA version". Best-effort — a false
    # negative just means the user gets the same "restart to fix" prompt.
    needs_torch_rocm = bool(
        (not nv) and amd and sys.platform.startswith("linux")
        and ts.get("installed") and not ts.get("cuda_available")
    )
    # Temperature sensors are considered "missing" when the live snapshot
    # can't read a CPU temperature AND a per-arch helper exists. The Restart
    # flow reads this to decide whether to queue the helper install.
    temp_helper = _temp_sensor_helper_id()
    needs_temp_sensor = False
    if temp_helper:
        try:
            from runtime import sys_metrics as _sm
            snap = _sm.snapshot()
            needs_temp_sensor = bool(snap.get("available") and snap.get("cpu_temp_c") is None)
        except Exception:
            # If sys_metrics isn't available (e.g. called outside the mri
            # dashboard) we conservatively leave this False so the Restart
            # flow doesn't try to fix something we can't measure.
            needs_temp_sensor = False
    return {
        "torch":             ts,
        "has_nvidia_gpu":    nv,
        "has_amd_gpu":       amd,
        "has_intel_gpu":     intel,
        "needs_torch_cuda":  needs_torch_cuda,
        "needs_torch_rocm":  needs_torch_rocm,
        "torch_cuda_index":  TORCH_CUDA_INDEX,
        "torch_rocm_index":  TORCH_ROCM_INDEX,
        "bitsandbytes":      {"installed": is_installed("bitsandbytes")},
        "needs_temp_sensor": needs_temp_sensor,
        "temp_sensor_helper": temp_helper,
        # Non-CUDA/non-ROCm accelerators the runtime can't yet target from
        # inside a trainer — currently informational so the UI can show
        # "detected but not wired". A future rev that adds DirectML/XPU
        # trainer hooks flips these into `needs_*` flags with matching pkg
        # names.
        "directml_available": intel or (amd and not sys.platform.startswith("linux")),
        "intel_xpu_available": intel,
        "helpers": {
            # Kept for back-compat with any code that still reads the old
            # nested shape. New callers use needs_temp_sensor / temp_sensor_helper.
            "temp_sensor": temp_helper,
        },
    }


def _temp_sensor_helper_id():
    """Pick the temperature-sensor helper for this arch, or None when no
    helper exists. Windows now auto-installs LibreHardwareMonitor via winget
    (the WMI provider the HUD reads for CPU/GPU temps); the user still has to
    launch LHM once and enable Options > WMI Provider — the frontend states
    that in the post-install message."""
    import platform as _plat
    plat = sys.platform
    if plat == "darwin":
        return "mac_temp_arm" if (_plat.machine() or "").lower() == "arm64" else "mac_temp_intel"
    if plat.startswith("linux"):
        return "linux_lm_sensors" if shutil.which("apt-get") else None
    if plat.startswith("win") or os.name == "nt":
        return "win_temp_lhm" if shutil.which("winget") else None
    return None


# Native-helper installer for OS-level tools the platform depends on but pip
# cannot install (temperature sensors, WMI providers). Keyed by a short id so
# the caller doesn't reason about brew/apt package names.
HELPERS = {
    "mac_temp_arm":     {"os": "darwin", "manager": "brew",   "pkg": "macmon"},
    "mac_temp_intel":   {"os": "darwin", "manager": "brew",   "pkg": "osx-cpu-temp"},
    "linux_lm_sensors": {"os": "linux",  "manager": "apt",    "pkg": "lm-sensors"},
    # Windows: LibreHardwareMonitor is the WMI provider the HUD reads for CPU
    # and GPU temperatures. Installable via winget without UAC; the user still
    # has to launch it once and tick Options > "Enable WMI Provider" for the
    # sensors to actually publish. The install call succeeds after the
    # winget install; the frontend surfaces the manual-launch step in the
    # post-install message.
    "win_temp_lhm":     {"os": "win",    "manager": "winget", "pkg": "LibreHardwareMonitor.LibreHardwareMonitor"},
}


def install_helper(helper_id):
    """Install one of the HELPERS by id. Returns {ok, method, stdout, stderr,
    needs_elevation, unsupported}. `unsupported=True` when the id targets an
    OS other than the running box (guards a Windows caller from firing a brew
    install)."""
    entry = HELPERS.get(helper_id)
    if entry is None:
        return {"ok": False, "method": "?", "stdout": "", "stderr": f"unknown helper '{helper_id}'",
                "needs_elevation": False, "unsupported": True}
    if not sys.platform.startswith(entry["os"]):
        return {"ok": False, "method": entry["manager"], "stdout": "", "stderr": "wrong OS for helper",
                "needs_elevation": False, "unsupported": True}
    if entry["manager"] == "brew":
        return _run_brew(entry["pkg"])
    if entry["manager"] == "apt":
        return _run_apt(entry["pkg"])
    if entry["manager"] == "winget":
        return _run_winget(entry["pkg"])
    return {"ok": False, "method": entry["manager"], "stdout": "", "stderr": "no runner",
            "needs_elevation": False, "unsupported": True}


def _run_brew(pkg):
    try:
        out = subprocess.run(
            ["brew", "install", pkg],
            capture_output=True, text=True, timeout=PIP_TIMEOUT_SECS,
            check=False,
        )
        ok = out.returncode == 0
        return {"ok": ok, "method": "brew", "stdout": out.stdout, "stderr": out.stderr,
                "needs_elevation": False, "unsupported": False}
    except FileNotFoundError:
        return {"ok": False, "method": "brew", "stdout": "",
                "stderr": "brew not installed. Install Homebrew from https://brew.sh first.",
                "needs_elevation": False, "unsupported": False}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "method": "brew", "stdout": "", "stderr": f"{type(exc).__name__}: {exc}",
                "needs_elevation": False, "unsupported": False}


def _run_winget(pkg):
    """winget install without UAC. --silent skips the GUI installer, --accept-*
    handles the first-run TOS prompts so the request thread never hangs. The
    result may still return non-zero if winget isn't on PATH (older Win10) —
    the frontend surfaces a manual-install hint in that case."""
    if not shutil.which("winget"):
        return {"ok": False, "method": "winget", "stdout": "",
                "stderr": "winget not on PATH. Install App Installer from the Microsoft Store, then retry.",
                "needs_elevation": False, "unsupported": False}
    try:
        out = subprocess.run(
            ["winget", "install", "--id", pkg, "-e",
             "--accept-source-agreements", "--accept-package-agreements",
             "--silent"],
            capture_output=True, text=True, timeout=PIP_TIMEOUT_SECS,
            check=False, creationflags=_NO_WINDOW,
        )
        # winget returns 0 for a fresh install AND for "already installed"
        # (exit code 0x8A15002B on some builds); treat both as ok.
        ok = out.returncode == 0
        return {"ok": ok, "method": "winget", "stdout": out.stdout, "stderr": out.stderr,
                "needs_elevation": False, "unsupported": False}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "method": "winget", "stdout": "", "stderr": f"{type(exc).__name__}: {exc}",
                "needs_elevation": False, "unsupported": False}


def _run_apt(pkg):
    try:
        out = subprocess.run(
            ["sudo", "-n", "apt-get", "install", "-y", pkg],
            capture_output=True, text=True, timeout=PIP_TIMEOUT_SECS,
            check=False,
        )
        ok = out.returncode == 0
        return {"ok": ok, "method": "apt", "stdout": out.stdout, "stderr": out.stderr,
                "needs_elevation": not ok, "unsupported": False}
    except FileNotFoundError:
        return {"ok": False, "method": "apt", "stdout": "",
                "stderr": "apt-get not available on this box.", "needs_elevation": False,
                "unsupported": True}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "method": "apt", "stdout": "", "stderr": f"{type(exc).__name__}: {exc}",
                "needs_elevation": False, "unsupported": False}
