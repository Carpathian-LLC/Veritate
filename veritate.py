# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Single-file installer + launcher. Invoked by start.bat (Windows),
#   start.command (macOS), or directly on any OS: `python veritate.py`.
# - Two phases driven by a sentinel env var. Top-level (system python): create
#   venv at ./venv if missing, install requirements.txt when its hash changes,
#   then re-exec self under the venv's interpreter. Launch phase (venv python):
#   import the MRI dashboard and serve it. The fast path — venv exists, hash
#   matches — is silent and re-execs in milliseconds.
# - Stdlib-only at the top because the system Python may not have any deps
#   installed yet.
# veritate.py
# ------------------------------------------------------------------------------------
# Imports

import glob
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import venv
import webbrowser
from pathlib import Path

# ------------------------------------------------------------------------------------
# Constants

HERE             = Path(__file__).resolve().parent
VENV_DIR         = HERE / "venv"
REQUIREMENTS     = HERE / "requirements.txt"
HASH_SENTINEL    = VENV_DIR / ".req_hash"
DEFAULT_PORT     = 8001
DEFAULT_THREADS  = 0   # 0 = auto: physical cores capped at 16. Was 8.
BROWSER_DELAY_S  = 3.0
PY_MIN           = (3, 10)
PY_MAX_TESTED    = (3, 13)
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_CPU_INDEX  = "https://download.pytorch.org/whl/cpu"
TORCH_ROCM_INDEX = "https://download.pytorch.org/whl/rocm6.2"
LAUNCH_PHASE_ENV = "VERITATE_LAUNCH_PHASE"   # set on re-exec; tells phase-2 to skip bootstrap
TIER_ENV         = "VERITATE_TIER"           # propagated to runtime so feature gates can read it
MINIMAL_ENV      = "VERITATE_MINIMAL"        # "1" => power-save dashboard (no brain, no analytics)
PY_REEXEC_ENV    = "VERITATE_PY_REEXEC"      # set after self-heal Python re-exec to detect loops

# Hardware tier labels. The Veritate mission requires runnability on older
# consumer hardware, so the launcher detects the host and dispatches per-tier
# dependency pins and runtime feature gates. Intel Mac specifically is capped
# at torch 2.2 because PyTorch dropped Intel macOS wheels at 2.3.
TIER_MAC_ARM     = "mac_arm"
TIER_MAC_INTEL   = "mac_intel"
TIER_LINUX_X86   = "linux_x86"
TIER_LINUX_ARM   = "linux_arm"
TIER_WINDOWS_X86 = "windows_x86"
TIER_UNSUPPORTED = "unsupported"

# (min_py, max_py) inclusive. max_py reflects what the tier's torch ceiling
# was built against. min_py reflects the floor of every pinned dep in
# requirements.txt for that tier — notably numpy 2.4 requires 3.11+, so every
# tier on modern torch is gated to 3.11. mac_intel stays at 3.10 because its
# torch 2.2 / numpy <2.0 line still supports it.
TIER_PYTHON_RANGE = {
    TIER_MAC_ARM:     ((3, 11), (3, 13)),
    TIER_MAC_INTEL:   ((3, 10), (3, 11)),
    TIER_LINUX_X86:   ((3, 11), (3, 13)),
    TIER_LINUX_ARM:   ((3, 11), (3, 13)),
    TIER_WINDOWS_X86: ((3, 11), (3, 13)),
}

# Homebrew installs under a different prefix per Mac arch. Both the manual
# install hint and the post-`brew install` probe resolve through this table.
BREW_PREFIX_BY_TIER = {
    TIER_MAC_ARM:   "/opt/homebrew",
    TIER_MAC_INTEL: "/usr/local",
}
BREW_PYTHON_TEMPLATE = "{prefix}/opt/python@{tag}/bin/python{tag}"

# On-disk locations where an interpreter for a given tier may already exist.
# Formatted with tag (e.g. "3.12"), ver_nodot ("312"), home, program_files,
# local_appdata. Rule 34c: every supported tier has an entry; a missing tier
# silently degrades to PATH-only lookup.
_LINUX_PYTHON_TEMPLATES = (
    "/usr/bin/python{tag}",
    "/usr/local/bin/python{tag}",
    "{home}/.local/bin/python{tag}",
)
PYTHON_PATH_TEMPLATES_BY_TIER = {
    TIER_MAC_ARM: (
        "/opt/homebrew/opt/python@{tag}/bin/python{tag}",
        "/opt/homebrew/bin/python{tag}",
        "/Library/Frameworks/Python.framework/Versions/{tag}/bin/python{tag}",
    ),
    TIER_MAC_INTEL: (
        "/usr/local/opt/python@{tag}/bin/python{tag}",
        "/usr/local/bin/python{tag}",
        "/Library/Frameworks/Python.framework/Versions/{tag}/bin/python{tag}",
    ),
    TIER_LINUX_X86:   _LINUX_PYTHON_TEMPLATES,
    TIER_LINUX_ARM:   _LINUX_PYTHON_TEMPLATES,
    TIER_WINDOWS_X86: (
        "{program_files}\\Python{ver_nodot}\\python.exe",
        "{local_appdata}\\Programs\\Python\\Python{ver_nodot}\\python.exe",
    ),
}
# Version-manager layouts, searched on every tier (a Windows box can hold a
# uv-managed cpython, a Mac can hold a pyenv build).
PYTHON_GLOB_TEMPLATES = (
    "{home}/.pyenv/versions/{tag}.*/bin/python{tag}",
    "{home}/.local/share/uv/python/cpython-{tag}.*/bin/python{tag}",
    "{home}/.local/share/uv/python/cpython-{tag}.*/python.exe",
    "{home}/Library/Application Support/uv/python/cpython-{tag}.*/bin/python{tag}",
    "{home}/.asdf/installs/python/{tag}.*/bin/python{tag}",
)
ENV_PROGRAM_FILES        = "ProgramFiles"
ENV_LOCALAPPDATA         = "LOCALAPPDATA"
WIN_PROGRAM_FILES_FALLBACK = r"C:\Program Files"
WIN_LOCALAPPDATA_FALLBACK  = r"~\AppData\Local"

UV_INSTALL_PS1   = "https://astral.sh/uv/install.ps1"
UV_INSTALL_SH    = "https://astral.sh/uv/install.sh"
UV_BIN_CANDIDATES = ("~/.local/bin/uv", "~/.cargo/bin/uv")

# GPU probe surface. Duplicated from veritate_core/plugin/deps.py because the
# bootstrap phase runs under the system python before any venv exists and must
# stay stdlib-only with no package imports.
DRM_SYSFS_ROOT     = "/sys/class/drm"
DRM_DEVICE_SUBDIR  = "device"
DRM_VENDOR_FILE    = "vendor"
PCI_VENDOR_NVIDIA  = "0x10de"
PCI_VENDOR_AMD     = "0x1002"
DEV_NVIDIA0        = "/dev/nvidia0"
DEV_NVIDIACTL      = "/dev/nvidiactl"
DEV_KFD            = "/dev/kfd"
LIBCUDA_SONAME     = "libcuda.so.1"
NVCUDA_DLL         = "nvcuda"
PS_VIDEO_CONTROLLER_QUERY = ("(Get-CimInstance Win32_VideoController | "
                             "Select-Object -ExpandProperty Name) -join '|'")
NVIDIA_NAME_TOKENS = ("nvidia", "geforce", "rtx", "gtx", "quadro", "tesla")

BIND_HOST             = "0.0.0.0"
DASHBOARD_URL_TEMPLATE = "http://localhost:{port}"
PORT_PROBE_TIMEOUT_S  = 0.5
PORT_WAIT_TIMEOUT_S   = 10.0
PORT_POLL_INTERVAL_S  = 0.25
PROC_TERM_GRACE_S     = 3.0
PROC_KILL_GRACE_S     = 2.0
CMDLINE_PREVIEW_CHARS = 80

INTERPRETER_PROBE_TIMEOUT_S = 15
PY_LAUNCHER_TIMEOUT_S       = 10
GPU_PROBE_TIMEOUT_S         = 10
POWERSHELL_TIMEOUT_S        = 8
TORCH_VERIFY_TIMEOUT_S      = 60

# ------------------------------------------------------------------------------------
# Bootstrap phase (runs under system python)

def _detect_tier() -> str:
    plat = sys.platform
    arch = (platform.machine() or "").lower()
    if plat == "darwin":
        return TIER_MAC_ARM if arch == "arm64" else TIER_MAC_INTEL
    if plat.startswith("linux"):
        return TIER_LINUX_X86 if arch in ("x86_64", "amd64") else TIER_LINUX_ARM
    if plat.startswith("win") or os.name == "nt":
        return TIER_WINDOWS_X86
    return TIER_UNSUPPORTED


def _tier_install_hint(tier: str, py_max: tuple) -> str:
    pmaj, pmin = py_max
    if tier in BREW_PREFIX_BY_TIER:
        tag = f"{pmaj}.{pmin}"
        py = BREW_PYTHON_TEMPLATE.format(prefix=BREW_PREFIX_BY_TIER[tier], tag=tag)
        return f"brew install python@{tag} && {py} {os.path.abspath(__file__)}"
    if tier in (TIER_LINUX_X86, TIER_LINUX_ARM):
        return (f"sudo apt install python{pmaj}.{pmin} python{pmaj}.{pmin}-venv  (or distro equivalent), "
                f"then run with python{pmaj}.{pmin}")
    if tier == TIER_WINDOWS_X86:
        return f"install Python {pmaj}.{pmin} from python.org and re-launch via start.bat"
    return ""


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _requirements_hash() -> str:
    if not REQUIREMENTS.exists():
        return ""
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def _deps_satisfied() -> bool:
    if not _venv_python().exists():
        return False
    if not HASH_SENTINEL.exists():
        return False
    if HASH_SENTINEL.read_text(encoding="utf-8").strip() != _requirements_hash():
        return False
    tier = _detect_tier()
    if tier == TIER_UNSUPPORTED:
        return False
    py_min, py_max = TIER_PYTHON_RANGE[tier]
    vver = _interpreter_version(str(_venv_python()))
    return vver is not None and py_min <= vver <= py_max


def _interpreter_version(py_path: str) -> "tuple | None":
    """Probe a Python interpreter for its (major, minor). None on failure."""
    try:
        out = subprocess.check_output(
            [py_path, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            text=True, stderr=subprocess.DEVNULL, timeout=INTERPRETER_PROBE_TIMEOUT_S,
        ).strip()
        maj_s, min_s = out.split(".", 1)
        return (int(maj_s), int(min_s))
    except Exception:
        return None


def _candidate_python_paths(tier: str, version: tuple) -> list:
    """Common on-disk locations where Python X.Y may already be installed."""
    maj, mn = version
    fields = {
        "tag":       f"{maj}.{mn}",
        "ver_nodot": f"{maj}{mn}",
        "home":      os.path.expanduser("~"),
        "program_files": os.environ.get(ENV_PROGRAM_FILES, WIN_PROGRAM_FILES_FALLBACK),
        "local_appdata": os.environ.get(ENV_LOCALAPPDATA,
                                        os.path.expanduser(WIN_LOCALAPPDATA_FALLBACK)),
    }
    paths = [t.format(**fields) for t in PYTHON_PATH_TEMPLATES_BY_TIER.get(tier, ())]
    # Cross-platform: pyenv, uv-managed, asdf
    for tmpl in PYTHON_GLOB_TEMPLATES:
        paths += sorted(glob.glob(tmpl.format(**fields)))
    return paths


def _find_existing_python(tier: str, py_min: tuple, py_max: tuple) -> "str | None":
    """Walk versions high→low; return the first interpreter in range that works."""
    if py_min[0] != py_max[0]:
        return None  # only handle a single major version range
    for minor in range(py_max[1], py_min[1] - 1, -1):
        version = (py_max[0], minor)
        tag = f"{version[0]}.{version[1]}"
        candidates: list = []
        path_hit = shutil.which(f"python{tag}")
        if path_hit:
            candidates.append(path_hit)
        candidates += _candidate_python_paths(tier, version)
        # Windows py launcher
        if tier == TIER_WINDOWS_X86 and shutil.which("py"):
            try:
                out = subprocess.check_output(
                    ["py", f"-{tag}", "-c", "import sys; print(sys.executable)"],
                    text=True, stderr=subprocess.DEVNULL, timeout=PY_LAUNCHER_TIMEOUT_S,
                ).strip()
                if out:
                    candidates.append(out)
            except Exception:
                pass
        for cand in candidates:
            if not cand or not os.path.exists(cand):
                continue
            ver = _interpreter_version(cand)
            if ver and py_min <= ver <= py_max:
                return cand
    return None


def _install_python_via_pkg_mgr(tier: str, py_target: tuple) -> "str | None":
    """Best-effort install via the platform's native package manager."""
    maj, mn = py_target
    tag = f"{maj}.{mn}"

    if tier in (TIER_MAC_ARM, TIER_MAC_INTEL):
        if not shutil.which("brew"):
            return None
        print(f"[veritate] installing python@{tag} via Homebrew (this may take a few minutes) ...")
        try:
            subprocess.check_call(["brew", "install", f"python@{tag}"])
        except subprocess.CalledProcessError:
            return None
        cand = BREW_PYTHON_TEMPLATE.format(prefix=BREW_PREFIX_BY_TIER[tier], tag=tag)
        if os.path.exists(cand):
            return cand
        return shutil.which(f"python{tag}")

    if tier == TIER_WINDOWS_X86:
        if not shutil.which("winget"):
            return None
        print(f"[veritate] installing Python.Python.{tag} via winget ...")
        try:
            subprocess.check_call([
                "winget", "install", "-e", "--silent",
                "--accept-source-agreements", "--accept-package-agreements",
                "--id", f"Python.Python.{tag}",
            ])
        except subprocess.CalledProcessError:
            return None
        return shutil.which(f"python{tag}") or _find_existing_python(tier, py_target, py_target)

    if tier in (TIER_LINUX_X86, TIER_LINUX_ARM):
        # sudo -n: don't prompt. If passwordless sudo isn't available we silently
        # fall through to the uv fallback instead of blocking the launcher.
        cmd: list | None = None
        if shutil.which("apt-get"):
            cmd = ["sudo", "-n", "apt-get", "install", "-y",
                   f"python{tag}", f"python{tag}-venv", f"python{tag}-dev"]
        elif shutil.which("dnf"):
            cmd = ["sudo", "-n", "dnf", "install", "-y", f"python{tag}"]
        elif shutil.which("yum"):
            cmd = ["sudo", "-n", "yum", "install", "-y", f"python{tag}"]
        elif shutil.which("pacman"):
            cmd = ["sudo", "-n", "pacman", "-S", "--noconfirm", "python"]
        if not cmd:
            return None
        print(f"[veritate] installing Python {tag} via {cmd[2]} ...")
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            return None
        return shutil.which(f"python{tag}")

    return None


def _install_python_via_uv(py_target: tuple) -> "str | None":
    """Cross-platform fallback. uv ships a portable CPython without needing root."""
    maj, mn = py_target
    tag = f"{maj}.{mn}"

    uv = shutil.which("uv")
    if not uv:
        print("[veritate] installing uv (portable Python manager) ...")
        try:
            if os.name == "nt":
                subprocess.check_call([
                    "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-Command", f"irm {UV_INSTALL_PS1} | iex",
                ])
            else:
                subprocess.check_call(
                    f"curl -LsSf {UV_INSTALL_SH} | sh",
                    shell=True,
                )
        except subprocess.CalledProcessError:
            return None
        for cand in [os.path.expanduser(p) for p in UV_BIN_CANDIDATES] + [shutil.which("uv")]:
            if cand and os.path.exists(cand):
                uv = cand
                break
        if not uv:
            return None

    print(f"[veritate] installing Python {tag} via uv ...")
    try:
        subprocess.check_call([uv, "python", "install", tag])
    except subprocess.CalledProcessError:
        return None
    try:
        out = subprocess.check_output([uv, "python", "find", tag], text=True).strip()
        if out and os.path.exists(out):
            return out
    except Exception:
        pass
    return None


def _self_heal_python(tier: str, py_min: tuple, py_max: tuple) -> "str | None":
    """Find or install a Python in [py_min, py_max]. None if nothing worked."""
    found = _find_existing_python(tier, py_min, py_max)
    if found:
        print(f"[veritate] found compatible interpreter: {found}")
        return found
    print(f"[veritate] no Python {py_min[0]}.{py_min[1]}–{py_max[0]}.{py_max[1]} found; "
          f"attempting auto-install ...")
    installed = _install_python_via_pkg_mgr(tier, py_max)
    if installed:
        ver = _interpreter_version(installed)
        if ver and py_min <= ver <= py_max:
            return installed
    print("[veritate] native package manager unavailable or failed; falling back to uv ...")
    installed = _install_python_via_uv(py_max)
    if installed:
        ver = _interpreter_version(installed)
        if ver and py_min <= ver <= py_max:
            return installed
    return None


def _drm_vendor_present(vendor_id: str) -> bool:
    """True when any Linux DRM device reports PCI `vendor_id`."""
    try:
        entries = os.listdir(DRM_SYSFS_ROOT)
    except OSError:
        return False
    for entry in entries:
        vend = os.path.join(DRM_SYSFS_ROOT, entry, DRM_DEVICE_SUBDIR, DRM_VENDOR_FILE)
        if not os.path.isfile(vend):
            continue
        try:
            with open(vend) as f:
                if f.read().strip().lower() == vendor_id:
                    return True
        except OSError:
            pass
    return False


def _has_nvidia_gpu() -> bool:
    """True when a usable NVIDIA GPU is present. Cross-checks several signals so
    a machine with a working GPU + driver but a missing `nvidia-smi` on PATH
    (common on Windows installs that skip the CLI) still gets the CUDA build.

    Order (short-circuits on first hit):
      1. `nvidia-smi -L` listing a device (ground truth when reachable).
      2. Windows: `nvcuda.dll` loadable via ctypes (driver is installed).
      3. Windows: `Win32_VideoController` names any NVIDIA / GeForce / RTX / GTX
         adapter (WMI works even when the CLI is not on PATH).
      4. Linux: `/dev/nvidia0` present, or `libcuda.so.1` loadable, or
         `/sys/class/drm/card*` contains an NVIDIA vendor id (0x10de)."""
    # (1) nvidia-smi is the cheapest and clearest signal.
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL,
                timeout=GPU_PROBE_TIMEOUT_S,
            )
            if "GPU" in out:
                return True
        except Exception:
            pass
    plat = sys.platform
    # (2, 3) Windows: driver DLL + WMI.
    if plat.startswith("win") or os.name == "nt":
        try:
            import ctypes
            try:
                ctypes.WinDLL(NVCUDA_DLL)
                return True
            except OSError:
                pass
        except Exception:
            pass
        ps = shutil.which("pwsh") or shutil.which("powershell")
        if ps:
            try:
                out = subprocess.check_output(
                    [ps, "-NoProfile", "-Command", PS_VIDEO_CONTROLLER_QUERY],
                    text=True, stderr=subprocess.DEVNULL, timeout=POWERSHELL_TIMEOUT_S,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                low = out.lower()
                if any(tok in low for tok in NVIDIA_NAME_TOKENS):
                    return True
            except Exception:
                pass
        return False
    # (4) Linux: several corroborating signals; any one is sufficient.
    if plat.startswith("linux"):
        if os.path.exists(DEV_NVIDIA0) or os.path.exists(DEV_NVIDIACTL):
            return True
        try:
            import ctypes
            try:
                ctypes.CDLL(LIBCUDA_SONAME)
                return True
            except OSError:
                pass
        except Exception:
            pass
        return _drm_vendor_present(PCI_VENDOR_NVIDIA)
    return False


def _has_amd_gpu() -> bool:
    """True when a ROCm-capable AMD GPU is present on Linux. ROCm exists on
    Linux only — macOS never had ROCm, Windows uses DirectML which is a
    separate package installed on top of a CPU torch. Signals: `/dev/kfd`,
    `rocm-smi`/`rocminfo` on PATH, or PCI vendor 0x1002 in
    `/sys/class/drm/*/device/vendor`."""
    if not sys.platform.startswith("linux"):
        return False
    if os.path.exists(DEV_KFD):
        return True
    if shutil.which("rocm-smi") or shutil.which("rocminfo"):
        return True
    return _drm_vendor_present(PCI_VENDOR_AMD)


def _torch_index_url(tier: str) -> "str | None":
    """Wheel index for torch. Priority per tier:
      - macOS: PyPI (arm64 -> MPS, Intel -> CPU; PyTorch stops publishing x86
        Mac wheels above 2.2 which requirements.txt already pins)
      - Linux x86: NVIDIA CUDA > AMD ROCm > CPU
      - Linux ARM: PyPI (no accelerator wheels published)
      - Windows x86: NVIDIA CUDA > CPU (no ROCm on Windows)
    Rule 34c: every arch picks a wheel that matches its real hardware."""
    if tier in (TIER_MAC_ARM, TIER_MAC_INTEL):
        return None
    if tier == TIER_LINUX_ARM:
        return None
    if _has_nvidia_gpu():
        return TORCH_CUDA_INDEX
    if tier == TIER_LINUX_X86 and _has_amd_gpu():
        return TORCH_ROCM_INDEX
    return TORCH_CPU_INDEX


def _verify_torch_cuda(py: str) -> bool:
    """Run a subprocess to check the freshly-installed torch actually links
    against CUDA. Subprocess because our own interpreter may not have torch on
    sys.path yet (or may be caching an old copy)."""
    try:
        out = subprocess.check_output(
            [py, "-c",
             "import torch,sys; "
             "print(int(bool(torch.cuda.is_available()) and (torch.version.cuda or '') != ''))"],
            text=True, stderr=subprocess.DEVNULL, timeout=TORCH_VERIFY_TIMEOUT_S,
        ).strip()
        return out.endswith("1")
    except Exception:
        return False


def _ensure_venv_and_deps() -> None:
    """Idempotent. Silent when nothing needs doing."""
    if _deps_satisfied():
        return

    tier = _detect_tier()
    if tier == TIER_UNSUPPORTED:
        sys.exit(
            f"[veritate] unsupported platform: sys.platform={sys.version_info!r} "
            f"machine={platform.machine()!r}. Supported tiers: macOS arm64/x86_64, "
            f"Linux x86_64/arm64, Windows x86_64."
        )

    py_min, py_max = TIER_PYTHON_RANGE[tier]
    cur = sys.version_info
    cur_ver = (cur.major, cur.minor)

    if cur_ver < py_min or cur_ver > py_max:
        # Self-heal: locate or install a compatible interpreter, then re-exec.
        # The PY_REEXEC sentinel guards against an infinite loop if the installer
        # claims success but the new interpreter still doesn't match.
        if os.environ.get(PY_REEXEC_ENV) == "1":
            sys.exit(
                f"[veritate] self-heal already ran but the interpreter is still "
                f"Python {cur.major}.{cur.minor} (need {py_min[0]}.{py_min[1]}–"
                f"{py_max[0]}.{py_max[1]}).\n"
                f"Manual fix: {_tier_install_hint(tier, py_max)}"
            )
        print(f"[veritate] tier={tier}: current Python {cur.major}.{cur.minor} outside "
              f"supported range {py_min[0]}.{py_min[1]}–{py_max[0]}.{py_max[1]}; "
              f"attempting self-heal ...")
        better = _self_heal_python(tier, py_min, py_max)
        if not better:
            sys.exit(
                f"[veritate] could not locate or auto-install a compatible Python.\n"
                f"Manual fix: {_tier_install_hint(tier, py_max)}"
            )
        print(f"[veritate] re-executing under {better}")
        env = os.environ.copy()
        env[PY_REEXEC_ENV] = "1"
        argv = [better, str(Path(__file__).resolve()), *sys.argv[1:]]
        try:
            os.execve(better, argv, env)
        except OSError:
            # Some shells (e.g. cmd.exe with .exe handlers) prefer spawn over exec.
            rc = subprocess.call(argv, env=env)
            sys.exit(rc)

    print(f"[veritate] tier={tier} python={cur.major}.{cur.minor}")

    # If a venv exists but was built with an interpreter no longer in range
    # (e.g. system upgrade rendered it stale), rebuild it from scratch.
    if _venv_python().exists():
        vver = _interpreter_version(str(_venv_python()))
        if vver is None or vver < py_min or vver > py_max:
            print(f"[veritate] existing venv Python {vver} out of supported range; rebuilding ...")
            shutil.rmtree(VENV_DIR, ignore_errors=True)

    if not _venv_python().exists():
        print(f"[veritate] creating virtual environment at {VENV_DIR} ...")
        try:
            venv.create(str(VENV_DIR), with_pip=True, clear=False, upgrade_deps=False)
        except Exception as e:
            sys.exit(
                f"[veritate] failed to create venv: {e}\n"
                f"On Debian/Ubuntu you may need: sudo apt install python3-venv"
            )
        if not _venv_python().exists():
            sys.exit(f"[veritate] venv created but interpreter missing at {_venv_python()}")

    if not REQUIREMENTS.exists():
        return

    py = str(_venv_python())
    index = _torch_index_url(tier)
    if index == TORCH_CUDA_INDEX:
        build = "CUDA"
    elif index == TORCH_ROCM_INDEX:
        build = "ROCm"
    elif index == TORCH_CPU_INDEX:
        build = "CPU"
    else:
        build = "PyPI (macOS/Linux-ARM native)"
    print(f"[veritate] installing python dependencies ({build} torch build; "
          f"first run can take several minutes) ...")
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    _install_torch_then_rest(py, index)
    HASH_SENTINEL.write_text(_requirements_hash(), encoding="utf-8")
    print("[veritate] dependencies ready.")


def _install_torch_then_rest(py: str, index: "str | None") -> None:
    """Two-phase install to defeat the classic PyPI/pytorch-index resolver mixup.

    Phase 1: install `torch` alone from the pytorch wheel index using
    `--index-url` (NOT `--extra-index-url`). This locks pip to that index only
    for torch so the resolver can't pick the same-versioned CPU wheel from PyPI.
    Skipped for macOS (rides PyPI cleanly).

    Phase 2: install the rest of requirements.txt normally from PyPI. Torch is
    already satisfied, so pip won't try to re-resolve it — the wheel we just
    installed stays put.

    Phase 3 (CUDA path only): verify `torch.cuda.is_available()` in a subprocess.
    If it comes back False we hit an edge (mirror mismatch, host without a
    working driver despite an adapter) — we log clearly instead of silently
    handing the user a CPU-only stack. This is what the auto-repair path in
    veritate_mri/runtime/deps.py can act on later."""
    # macOS: no wheel-index games; PyPI serves the right build per arch.
    if index is None:
        subprocess.check_call([py, "-m", "pip", "install", "-r", str(REQUIREMENTS)])
        return

    # Phase 1: torch from the correct index, ALONE. --index-url (not --extra-)
    # is the key: with --extra-index-url pip is free to pick same-version wheels
    # from either source, and on Windows/Linux the PyPI torch wheel is CPU-only.
    if index == TORCH_CUDA_INDEX:
        build = "CUDA"
    elif index == TORCH_ROCM_INDEX:
        build = "ROCm"
    else:
        build = "CPU"
    print(f"[veritate] phase 1/2: installing torch ({build}) from {index} ...")
    subprocess.check_call(
        [py, "-m", "pip", "install", "--index-url", index, "torch"],
    )

    # Phase 2: everything else from PyPI. Torch is already satisfied.
    print("[veritate] phase 2/2: installing remaining dependencies from PyPI ...")
    subprocess.check_call([py, "-m", "pip", "install", "-r", str(REQUIREMENTS)])

    # Phase 3: verify the accelerator path actually works. Both CUDA and ROCm
    # wheels expose `torch.cuda.is_available()` (ROCm's PyTorch presents its
    # HIP runtime through the CUDA API), so the same check applies.
    if index in (TORCH_CUDA_INDEX, TORCH_ROCM_INDEX):
        label = "CUDA" if index == TORCH_CUDA_INDEX else "ROCm"
        if _verify_torch_cuda(py):
            print(f"[veritate] verified: torch.cuda.is_available() == True ({label})")
        else:
            hint = ("NVIDIA driver too old for this CUDA runtime" if label == "CUDA"
                    else "AMD ROCm kfd device or driver missing (`/dev/kfd`, `rocminfo`)")
            print(f"[veritate] WARNING: {label} torch was installed but "
                  f"torch.cuda.is_available() returned False. "
                  f"Common causes: {hint}, or a stale CPU torch shadowing the venv wheel. "
                  "The dashboard's Restart button can retry this install; manual repair: "
                  f"'{py}' -m pip install --force-reinstall --index-url {index} torch")


def _reexec_under_venv() -> "int":
    """Hand off to the venv interpreter and return its exit code."""
    env = os.environ.copy()
    env[LAUNCH_PHASE_ENV] = "1"
    env[TIER_ENV] = _detect_tier()
    args = [str(_venv_python()), str(Path(__file__).resolve()), *sys.argv[1:]]
    try:
        return subprocess.call(args, env=env, cwd=str(HERE))
    except KeyboardInterrupt:
        return 0

# ------------------------------------------------------------------------------------
# Launch phase (runs under the venv's python)

def _open_browser_after_delay(url: str, delay: float) -> None:
    def _go() -> None:
        time.sleep(delay)
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()


def _wait_for_port_free(port: int, timeout: float = PORT_WAIT_TIMEOUT_S) -> bool:
    """Block until a fresh bind on (BIND_HOST, port) succeeds, or timeout.
    Used on relaunch to defeat the race where the parent's socket is still in
    TIME_WAIT when the child tries to start serving. connect-probes don't help
    here — TIME_WAIT blocks bind() but no listener is accepting connects."""
    import socket as _sock
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
        try:
            s.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
            s.bind((BIND_HOST, port))
            s.close()
            return True
        except OSError:
            try: s.close()
            except OSError: pass
            time.sleep(PORT_POLL_INTERVAL_S)
    return False


def _reclaim_orphan_on_port(port: int) -> int:
    """Terminate any Veritate-owned process listening on `port`. "Ours" means
    the process's cmdline or cwd references this repo's path — anything else
    is left alone with a printed warning so the user can decide. Tries SIGTERM
    first, escalates to SIGKILL after 3s. Returns the number reclaimed."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return 0
    here_str = str(HERE)
    reclaimed = 0
    for proc in psutil.process_iter(attrs=["pid", "cmdline"]):
        try:
            conns = proc.net_connections(kind="inet")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if not any(c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN
                   for c in conns):
            continue
        pid     = proc.info["pid"]
        cmdline = proc.info.get("cmdline") or []
        cmd     = " ".join(cmdline)
        try:
            cwd = proc.cwd()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            cwd = ""
        is_ours = (here_str in cmd) or (cwd == here_str)
        if not is_ours:
            print(f"[veritate] port {port} held by foreign PID {pid} "
                  f"({cmd[:CMDLINE_PREVIEW_CHARS]}); not killing", flush=True)
            continue
        print(f"[veritate] reclaiming port {port} from orphan Veritate PID {pid}",
              flush=True)
        try:
            proc.terminate()
            try:
                proc.wait(timeout=PROC_TERM_GRACE_S)
            except psutil.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=PROC_KILL_GRACE_S)
            reclaimed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"[veritate] could not kill PID {pid}: {e}", flush=True)
    return reclaimed


def _parse_launch_args():
    import argparse
    ap = argparse.ArgumentParser(
        prog="veritate",
        description="Veritate dashboard launcher (installer + run, all-in-one).",
    )
    ap.add_argument("--port",       type=int, default=DEFAULT_PORT)
    ap.add_argument("--threads",    type=int, default=DEFAULT_THREADS,
                    help="pytorch CPU threads. 0 = auto: physical cores capped at 16.")
    ap.add_argument("--model",      default="auto")
    ap.add_argument("--step",       type=int, default=None)
    ap.add_argument("--skip-build", action="store_true",
                    help="do not auto-build the engine. dashboard still serves PyTorch.")
    ap.add_argument("--no-browser", action="store_true",
                    help="do not auto-open the dashboard URL in a web browser.")
    ap.add_argument("--minimal", action="store_true",
                    help="power-save mode: dashboard reads/serves training state only. "
                         "Skips pytorch brain eager-load, idle watcher, heartbeat/analytics, "
                         "platform sync, and sys-metrics warm. ~10 GB lighter; "
                         "inference/atlas/teacher routes are inert until a full restart.")
    return ap.parse_known_args()


def _launch_dashboard() -> int:
    args, rest = _parse_launch_args()

    if args.minimal:
        os.environ[MINIMAL_ENV] = "1"
    else:
        os.environ.pop(MINIMAL_ENV, None)

    sys.path.insert(0, str(HERE / "veritate_mri"))
    from readers import paths as paths_mod
    from runtime import logs as logmod
    from training import build_runner

    logmod.info("veritate", f"detected {paths_mod.current_os()}/{paths_mod.current_arch()}")
    logmod.info("veritate", f"engine binary path: {paths_mod.engine_binary_path()}")

    if args.skip_build:
        logmod.info("veritate", "build skipped (--skip-build)")
    else:
        build_runner.start()

    if args.minimal:
        logmod.info("veritate", "MINIMAL mode: brain/sync/sys-warm disabled; "
                                "heartbeat stays active; training read-only views remain available.")

    if not args.no_browser:
        _open_browser_after_delay(DASHBOARD_URL_TEMPLATE.format(port=args.port),
                                  BROWSER_DELAY_S)

    relaunch_cmd = [sys.executable, os.path.abspath(__file__), *sys.argv[1:]]
    sys.argv = [sys.argv[0],
                "--model",   args.model,
                "--port",    str(args.port),
                "--threads", str(args.threads)]
    if args.step is not None:
        sys.argv += ["--step", str(args.step)]
    sys.argv += rest

    import app as mri_app
    mri_app.app.config["LAUNCH_CMD"] = relaunch_cmd
    if not _wait_for_port_free(args.port, timeout=PORT_PROBE_TIMEOUT_S):
        _reclaim_orphan_on_port(args.port)
        if not _wait_for_port_free(args.port, timeout=PORT_WAIT_TIMEOUT_S):
            msg = f"port {args.port} still bound after reclaim attempt — aborting launch"
            logmod.error("veritate", msg)
            print(f"[veritate] {msg}", flush=True)
            return 3
    mri_app.main()
    return 0

# ------------------------------------------------------------------------------------
# Entry

def main() -> int:
    if os.environ.get(LAUNCH_PHASE_ENV) == "1":
        return _launch_dashboard()
    _ensure_venv_and_deps()
    return _reexec_under_venv()


if __name__ == "__main__":
    sys.exit(main())
