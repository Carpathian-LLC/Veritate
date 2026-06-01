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

import hashlib
import os
import platform
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
LAUNCH_PHASE_ENV = "VERITATE_LAUNCH_PHASE"   # set on re-exec; tells phase-2 to skip bootstrap
TIER_ENV         = "VERITATE_TIER"           # propagated to runtime so feature gates can read it
MINIMAL_ENV      = "VERITATE_MINIMAL"        # "1" => power-save dashboard (no brain, no analytics)

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
# was built against.
TIER_PYTHON_RANGE = {
    TIER_MAC_ARM:     ((3, 10), (3, 13)),
    TIER_MAC_INTEL:   ((3, 10), (3, 11)),  # torch 2.2 supports through 3.11
    TIER_LINUX_X86:   ((3, 10), (3, 13)),
    TIER_LINUX_ARM:   ((3, 10), (3, 13)),
    TIER_WINDOWS_X86: ((3, 10), (3, 13)),
}

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
    if tier == TIER_MAC_ARM:
        return f"brew install python@{pmaj}.{pmin} && /opt/homebrew/opt/python@{pmaj}.{pmin}/bin/python{pmaj}.{pmin} {os.path.abspath(__file__)}"
    if tier == TIER_MAC_INTEL:
        return f"brew install python@{pmaj}.{pmin} && /usr/local/opt/python@{pmaj}.{pmin}/bin/python{pmaj}.{pmin} {os.path.abspath(__file__)}"
    if tier == TIER_LINUX_X86 or tier == TIER_LINUX_ARM:
        return f"sudo apt install python{pmaj}.{pmin} python{pmaj}.{pmin}-venv  (or distro equivalent), then run with python{pmaj}.{pmin}"
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
    return HASH_SENTINEL.read_text(encoding="utf-8").strip() == _requirements_hash()


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
    if (cur.major, cur.minor) < py_min:
        sys.exit(
            f"[veritate] tier={tier}: Python {py_min[0]}.{py_min[1]}+ required, "
            f"got {cur.major}.{cur.minor}."
        )
    if (cur.major, cur.minor) > py_max:
        hint = _tier_install_hint(tier, py_max)
        sys.exit(
            f"[veritate] tier={tier}: Python {py_max[0]}.{py_max[1]} is the newest "
            f"supported on this hardware (you're on {cur.major}.{cur.minor}).\n"
            f"This tier's torch ceiling doesn't have wheels for newer Python.\n"
            f"To fix: {hint}"
        )
    print(f"[veritate] tier={tier} python={cur.major}.{cur.minor}")

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

    print("[veritate] installing python dependencies (first run can take several "
          "minutes — torch is ~2 GB) ...")
    py = str(_venv_python())
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    subprocess.check_call([py, "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    HASH_SENTINEL.write_text(_requirements_hash(), encoding="utf-8")
    print("[veritate] dependencies ready.")


def _reexec_under_venv() -> "int":
    """Hand off to the venv interpreter and return its exit code."""
    env = os.environ.copy()
    env[LAUNCH_PHASE_ENV] = "1"
    env[TIER_ENV] = _detect_tier()
    args = [str(_venv_python()), str(Path(__file__).resolve())] + sys.argv[1:]
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


def _wait_for_port_free(port: int, timeout: float = 10.0) -> bool:
    """Block until a fresh bind on (0.0.0.0, port) succeeds, or timeout.
    Used on relaunch to defeat the race where the parent's socket is still in
    TIME_WAIT when the child tries to start serving. connect-probes don't help
    here — TIME_WAIT blocks bind() but no listener is accepting connects."""
    import socket as _sock
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
        try:
            s.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", port))
            s.close()
            return True
        except OSError:
            try: s.close()
            except OSError: pass
            time.sleep(0.25)
    return False


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
    from readers   import paths       as paths_mod      # noqa: E402
    from runtime   import logs        as logmod         # noqa: E402
    from training  import build_runner                  # noqa: E402

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
        _open_browser_after_delay(f"http://localhost:{args.port}", BROWSER_DELAY_S)

    relaunch_cmd = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:]
    sys.argv = [sys.argv[0],
                "--model",   args.model,
                "--port",    str(args.port),
                "--threads", str(args.threads)]
    if args.step is not None:
        sys.argv += ["--step", str(args.step)]
    sys.argv += rest

    import app as mri_app  # noqa: E402
    mri_app.app.config["LAUNCH_CMD"] = relaunch_cmd
    if not _wait_for_port_free(args.port, timeout=10.0):
        logmod.error("veritate", f"port {args.port} still bound after 10s — aborting launch")
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
