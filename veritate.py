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
DEFAULT_THREADS  = 8
BROWSER_DELAY_S  = 3.0
PY_MIN           = (3, 10)
LAUNCH_PHASE_ENV = "VERITATE_LAUNCH_PHASE"   # set on re-exec; tells phase-2 to skip bootstrap

# ------------------------------------------------------------------------------------
# Bootstrap phase (runs under system python)

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

    if sys.version_info < PY_MIN:
        sys.exit(
            f"[veritate] Python {PY_MIN[0]}.{PY_MIN[1]}+ required, but this "
            f"interpreter is {sys.version_info.major}.{sys.version_info.minor}."
        )

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
    return ap.parse_known_args()


def _launch_dashboard() -> int:
    args, rest = _parse_launch_args()

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
