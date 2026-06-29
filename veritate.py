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
            text=True, stderr=subprocess.DEVNULL, timeout=15,
        ).strip()
        maj_s, min_s = out.split(".", 1)
        return (int(maj_s), int(min_s))
    except Exception:
        return None


def _candidate_python_paths(tier: str, version: tuple) -> list:
    """Common on-disk locations where Python X.Y may already be installed."""
    maj, mn = version
    tag = f"{maj}.{mn}"
    home = os.path.expanduser("~")
    paths: list = []

    if tier == TIER_MAC_ARM:
        paths += [
            f"/opt/homebrew/opt/python@{tag}/bin/python{tag}",
            f"/opt/homebrew/bin/python{tag}",
            f"/Library/Frameworks/Python.framework/Versions/{tag}/bin/python{tag}",
        ]
    elif tier == TIER_MAC_INTEL:
        paths += [
            f"/usr/local/opt/python@{tag}/bin/python{tag}",
            f"/usr/local/bin/python{tag}",
            f"/Library/Frameworks/Python.framework/Versions/{tag}/bin/python{tag}",
        ]
    elif tier in (TIER_LINUX_X86, TIER_LINUX_ARM):
        paths += [
            f"/usr/bin/python{tag}",
            f"/usr/local/bin/python{tag}",
            f"{home}/.local/bin/python{tag}",
        ]
    elif tier == TIER_WINDOWS_X86:
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        local_appdata = os.environ.get("LOCALAPPDATA", os.path.expanduser(r"~\AppData\Local"))
        ver_nodot = f"{maj}{mn}"
        paths += [
            f"{program_files}\\Python{ver_nodot}\\python.exe",
            f"{local_appdata}\\Programs\\Python\\Python{ver_nodot}\\python.exe",
        ]

    # Cross-platform: pyenv, uv-managed, asdf
    paths += sorted(glob.glob(f"{home}/.pyenv/versions/{tag}.*/bin/python{tag}"))
    paths += sorted(glob.glob(f"{home}/.local/share/uv/python/cpython-{tag}.*/bin/python{tag}"))
    paths += sorted(glob.glob(f"{home}/.local/share/uv/python/cpython-{tag}.*/python.exe"))
    paths += sorted(glob.glob(f"{home}/Library/Application Support/uv/python/cpython-{tag}.*/bin/python{tag}"))
    paths += sorted(glob.glob(f"{home}/.asdf/installs/python/{tag}.*/bin/python{tag}"))
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
                    text=True, stderr=subprocess.DEVNULL, timeout=10,
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
        prefix = "/opt/homebrew" if tier == TIER_MAC_ARM else "/usr/local"
        cand = f"{prefix}/opt/python@{tag}/bin/python{tag}"
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
        cmd: "list | None" = None
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
                    "-Command", "irm https://astral.sh/uv/install.ps1 | iex",
                ])
            else:
                subprocess.check_call(
                    "curl -LsSf https://astral.sh/uv/install.sh | sh",
                    shell=True,
                )
        except subprocess.CalledProcessError:
            return None
        for cand in [
            os.path.expanduser("~/.local/bin/uv"),
            os.path.expanduser("~/.cargo/bin/uv"),
            shutil.which("uv"),
        ]:
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
        argv = [better, str(Path(__file__).resolve())] + sys.argv[1:]
        try:
            os.execve(better, argv, env)
        except OSError as e:
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
                  f"({cmd[:80]}); not killing", flush=True)
            continue
        print(f"[veritate] reclaiming port {port} from orphan Veritate PID {pid}",
              flush=True)
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except psutil.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
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
    if not _wait_for_port_free(args.port, timeout=0.5):
        _reclaim_orphan_on_port(args.port)
        if not _wait_for_port_free(args.port, timeout=10.0):
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
