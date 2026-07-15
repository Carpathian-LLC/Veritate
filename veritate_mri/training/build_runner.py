# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - background engine build manager. detects OS and arch, runs the matching script
#   under veritate_engine/build/, streams stdout/stderr into the in-memory log ring,
#   exposes status via state().
# - never blocks the flask server. start() spawns a thread.
# veritate_mri/training/build_runner.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import re
import subprocess
import sys
import threading
import time

from readers import paths
from runtime import logs as logmod

# ------------------------------------------------------------------------------------
# Constants

STATUS_IDLE     = "idle"
STATUS_BUILDING = "building"
STATUS_OK       = "ok"
STATUS_FAILED   = "failed"
STATUS_SKIPPED  = "skipped"

_LOCK = threading.Lock()
_PROC = None
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
_PRE_BUILD_HOOK = None
_STATE = {
    "status":      STATUS_IDLE,
    "os":          paths.current_os(),
    "arch":        paths.current_arch(),
    "binary_path": paths.engine_binary_path(),
    "binary_present": False,
    "started_at":  None,
    "finished_at": None,
    "exit_code":   None,
    "error":       None,
}

# ------------------------------------------------------------------------------------
# Functions

def _refresh_present():
    _STATE["binary_present"] = os.path.isfile(_STATE["binary_path"])
    return _STATE["binary_present"]


def _binary_is_stale():
    """True if any .c/.h/.S under the engine's src/ or kernels/ is newer than
    the binary. Returns False if binary doesn't exist (let _refresh_present
    handle that case) or if both source roots are missing. Guards against the
    silent stale-binary failure mode where source got updated but the binary
    wasn't rebuilt, Python parser ends up expecting a wire format the engine
    doesn't emit, causing pipe desync."""
    bin_path = _STATE["binary_path"]
    if not os.path.isfile(bin_path):
        return False
    roots = [
        os.path.join(paths.ENGINE_PRIMARY, "src"),
        os.path.join(paths.ENGINE_PRIMARY, "kernels"),
    ]
    roots = [r for r in roots if os.path.isdir(r)]
    if not roots:
        return False
    try:
        bin_mtime = os.path.getmtime(bin_path)
    except OSError:
        return False
    # The build script itself is part of the build inputs: editing build.bat/
    # build.sh (e.g. adding a kernel TU to the compile+link list) changes the
    # produced binary even when no .c/.h did. Without this, a stale binary from
    # before a script edit would be silently reused.
    try:
        if os.path.getmtime(paths.build_script_path()) > bin_mtime:
            return True
    except OSError:
        pass
    for src_dir in roots:
        for root, _dirs, files in os.walk(src_dir):
            for fn in files:
                if not (fn.endswith(".c") or fn.endswith(".h") or fn.endswith(".S")):
                    continue
                try:
                    if os.path.getmtime(os.path.join(root, fn)) > bin_mtime:
                        return True
                except OSError:
                    continue
    return False


# Matches a src/ or kernels/ .c path token in either build script, with `\` or
# `/` separators (build.bat uses backslashes, build.sh forward slashes).
_TU_REF_RE = re.compile(r"(?:src|kernels)[\\/][\w./\\-]*\.c")


def _referenced_shared_tus(script_path):
    """Parse a build script for the src/ and kernels/x86_64/ .c files it
    compiles, returned as a set of forward-slash relative paths. Returns None if
    the script can't be read. arm64/metal/scalar-only kernels are intentionally
    excluded so build.bat (x86_64-only) and build.sh (multi-arch) compare on the
    exact TU set they are both required to compile."""
    try:
        with open(script_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    out = set()
    for tok in _TU_REF_RE.findall(text):
        rel = tok.replace("\\", "/")
        if rel.startswith("src/") or rel.startswith("kernels/x86_64/"):
            out.add(rel)
    return out


def check_build_parity():
    """Warn when build.bat and build.sh disagree on the src/ + x86_64 kernel set
    they must both compile. This drift is exactly what produces an undefined-
    symbol link failure on one platform but not the other (a kernel TU added to
    one script but forgotten in the other). Non-fatal: the goal is to surface
    the cause in the build log *before* the linker's cryptic dump. Returns the
    (only_in_bat, only_in_sh) path lists for testability."""
    build_dir = os.path.join(paths.ENGINE_PRIMARY, "build")
    bat = _referenced_shared_tus(os.path.join(build_dir, "build.bat"))
    sh  = _referenced_shared_tus(os.path.join(build_dir, "build.sh"))
    if bat is None or sh is None:
        return [], []
    only_bat = sorted(bat - sh)
    only_sh  = sorted(sh - bat)
    if only_sh:
        logmod.warn("build",
                    "build.bat is missing x86_64 TUs that build.sh compiles: "
                    + ", ".join(only_sh)
                    + " — Windows link will fail with undefined symbols until added")
    if only_bat:
        logmod.warn("build",
                    "build.sh is missing x86_64 TUs that build.bat compiles: "
                    + ", ".join(only_bat))
    return only_bat, only_sh


def state():
    with _LOCK:
        _refresh_present()
        return dict(_STATE)


def _set(**kw):
    with _LOCK:
        _STATE.update(kw)
        _refresh_present()


def set_pre_build_hook(fn):
    """Register a callable invoked before the build subprocess starts. Used by
    app.py to close the C engine subprocess so the linker can replace the
    binary on Windows where running .exe files are locked."""
    global _PRE_BUILD_HOOK
    _PRE_BUILD_HOOK = fn


def _run_build():
    global _PROC
    script = paths.build_script_path()
    if not os.path.isfile(script):
        msg = f"build script missing: {script}"
        logmod.error("build", msg)
        _set(status=STATUS_FAILED, error=msg, finished_at=time.time())
        return
    if _PRE_BUILD_HOOK is not None:
        try:
            _PRE_BUILD_HOOK()
        except Exception as e:
            logmod.warn("build", f"pre-build hook failed: {e}")
    logmod.info("build", f"starting build for {_STATE['os']}/{_STATE['arch']}: {os.path.basename(script)}")
    # Surface build.bat/build.sh drift up front, so a missing kernel TU reads as
    # a clear warning here instead of an undefined-symbol dump from the linker.
    try:
        check_build_parity()
    except Exception as e:
        logmod.warn("build", f"parity check skipped: {e}")
    # start() already set STATUS_BUILDING synchronously; this is the worker
    # confirming the running state. left intentionally so the worker can be
    # invoked directly in tests without going through start().
    _set(status=STATUS_BUILDING, started_at=_STATE.get("started_at") or time.time(),
         finished_at=None, exit_code=None, error=None)
    try:
        if _STATE["os"] == paths.OS_WINDOWS:
            cmd = ["cmd", "/c", script]
        else:
            cmd = ["/bin/sh", script]
        proc = subprocess.Popen(
            cmd, cwd=paths.ENGINE_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            creationflags=_NO_WINDOW,
        )
        with _LOCK:
            _PROC = proc
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    logmod.info("build", line)
            proc.wait()
        finally:
            with _LOCK:
                _PROC = None
        ok = proc.returncode == 0 and _refresh_present()
        if ok:
            logmod.ok("build", f"build complete: {_STATE['binary_path']}")
            _set(status=STATUS_OK, exit_code=proc.returncode, finished_at=time.time())
        else:
            err = f"build failed: exit={proc.returncode} binary={'present' if _refresh_present() else 'missing'}"
            logmod.error("build", err)
            _set(status=STATUS_FAILED, exit_code=proc.returncode, finished_at=time.time(), error=err)
    except Exception as e:
        msg = f"build crashed: {e}"
        logmod.error("build", msg)
        _set(status=STATUS_FAILED, finished_at=time.time(), error=msg)


def stop():
    with _LOCK:
        proc = _PROC
    if proc is None:
        return False
    logmod.warn("build", "stop requested, killing build subprocess")
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(int(proc.pid)), "/T", "/F"],
                           capture_output=True, timeout=10, creationflags=_NO_WINDOW)
        else:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try: proc.kill()
            except Exception: pass
    except Exception as e:
        logmod.error("build", f"stop failed: {e}")
        return False
    return True


def start(force=False):
    """Kick off a background build. Returns immediately. No-op if already
    building. When force is False, a fresh-looking binary (present and not
    stale) short-circuits with STATUS_OK. When force is True (user clicked
    rebuild), we always run the build, even if the binary looks fresh.

    Sets STATUS_BUILDING synchronously before returning when an actual build
    will happen, so callers polling /engine/status immediately after the POST
    cannot observe a transient idle/ok state and skip the wait."""
    with _LOCK:
        already_building = _STATE["status"] == STATUS_BUILDING
    if already_building:
        return state()
    if not force and _refresh_present() and not _binary_is_stale():
        logmod.info("build", f"binary already present for {_STATE['os']}/{_STATE['arch']}: skipping build")
        _set(status=STATUS_OK)
        return state()
    if _refresh_present() and _binary_is_stale():
        logmod.warn("build", "binary present but source is newer, forcing rebuild to avoid wire-format mismatch")
    elif force:
        logmod.info("build", "user-triggered rebuild: forcing fresh compile")
    _set(status=STATUS_BUILDING, started_at=time.time(),
         finished_at=None, exit_code=None, error=None)
    t = threading.Thread(target=_run_build, name="build-runner", daemon=True)
    t.start()
    return state()
