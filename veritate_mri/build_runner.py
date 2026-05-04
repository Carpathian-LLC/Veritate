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
# veritate_mri/build_runner.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import subprocess
import sys
import threading
import time

import logs as logmod
from readers import paths

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
    """True if any .c/.h under veritate_engine/src/ is newer than the binary.
    Returns False if binary doesn't exist (let _refresh_present handle that case)
    or if src dir is missing. Guards against the silent stale-binary failure mode
    where source got updated but the binary wasn't rebuilt — Python parser ends
    up expecting a wire format the engine doesn't emit, causing pipe desync."""
    bin_path = _STATE["binary_path"]
    if not os.path.isfile(bin_path):
        return False
    src_dir = os.path.join(paths.ENGINE_ROOT, "src")
    if not os.path.isdir(src_dir):
        return False
    try:
        bin_mtime = os.path.getmtime(bin_path)
    except OSError:
        return False
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
    _set(status=STATUS_BUILDING, started_at=time.time(),
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
    logmod.warn("build", "stop requested — killing build subprocess")
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


def start():
    """Kick off a background build. Returns immediately. No-op if already building."""
    with _LOCK:
        if _STATE["status"] == STATUS_BUILDING:
            return state()
    if _refresh_present() and not _binary_is_stale():
        logmod.info("build", f"binary already present for {_STATE['os']}/{_STATE['arch']}: skipping build")
        _set(status=STATUS_OK)
        return state()
    if _refresh_present() and _binary_is_stale():
        logmod.warn("build", "binary present but source is newer — forcing rebuild to avoid wire-format mismatch")
    t = threading.Thread(target=_run_build, name="build-runner", daemon=True)
    t.start()
    return state()
