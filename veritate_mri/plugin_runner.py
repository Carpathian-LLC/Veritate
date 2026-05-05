# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - spawn a plugin script as a subprocess. one at a time globally.
# - stdout streams into the in-memory log ring. status exposed via state().
# veritate_mri/plugin_runner.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import signal
import subprocess
import sys
import threading
import time

import logs as logmod
from readers import paths, plugins as plugins_reader

# ------------------------------------------------------------------------------------
# Constants

STATUS_IDLE    = "idle"
STATUS_RUNNING = "running"
STATUS_OK      = "ok"
STATUS_FAILED  = "failed"
STATUS_STOPPED = "stopped"

PID_FILE = os.path.join(paths.REPO_ROOT, ".plugin_pid.json")
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

PYTORCH_ALLOC_ENV_KEY     = "PYTORCH_CUDA_ALLOC_CONF"
PYTORCH_ALLOC_ENV_DEFAULT = "expandable_segments:True"

_LOCK = threading.Lock()
_STATE = {
    "status":      STATUS_IDLE,
    "plugin_id":   None,
    "args":        None,
    "started_at":  None,
    "finished_at": None,
    "exit_code":   None,
}
_PROC = None
_RECOVERED_PID = None  # set when state was restored from PID_FILE after a server restart


# ------------------------------------------------------------------------------------
# PID-file persistence — lets the dashboard re-discover an in-flight training run
# after a server restart. The training subprocess survives parent exit on Windows;
# we just need to remember its PID so the Stop button can still reach it.

def _write_pid_file(plugin_id, pid, args, cmd_marker):
    try:
        with open(PID_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "plugin_id":   plugin_id,
                "pid":         int(pid),
                "started_at":  time.time(),
                "args":        args,
                "cmd_marker":  cmd_marker,
            }, f)
    except Exception as e:
        logmod.warn("plugin", f"could not write {PID_FILE}: {e}")


def _clear_pid_file():
    try:
        if os.path.isfile(PID_FILE):
            os.remove(PID_FILE)
    except Exception:
        pass


def _read_pid_file():
    if not os.path.isfile(PID_FILE):
        return None
    try:
        with open(PID_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _process_alive(pid, cmd_marker=None):
    """True iff a process with this pid exists and (if cmd_marker given) its
    command line contains the marker. The cmd_marker check guards against
    PID reuse — the OS may have recycled the PID to an unrelated process."""
    try:
        if os.name == "nt":
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5, creationflags=_NO_WINDOW,
            )
            if f'"{pid}"' not in r.stdout:
                return False
            if cmd_marker:
                # verify command line contains the marker via WMIC-like fallback
                try:
                    r2 = subprocess.run(
                        ["wmic", "process", "where", f"ProcessId={int(pid)}",
                         "get", "CommandLine", "/format:list"],
                        capture_output=True, text=True, timeout=5, creationflags=_NO_WINDOW,
                    )
                    return cmd_marker in r2.stdout
                except Exception:
                    return True  # tasklist confirmed; can't verify cmdline → trust it
            return True
        else:
            os.kill(int(pid), 0)
            if cmd_marker:
                try:
                    with open(f"/proc/{int(pid)}/cmdline", "rb") as f:
                        cmdline = f.read().decode("utf-8", "replace")
                    return cmd_marker in cmdline
                except Exception:
                    return True
            return True
    except Exception:
        return False


def _process_terminate(pid):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                           capture_output=True, timeout=10, creationflags=_NO_WINDOW)
        else:
            os.kill(int(pid), signal.SIGTERM)
    except Exception as e:
        logmod.warn("plugin", f"terminate pid {pid} failed: {e}")


def _recover_from_disk():
    """Called once at module load. If PID_FILE exists and the process is still
    alive, restore _STATE so the dashboard sees the run. The Popen handle is
    gone (parent died), but stop() can still kill via OS-level taskkill."""
    global _RECOVERED_PID
    rec = _read_pid_file()
    if rec is None:
        return
    pid = rec.get("pid")
    if not pid or not _process_alive(pid, cmd_marker=rec.get("cmd_marker")):
        _clear_pid_file()
        return
    _RECOVERED_PID = int(pid)
    with _LOCK:
        _STATE.update({
            "status":      STATUS_RUNNING,
            "plugin_id":   rec.get("plugin_id"),
            "args":        rec.get("args"),
            "started_at":  rec.get("started_at"),
            "finished_at": None,
            "exit_code":   None,
        })
    logmod.ok("plugin", f"recovered in-flight run: {rec.get('plugin_id')} pid={pid}")

# ------------------------------------------------------------------------------------
# Functions

def state():
    with _LOCK:
        return dict(_STATE)


def is_running():
    with _LOCK:
        return _STATE["status"] == STATUS_RUNNING


def _set(**kw):
    with _LOCK:
        _STATE.update(kw)


def _build_argv(plugin, args):
    out = [sys.executable, plugin["path"]]
    for k, v in (args or {}).items():
        if v is None or v == "":
            continue
        if isinstance(v, bool):
            if v: out.append(f"--{k}")
        else:
            out += [f"--{k}", str(v)]
    return out


def _run(plugin, args):
    global _PROC
    argv = _build_argv(plugin, args)
    logmod.info("plugin", f"start {plugin['id']}: {' '.join(argv[1:])}")
    _set(status=STATUS_RUNNING, plugin_id=plugin["id"], args=args,
         started_at=time.time(), finished_at=None, exit_code=None)
    env = os.environ.copy()
    env.setdefault(PYTORCH_ALLOC_ENV_KEY, PYTORCH_ALLOC_ENV_DEFAULT)
    try:
        proc = subprocess.Popen(argv, cwd=paths.REPO_ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1,
                                creationflags=_NO_WINDOW,
                                env=env)
    except Exception as e:
        logmod.error("plugin", f"spawn failed: {e}")
        _set(status=STATUS_FAILED, finished_at=time.time(), exit_code=None)
        return
    with _LOCK:
        _PROC = proc
    _write_pid_file(plugin["id"], proc.pid, args, plugin["path"])
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                logmod.info(f"plugin:{plugin['id']}", line)
        proc.wait()
    finally:
        with _LOCK:
            _PROC = None
        _clear_pid_file()
    code = proc.returncode
    if code == 0:
        logmod.ok("plugin", f"{plugin['id']} done")
        _set(status=STATUS_OK, finished_at=time.time(), exit_code=code)
    else:
        logmod.error("plugin", f"{plugin['id']} exit={code}")
        _set(status=STATUS_FAILED, finished_at=time.time(), exit_code=code)


def start(plugin_id, args=None):
    with _LOCK:
        if _STATE["status"] == STATUS_RUNNING:
            return {"ok": False, "error": f"already running: {_STATE['plugin_id']}"}
    plugins = plugins_reader.scan()
    plugin = next((p for p in plugins if p["id"] == plugin_id), None)
    if plugin is None:
        return {"ok": False, "error": f"plugin not found: {plugin_id}"}
    if plugins_reader.update_defaults(plugin_id, args or {}):
        logmod.info("plugin", f"manifest defaults updated: {plugin_id}")
    t = threading.Thread(target=_run, args=(plugin, args or {}),
                         name=f"plugin:{plugin_id}", daemon=True)
    t.start()
    return {"ok": True, "state": state()}


def stop():
    with _LOCK:
        proc = _PROC
        recovered_pid = _RECOVERED_PID
        plugin_id = _STATE.get("plugin_id")
    # Normal path: the run thread owns a Popen handle.
    if proc is not None:
        try:
            proc.terminate()
        except Exception as e:
            return {"ok": False, "error": f"terminate failed: {e}"}
        logmod.warn("plugin", f"stop signalled to {plugin_id}")
        _set(status=STATUS_STOPPED)
        return {"ok": True}
    # Recovered path: server was restarted while a run was in flight. We have
    # the PID but no Popen handle; kill via the OS.
    if recovered_pid:
        _process_terminate(recovered_pid)
        _clear_pid_file()
        logmod.warn("plugin", f"stop signalled to recovered pid {recovered_pid} ({plugin_id})")
        _set(status=STATUS_STOPPED, finished_at=time.time())
        return {"ok": True, "via": "recovered_pid"}
    return {"ok": False, "error": "no plugin running"}


# Run-once at module load: if a previous run is still alive (server restart
# scenario), restore state so the dashboard sees it and stop() can reach it.
_recover_from_disk()
