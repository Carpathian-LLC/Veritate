# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - spawn a plugin script as a subprocess. one at a time globally.
# - stdout streams into the in-memory log ring. status exposed via state().
# veritate_mri/training/trainer_runner.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import signal
import subprocess
import sys
import threading
import time

from readers import paths, trainers as plugins_reader
from runtime import logs as logmod
from runtime import settings as settings_mod

# ------------------------------------------------------------------------------------
# Constants

STATUS_IDLE    = "idle"
STATUS_RUNNING = "running"
STATUS_OK      = "ok"
STATUS_FAILED  = "failed"
STATUS_STOPPED = "stopped"

PID_FILE = os.path.join(paths.REPO_ROOT, ".plugin_pid.json")
RUN_LOG_FILE = os.path.join(paths.REPO_ROOT, ".plugin_run.log")
TAIL_POLL_SECS = 0.25
WATCH_POLL_SECS = 2.0
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

PYTORCH_ALLOC_ENV_KEY     = "PYTORCH_CUDA_ALLOC_CONF"
PYTORCH_ALLOC_ENV_DEFAULT = "expandable_segments:True"

# Mirror of save.PLUGIN_ID_ENV. Duplicated to avoid pulling save.py into the
# runner's import graph (save.py imports torch lazily; the runner runs in
# the parent process and shouldn't pay that cost).
PLUGIN_ID_ENV = "VERITATE_PLUGIN_ID"

# Trainer device override. The settings-tab "Device preference" dropdown
# writes this; trainers' pick_device() reads it. "auto" or unset = historical
# best-available selection.
DEVICE_ENV = "VERITATE_DEVICE"

# Sane upper bound for BLAS / OpenMP threads. Going above this rarely helps and
# often hurts (NUMA / thread-thrash / memory-bandwidth saturation).
_BLAS_THREAD_CAP = 16

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
# PID-file persistence. Lets the dashboard re-discover an in-flight training
# run after a server restart; training subprocess survives parent exit on
# Windows, so the PID alone is enough for the Stop button to reach it.

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
    PID reuse, the OS may have recycled the PID to an unrelated process."""
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


def _tail_run_log(plugin_id, start_pos, stop_event):
    """Stream new lines from RUN_LOG_FILE into the in-memory log ring.
    Decouples the dashboard log view from the subprocess stdout pipe so the
    plugin survives a parent restart."""
    pos = int(start_pos)
    src = f"plugin:{plugin_id}"
    leftover = ""
    while not stop_event.is_set():
        try:
            if os.path.isfile(RUN_LOG_FILE):
                with open(RUN_LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
                if chunk:
                    data = leftover + chunk
                    lines = data.split("\n")
                    leftover = lines.pop()
                    for line in lines:
                        line = line.rstrip("\r")
                        if line:
                            logmod.info(src, line)
        except Exception:
            pass
        if stop_event.wait(TAIL_POLL_SECS):
            return


def _watch_recovered(pid, plugin_id, cmd_marker, stop_event):
    """Mark a recovered run as finished when its OS process exits.
    Runs only when the dashboard reattached to a still-live PID after restart."""
    global _RECOVERED_PID
    while not stop_event.is_set():
        if not _process_alive(pid, cmd_marker=cmd_marker):
            _clear_pid_file()
            with _LOCK:
                if _STATE.get("status") == STATUS_RUNNING:
                    _STATE.update({
                        "status":      STATUS_OK,
                        "finished_at": time.time(),
                        "exit_code":   None,
                    })
                _RECOVERED_PID = None
            logmod.ok("plugin", f"recovered run finished (pid={pid}, {plugin_id})")
            return
        if stop_event.wait(WATCH_POLL_SECS):
            return


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
    start_pos = 0
    try:
        if os.path.isfile(RUN_LOG_FILE):
            start_pos = os.path.getsize(RUN_LOG_FILE)
    except Exception:
        start_pos = 0
    stop_event = threading.Event()
    threading.Thread(target=_tail_run_log,
                     args=(rec.get("plugin_id"), start_pos, stop_event),
                     name="plugin-tailer-recovered", daemon=True).start()
    threading.Thread(target=_watch_recovered,
                     args=(int(pid), rec.get("plugin_id"), rec.get("cmd_marker"), stop_event),
                     name="plugin-watch-recovered", daemon=True).start()

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
    out = [sys.executable, "-u", plugin["path"]]
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
    env[PLUGIN_ID_ENV] = str(plugin["id"])
    pref = (settings_mod.get().get("device_preference") or "auto").strip().lower()
    if pref and pref != "auto":
        env[DEVICE_ENV] = pref
    # macOS x86_64 (Intel Mac) reports torch.backends.mps.is_available() == True,
    # but MPS only works on Apple Silicon and crashes mid-step there. Force CPU on
    # that tier regardless of preference (incl. an explicit mps pick), since the
    # synced trainer pick_device() copies do not all arch-guard MPS yet.
    import platform as _plat
    if _plat.system() == "Darwin" and _plat.machine().lower() != "arm64":
        env[DEVICE_ENV] = "cpu"
    # Match BLAS and OpenMP thread budgets to the physical-core count so libtorch
    # and oneDNN parallelize across the same number of cores the trainer asks for.
    # Caller-set values win; we only fill in when the user hasn't already.
    try:
        from veritate_core.plugin import hardware as _hw
        _phys = _hw.physical_cores()
    except (ImportError, ValueError):
        _phys = 0
    if _phys:
        _budget = str(min(_phys, _BLAS_THREAD_CAP))
        env.setdefault("OMP_NUM_THREADS",       _budget)
        env.setdefault("MKL_NUM_THREADS",       _budget)
        env.setdefault("OPENBLAS_NUM_THREADS",  _budget)
        env.setdefault("VECLIB_MAXIMUM_THREADS", _budget)
        env.setdefault("NUMEXPR_NUM_THREADS",   _budget)
    try:
        log_fp = open(RUN_LOG_FILE, "w", encoding="utf-8", buffering=1)
    except Exception as e:
        logmod.error("plugin", f"open run log failed: {e}")
        _set(status=STATUS_FAILED, finished_at=time.time(), exit_code=None)
        return
    try:
        proc = subprocess.Popen(argv, cwd=paths.REPO_ROOT,
                                stdout=log_fp, stderr=subprocess.STDOUT,
                                creationflags=_NO_WINDOW,
                                env=env)
    except Exception as e:
        logmod.error("plugin", f"spawn failed: {e}")
        try: log_fp.close()
        except Exception: pass
        _set(status=STATUS_FAILED, finished_at=time.time(), exit_code=None)
        return
    with _LOCK:
        _PROC = proc
    _write_pid_file(plugin["id"], proc.pid, args, plugin["path"])
    stop_event = threading.Event()
    tailer = threading.Thread(target=_tail_run_log,
                              args=(plugin["id"], 0, stop_event),
                              name=f"plugin-tailer:{plugin['id']}", daemon=True)
    tailer.start()
    try:
        proc.wait()
    finally:
        stop_event.set()
        try: log_fp.close()
        except Exception: pass
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
    # Atomic claim: test-and-set status under the lock so two near-simultaneous
    # starts cannot both pass the guard (the worker thread only sets RUNNING
    # later, leaving a race window otherwise).
    with _LOCK:
        if _STATE["status"] == STATUS_RUNNING:
            return {"ok": False, "error": f"already running: {_STATE['plugin_id']}"}
        _STATE["status"] = STATUS_RUNNING
        _STATE["plugin_id"] = plugin_id
    plugins = plugins_reader.scan()
    plugin = next((p for p in plugins if p["id"] == plugin_id), None)
    if plugin is None:
        with _LOCK:
            _STATE["status"] = STATUS_IDLE
        return {"ok": False, "error": f"plugin not found: {plugin_id}"}
    if plugins_reader.update_defaults(plugin_id, args or {}):
        logmod.info("plugin", f"manifest defaults updated: {plugin_id}")
    a = args or {}
    model_name = a.get("resume") or a.get("name") or a.get("version") or plugin_id
    try:
        from runtime import heartbeat as _hb
        _hb.record_training_event(str(model_name), plugin_id)
    except Exception:
        pass
    t = threading.Thread(target=_run, args=(plugin, a),
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
