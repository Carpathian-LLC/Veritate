# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - server lifecycle controls for the Settings page. all three actions go
#   through _cleanup(); the stop_plugin flag is what differentiates them.
# - soft_reload    : closes only the C engine subprocess, then re-execs.
#                    training subprocess survives and the new server reattaches
#                    via plugin_runner's PID file.
# - restart        : same as soft_reload plus shuts down the build runner.
#                    training subprocess still survives.
# - kill           : full cleanup including stopping the training subprocess,
#                    then exits via os._exit. nothing comes back without a
#                    manual relaunch.
# veritate_mri/runtime/lifecycle.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import subprocess
import threading
import time

from training import build_runner
from training import trainer_runner as plugin_runner
from . import logs as logmod

# ------------------------------------------------------------------------------------
# Constants

DELAY_SECS     = 0.2
RELAUNCH_GAP   = 0.1
RELAUNCH_FLAGS = ("--skip-build", "--no-browser")

# ------------------------------------------------------------------------------------
# Functions

def _detached_popen_kwargs():
    kwargs = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin":  subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        kwargs["start_new_session"] = True
    return kwargs

def _cleanup(app_config, stop_plugin=False, stop_build=True):
    if stop_plugin:
        try:
            if plugin_runner.is_running():
                plugin_runner.stop()
                logmod.warn("lifecycle", "stopped active plugin run")
        except Exception as e:
            logmod.error("lifecycle", f"plugin stop: {e}")
    if stop_build:
        try:
            if build_runner.stop():
                logmod.warn("lifecycle", "stopped active engine build")
        except Exception as e:
            logmod.error("lifecycle", f"build stop: {e}")
    try:
        sub = app_config.get("C_SUBPROCESS") if app_config is not None else None
        if sub is not None:
            sub.close()
            logmod.ok("lifecycle", "closed c engine subprocess")
    except Exception as e:
        logmod.error("lifecycle", f"c subprocess close: {e}")


def _relaunch(launch_cmd, mode):
    """Shared exec-self relaunch for restart and soft_reload. mode is a label
    for the log line; launch_cmd is mutated in place to carry RELAUNCH_FLAGS."""
    if not launch_cmd:
        logmod.error("lifecycle", f"no LAUNCH_CMD captured; cannot {mode}.")
        os._exit(2)
    for flag in RELAUNCH_FLAGS:
        if flag not in launch_cmd:
            launch_cmd = launch_cmd + [flag]
    time.sleep(RELAUNCH_GAP)
    logmod.warn("lifecycle", f"{mode} (training preserved): {' '.join(launch_cmd)}")
    try:
        subprocess.Popen(launch_cmd, **_detached_popen_kwargs())
    except Exception as e:
        logmod.error("lifecycle", f"{mode} failed: {e}")
        os._exit(4)
    os._exit(0)


def _do_restart(app_config, launch_cmd):
    time.sleep(DELAY_SECS)
    _cleanup(app_config, stop_plugin=False, stop_build=True)
    _relaunch(launch_cmd, "detached relaunch")


def _do_kill(app_config):
    time.sleep(DELAY_SECS)
    _cleanup(app_config, stop_plugin=True, stop_build=True)
    logmod.warn("lifecycle", "server kill requested (training stopped). exiting.")
    os._exit(0)


def _do_soft_reload(app_config, launch_cmd):
    time.sleep(DELAY_SECS)
    _cleanup(app_config, stop_plugin=False, stop_build=False)
    _relaunch(launch_cmd, "soft reload")


def restart(app_config):
    if app_config is None:
        return {"ok": False, "error": "app config unavailable"}
    launch_cmd = app_config.get("LAUNCH_CMD")
    if not launch_cmd:
        return {"ok": False, "error": "LAUNCH_CMD not captured at startup. "
                                       "use kill + relaunch instead."}
    threading.Thread(target=_do_restart, args=(app_config, list(launch_cmd)),
                     name="lifecycle-restart", daemon=True).start()
    return {"ok": True, "action": "restart", "delay_secs": DELAY_SECS}


def kill(app_config):
    threading.Thread(target=_do_kill, args=(app_config,),
                     name="lifecycle-kill", daemon=True).start()
    return {"ok": True, "action": "kill", "delay_secs": DELAY_SECS}


def soft_reload(app_config):
    if app_config is None:
        return {"ok": False, "error": "app config unavailable"}
    launch_cmd = app_config.get("LAUNCH_CMD")
    if not launch_cmd:
        return {"ok": False, "error": "LAUNCH_CMD not captured at startup. "
                                       "use kill + relaunch instead."}
    threading.Thread(target=_do_soft_reload, args=(app_config, list(launch_cmd)),
                     name="lifecycle-soft-reload", daemon=True).start()
    return {"ok": True, "action": "soft_reload", "delay_secs": DELAY_SECS}
