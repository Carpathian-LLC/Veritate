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
# veritate_mri/lifecycle.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import subprocess
import threading
import time

import build_runner
import logs as logmod
import plugin_runner

# ------------------------------------------------------------------------------------
# Constants

DELAY_SECS = 0.4
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

# ------------------------------------------------------------------------------------
# Functions

def _cleanup(app_config, stop_plugin=False):
    if stop_plugin:
        try:
            if plugin_runner.is_running():
                plugin_runner.stop()
                logmod.warn("lifecycle", "stopped active plugin run")
        except Exception as e:
            logmod.error("lifecycle", f"plugin stop: {e}")
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


def _do_restart(app_config, launch_cmd):
    time.sleep(DELAY_SECS)
    _cleanup(app_config, stop_plugin=False)
    if not launch_cmd:
        logmod.error("lifecycle", "no LAUNCH_CMD captured; cannot restart in place. "
                                  "falling back to kill. re-run the launcher manually.")
        os._exit(2)
    if "--skip-build" not in launch_cmd:
        launch_cmd = launch_cmd + ["--skip-build"]
    time.sleep(0.5)
    logmod.warn("lifecycle", f"detached relaunch (training preserved): {' '.join(launch_cmd)}")
    try:
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
                | _NO_WINDOW
            )
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen(launch_cmd, **kwargs)
    except Exception as e:
        logmod.error("lifecycle", f"detached relaunch failed: {e}")
        os._exit(4)
    os._exit(0)


def _do_kill(app_config):
    time.sleep(DELAY_SECS)
    _cleanup(app_config, stop_plugin=True)
    logmod.warn("lifecycle", "server kill requested (training stopped). exiting.")
    os._exit(0)


def _do_soft_reload(app_config, launch_cmd):
    time.sleep(DELAY_SECS)
    try:
        sub = app_config.get("C_SUBPROCESS") if app_config is not None else None
        if sub is not None:
            sub.close()
            logmod.ok("lifecycle", "closed c engine subprocess (training subprocess preserved)")
    except Exception as e:
        logmod.error("lifecycle", f"c subprocess close: {e}")
    if not launch_cmd:
        logmod.error("lifecycle", "no LAUNCH_CMD captured; cannot soft-reload.")
        os._exit(2)
    if "--skip-build" not in launch_cmd:
        launch_cmd = launch_cmd + ["--skip-build"]
    time.sleep(0.5)
    logmod.warn("lifecycle", f"soft reload (training preserved): {' '.join(launch_cmd)}")
    try:
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
                | _NO_WINDOW
            )
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen(launch_cmd, **kwargs)
    except Exception as e:
        logmod.error("lifecycle", f"soft reload failed: {e}")
        os._exit(4)
    os._exit(0)


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
