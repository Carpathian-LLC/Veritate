# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - dashboard flow to build the rag (RAG) corpus and run rag SFT.
# - one job at a time; spawns the rag scripts as a subprocess, streams stdout to
#   a log file, exposes running/done + a log tail via status.
# - subprocess + log + status approach mirrors training/trainer_runner.py.
# veritate_mri/routes/rag_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import subprocess
import sys
import threading
import time

from flask import request

from runtime import logs as logmod
from readers import paths

from ._common import user_error

# ------------------------------------------------------------------------------------
# Constants

BUILD_SCRIPT = os.path.join(paths.REPO_ROOT, "experiments", "v2", "rag", "build_grounded.py")
TRAIN_SCRIPT = os.path.join(paths.REPO_ROOT, "experiments", "v2", "rag", "sft_grounded.py")
LOG_FILE     = os.path.join(paths.REPO_ROOT, ".rag_run.log")
RAG_STEM = "rag_ui"
DEFAULT_FACTS = 200
DEFAULT_STEPS = 1500
TAIL_BYTES   = 8192
_NO_WINDOW   = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

_LOCK  = threading.Lock()
_STATE = {"status": "idle", "phase": None, "started_at": None,
          "finished_at": None, "exit_code": None}
_PROC  = None

# ------------------------------------------------------------------------------------
# Functions

def _claim(phase):
    with _LOCK:
        if _STATE["status"] == "running":
            return False
        _STATE.update(status="running", phase=phase, started_at=time.time(),
                      finished_at=None, exit_code=None)
        return True


def _set(**kw):
    with _LOCK:
        _STATE.update(kw)


def _run_steps(steps):
    global _PROC
    log_fp = open(LOG_FILE, "w", encoding="utf-8", buffering=1)
    try:
        for phase, argv in steps:
            _set(phase=phase)
            logmod.info("rag", f"start {phase}: {' '.join(argv[1:])}")
            try:
                proc = subprocess.Popen(argv, cwd=paths.REPO_ROOT, stdout=log_fp,
                                        stderr=subprocess.STDOUT, creationflags=_NO_WINDOW,
                                        env=os.environ.copy())
            except OSError as e:
                logmod.error("rag", f"{phase} launch failed: {e}")
                _set(status="failed", finished_at=time.time(), exit_code=None)
                return
            with _LOCK:
                _PROC = proc
            try:
                proc.wait()
            finally:
                with _LOCK:
                    _PROC = None
            with _LOCK:
                stopped = _STATE["status"] == "stopped"
            if stopped:
                logmod.info("rag", f"{phase} stopped")
                return
            code = proc.returncode
            if code != 0:
                logmod.error("rag", f"{phase} exit={code}")
                _set(status="failed", finished_at=time.time(), exit_code=code)
                return
            logmod.ok("rag", f"{phase} done")
        _set(status="ok", finished_at=time.time(), exit_code=0)
    finally:
        try: log_fp.close()
        except OSError: pass


def _spawn(steps):
    t = threading.Thread(target=_run_steps, args=(steps,), name="rag:job", daemon=True)
    t.start()


def _tail():
    if not os.path.isfile(LOG_FILE):
        return ""
    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
        try:
            f.seek(max(0, os.path.getsize(LOG_FILE) - TAIL_BYTES))
        except OSError:
            pass
        return f.read()


def register(app):
    @app.route("/rag/build_corpus", methods=["POST"])
    def rag_build_corpus():
        body = request.get_json(silent=True) or {}
        n_facts = int(body.get("n_facts") or 0)
        if n_facts <= 0:
            return ({"ok": False, "error": "n_facts must be a positive integer"}, 400)
        if not _claim("build_corpus"):
            return ({"ok": False, "error": f"already running: {_STATE['phase']}"}, 409)
        argv = [sys.executable, "-u", BUILD_SCRIPT,
                "--n_facts", str(n_facts), "--stem", RAG_STEM]
        _spawn([("build_corpus", argv)])
        return {"ok": True, "phase": "build_corpus", "stem": RAG_STEM}

    @app.route("/rag/train", methods=["POST"])
    def rag_train():
        body = request.get_json(silent=True) or {}
        source = (body.get("source") or "").strip()
        if not source:
            return ({"ok": False, "error": "no base model available; train or import a checkpoint first"}, 400)
        name = (body.get("name") or "").strip() or f"{source}_rag"
        steps = int(body.get("steps") or 0) or DEFAULT_STEPS
        n_facts = int(body.get("n_facts") or 0) or DEFAULT_FACTS
        steps_list = []
        train_bin = paths.corpus_train_path(RAG_STEM)
        if not (os.path.isfile(train_bin) and os.path.getsize(train_bin) > 0):
            steps_list.append(("build_corpus", [sys.executable, "-u", BUILD_SCRIPT,
                                                "--n_facts", str(n_facts), "--stem", RAG_STEM]))
        steps_list.append(("train", [sys.executable, "-u", TRAIN_SCRIPT,
                                     "--source", source, "--name", name,
                                     "--corpus", RAG_STEM, "--steps", str(steps)]))
        if not _claim(steps_list[0][0]):
            return ({"ok": False, "error": f"already running: {_STATE['phase']}"}, 409)
        _spawn(steps_list)
        return {"ok": True, "phase": steps_list[0][0], "name": name,
                "auto_built": len(steps_list) > 1}

    @app.route("/rag/stop", methods=["POST"])
    def rag_stop():
        with _LOCK:
            proc = _PROC
            running = _STATE["status"] == "running"
        if not running or proc is None:
            return {"ok": True, "running": False}
        try:
            proc.terminate()
        except OSError as e:
            return ({"ok": False, "error": user_error(e)}, 500)
        _set(status="stopped", finished_at=time.time(), exit_code=None)
        logmod.info("rag", "stop requested")
        return {"ok": True, "running": False}

    @app.route("/rag/status")
    def rag_status():
        try:
            with _LOCK:
                st = dict(_STATE)
            st["ok"] = True
            st["running"] = st["status"] == "running"
            st["log"] = _tail()
            return st
        except Exception as e:
            logmod.error("rag", f"status failed: {type(e).__name__}: {e}")
            return ({"ok": False, "error": user_error(e)}, 500)
