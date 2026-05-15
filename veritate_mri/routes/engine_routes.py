# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - c-engine build status + trigger, c engine binary listing, c model listing,
#   c subprocess respawn via /c-config.
# veritate_mri/routes/engine_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import os

from flask import current_app, request

from inference.backends.c_engine import CTracedSubprocess
from readers import bin as binr, config as cfg_reader, engine, models, paths
from runtime import logs as logmod
from training import build_runner

# ------------------------------------------------------------------------------------
# Constants


# ------------------------------------------------------------------------------------
# Functions

def register(app):
    @app.route("/engine/status")
    def engine_status():
        s = build_runner.state()
        cfg = current_app.config
        cur_exe = cfg.get("C_EXE")
        s["c_subprocess_running"] = cfg.get("C_SUBPROCESS") is not None
        s["c_exe"] = cur_exe
        return s

    @app.route("/engine/build", methods=["POST"])
    def engine_build_trigger():
        return build_runner.start()

    @app.route("/c-engines")
    def c_engines_index():
        out = []
        cur_exe = current_app.config.get("C_EXE")
        cur_abs = os.path.abspath(cur_exe) if cur_exe else None
        for e in engine.engines():
            ap = os.path.abspath(e.get("path") or "")
            if not os.path.isfile(ap): continue
            try: st = os.stat(ap)
            except OSError: continue
            out.append({
                **e,
                "path": ap,
                "exists": True,
                "is_current": ap == cur_abs,
                "mtime": st.st_mtime,
                "size":  st.st_size,
            })
        return {"engines": out}

    @app.route("/c-models")
    def c_models_index():
        out = []
        cur_path = current_app.config.get("C_MODEL")
        cur_abs = os.path.abspath(cur_path) if cur_path else None
        for name in models.list_models():
            if not binr.exists(name): continue
            bp = paths.bin_path(name)
            try: st = os.stat(bp)
            except OSError: continue
            precision, version = binr.header(name)
            training, activation = cfg_reader.training_kind(name)
            out.append({
                "name": name,
                "bin_path": os.path.abspath(bp),
                "is_current": os.path.abspath(bp) == cur_abs,
                "mtime": st.st_mtime,
                "size":  st.st_size,
                "precision":   precision,
                "bin_version": version,
                "training":    training,
                "activation":  activation,
                "act_boost":   binr.act_boost(name),
                "description": cfg_reader.description(name),
            })
        out.sort(key=lambda r: -r["mtime"])
        return {"models": out}

    @app.route("/c-config", methods=["POST"])
    def c_config():
        cfg = current_app.config
        body = request.get_json(silent=True) or {}
        new_exe   = body.get("exe",   cfg["C_EXE"])
        new_model = body.get("model", cfg["C_MODEL"])
        if new_exe is not None and not os.path.isfile(new_exe):
            return ({"ok": False, "error": f"exe not found: {new_exe}"}, 400)
        if new_model is not None and not os.path.isfile(new_model):
            return ({"ok": False, "error": f"model not found: {new_model}"}, 400)
        if new_exe is None:
            return ({"ok": False, "error": "no c engine exe selected"}, 400)
        old = cfg.get("C_SUBPROCESS")
        if old is not None:
            try: old.close()
            except Exception: pass
        name = os.path.basename(os.path.dirname(new_model)) if new_model else None
        boost = binr.act_boost(name) if name else None
        if boost is not None and boost > 1:
            logmod.warn("backends", f"c-config: {name} act_boost={boost} (untrusted); engine loads anyway via VERITATE_ALLOW_HIGH_ACT_BOOST=1, output may be gibberish")
        try:
            sub = CTracedSubprocess(new_exe, new_model)
        except Exception as e:
            cfg["C_SUBPROCESS"] = None
            return ({"ok": False, "error": f"respawn failed: {e}"}, 500)
        cfg["C_EXE"]        = new_exe
        cfg["C_MODEL"]      = new_model
        cfg["C_SUBPROCESS"] = sub
        cfg["C_BLOCKED_REASON"] = None
        cfg["C_BLOCKED_MODEL"]  = None
        logmod.info("c-config", f"exe={new_exe} model={new_model} pid={sub.proc.pid}")
        precision, version = (binr.header(name) if name else ("?", 0))
        training, activation = (cfg_reader.training_kind(name) if name else ("", ""))
        return {
            "ok": True,
            "c_exe_path":  new_exe,
            "c_exe":       os.path.basename(new_exe),
            "c_model_path": new_model,
            "c_model":     os.path.basename(new_model) if new_model else None,
            "c_model_dir": name,
            "c_model_precision":   precision,
            "c_model_bin_version": version,
            "c_model_training":    training,
            "c_model_activation":  activation,
            "c_model_act_boost":   boost,
        }
