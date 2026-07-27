# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - settings GET/POST, settings notices, ai assist ask. POST settings flips
#   that turn pytorch_load_mode to "always" eagerly load the brain.
# veritate_mri/routes/settings_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import time

from flask import current_app, request
from readers import checkpoints, models
from runtime import ai_assist as ai_assist_mod
from runtime import logs as logmod
from runtime import settings as settings_mod
from runtime import typing_samples

from . import _brain
from ._common import auto_thread_count, user_error

# ------------------------------------------------------------------------------------
# Constants


# ------------------------------------------------------------------------------------
# Functions

def register(app):
    @app.route("/settings", methods=["GET", "POST"])
    def settings_route():
        if request.method == "POST":
            body = request.get_json(silent=True) or {}
            try:
                out = settings_mod.update(body)
            except ValueError as ve:
                return {"error": str(ve)}, 400
            cfg = current_app.config
            if "warm_models" in body:
                from .backends_routes import warm_apply
                warm_apply(cfg, out.get("warm_models") or [])
            if out.get("pytorch_load_mode") == "always" and cfg.get("BRAIN") is None:
                try:
                    name = cfg.get("BRAIN_MODEL") or cfg.get("DEFAULT_MODEL")
                    step = cfg.get("BRAIN_STEP")  or cfg.get("DEFAULT_STEP")
                    if not name or not models.exists(name):
                        name = _brain.resolve_pytorch_model("auto")
                        if name is not None:
                            cfg["DEFAULT_MODEL"] = name
                            step = checkpoints.latest_step(name)
                            cfg["DEFAULT_STEP"]  = step
                    if name and step is not None:
                        threads = int(cfg.get("DEFAULT_THREADS") or auto_thread_count())
                        brain, name, step = _brain.load_pytorch_brain(name, step, threads)
                        cfg["BRAIN"] = brain
                        cfg["BRAIN_MODEL"] = name
                        cfg["BRAIN_STEP"]  = int(step)
                        cfg["DEFAULT_MODEL"] = name
                        cfg["DEFAULT_STEP"]  = int(step)
                        cfg["BRAIN_LAST_USED"] = time.time()
                        cfg["BRAIN_LAST_ERROR"] = None
                        logmod.ok("backends", f"pytorch eager-loaded after settings flip: {name} step {step}")
                except Exception as e:
                    cfg["BRAIN_LAST_ERROR"] = user_error(e)
                    if isinstance(e, RuntimeError) and "PyTorch inference is not enabled" in str(e):
                        logmod.warn("backends", f"pytorch backend skipped for {name}: "
                                                "non-vanilla architecture (use C engine)")
                    else:
                        logmod.error("backends", f"pytorch eager load on settings flip failed: {type(e).__name__}: {e}")
            return out
        return settings_mod.get()

    @app.route("/settings/notices", methods=["GET"])
    def settings_notices_route():
        return {"notices": settings_mod.pending_notices()}

    @app.route("/settings/api-key", methods=["POST"])
    def settings_api_key_route():
        body = request.get_json(silent=True) or {}
        action = body.get("action") or ""
        if action == "rotate":
            return settings_mod.rotate_api_key()
        if action == "clear":
            return settings_mod.clear_api_key()
        return {"error": "action must be 'rotate' or 'clear'"}, 400

    @app.route("/ai/ask", methods=["POST"])
    def ai_ask_route():
        body = request.get_json(silent=True) or {}
        kind = body.get("kind") or ""
        payload = body.get("payload") or {}
        return ai_assist_mod.ask(kind, payload)

    @app.route("/typing/samples", methods=["GET", "POST"])
    def typing_samples_route():
        """POST stores one recorded typing session; GET lists what is stored. The
        session is the raw per-keystroke evidence the draft trigger is tuned against,
        so nothing here summarizes it."""
        if request.method == "GET":
            return {"samples": typing_samples.listing(), "dir": typing_samples.SAMPLES_DIR}
        body = request.get_json(silent=True) or {}
        try:
            name = typing_samples.save(body)
        except ValueError as ve:
            return {"error": str(ve)}, 400
        except OSError as e:
            logmod.error("typing", f"sample save failed: {type(e).__name__}: {e}")
            return {"error": user_error(e, "could not write the sample")}, 500
        return {"name": name, "path": typing_samples.path(name)}

    @app.route("/typing/samples/<name>", methods=["GET"])
    def typing_sample_route(name):
        try:
            return typing_samples.load(name)
        except (OSError, ValueError):
            return {"error": "no such sample"}, 404
