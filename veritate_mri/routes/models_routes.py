# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - models repo git sync + fork + open-folder, plus the pytorch-models picker list.
# veritate_mri/routes/models_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import os

from flask import current_app, request

from readers import (
    capabilities as caps_reader, checkpoints, config as cfg_reader, models, paths,
)
from training import fork as fork_mod
from training.sync import models_sync

from ._common import open_folder

# ------------------------------------------------------------------------------------
# Constants


# ------------------------------------------------------------------------------------
# Functions

def register(app):
    @app.route("/models/git/status")
    def models_git_status():
        return models_sync.status()

    @app.route("/models/git/sync", methods=["POST"])
    def models_git_sync():
        body = request.get_json(silent=True) or {}
        actions = body.get("actions") if isinstance(body.get("actions"), dict) else None
        branch  = body.get("branch") if isinstance(body.get("branch"), str) else None
        return models_sync.sync(actions=actions, branch=branch)

    @app.route("/models/git/check", methods=["POST"])
    def models_git_check():
        return models_sync.check()

    @app.route("/models/git/files")
    def models_git_files():
        """Per-file table + per-dir provenance for the models repo."""
        return models_sync.files()

    @app.route("/models/git/progress")
    def models_git_progress():
        """Live byte-counter for the active models_sync.sync() run."""
        return models_sync.progress()

    @app.route("/models/fork", methods=["POST"])
    def models_fork():
        """Copy latest checkpoint of <source> into a new dir <new_name>."""
        body = request.get_json(silent=True) or {}
        try:
            return fork_mod.fork_model(body.get("source"), body.get("new_name"))
        except fork_mod.ForkError as e:
            return ({"ok": False, "error": str(e)}, 400)

    @app.route("/models/open_folder", methods=["POST"])
    def models_open_folder():
        return open_folder(paths.MODELS_ROOT)

    @app.route("/pytorch-models")
    def pytorch_models_index():
        out = []
        cfg = current_app.config
        cur_model = cfg.get("BRAIN_MODEL") or cfg.get("DEFAULT_MODEL")
        for name in models.list_models():
            step = checkpoints.latest_step(name)
            if step is None:
                continue
            try: mcfg = cfg_reader.load(name) or {}
            except Exception: mcfg = {}
            plugin = (mcfg.get("plugin") or "").strip()
            n_params = mcfg.get("n_params_total")
            shape = mcfg.get("shape") or {}
            try: mtime = os.path.getmtime(checkpoints.path_for(name, step))
            except OSError: mtime = 0
            out.append({
                "name":        name,
                "step":        int(step),
                "is_current":  name == cur_model,
                "plugin":      plugin,
                "n_params":    int(n_params) if n_params else None,
                "hidden":      shape.get("hidden"),
                "layers":      shape.get("layers"),
                "description": cfg_reader.description(name) or "",
                "mtime":       mtime,
                "capabilities": caps_reader.read(name),
            })
        out.sort(key=lambda r: -r["mtime"])
        return {"models": out}
