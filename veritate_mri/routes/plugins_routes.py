# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - trainer catalog, start/stop, core plugins index, trainer repo git sync,
#   open-folder. enforces fresh-run name collisions on /trainers/run.
# veritate_mri/routes/plugins_routes.py
# ------------------------------------------------------------------------------------
# Imports:

from flask import request

from readers import models, trainers as plugins_reader
from training import trainer_runner as plugin_runner
from training.sync import trainers_sync as plugins_sync

from ._common import open_folder

# ------------------------------------------------------------------------------------
# Constants


# ------------------------------------------------------------------------------------
# Functions

def register(app):
    @app.route("/plugins")
    def plugins_index():
        return {"plugins": plugins_reader.scan(), "running": plugin_runner.state()}

    @app.route("/trainers/run", methods=["POST"])
    def plugins_run():
        body = request.get_json(silent=True) or {}
        plugin_id = body.get("id")
        if not plugin_id:
            return ({"ok": False, "error": "missing 'id'"}, 400)
        args = body.get("args") or {}
        if not (args.get("resume") or args.get("base_ckpt")):
            user_name = (args.get("name") or "").strip()
            size      = (args.get("size") or "").strip()
            if user_name and size:
                slug = models.slugify_user_name(user_name)
                if slug:
                    composed = f"{slug}_{size}"
                    if models.exists(composed):
                        return ({
                            "ok": False,
                            "error": f"model '{composed}' already exists. pick a different name "
                                     "or use Continue Training to extend the existing run.",
                        }, 409)
        return plugin_runner.start(plugin_id, args)

    @app.route("/trainers/stop", methods=["POST"])
    def plugins_stop():
        return plugin_runner.stop()

    @app.route("/core_plugins")
    def core_plugins_index():
        from veritate_core import core_plugins as _cp
        flow = (request.args.get("flow") or "").strip() or None
        return {"plugins": _cp.all_plugins(flow=flow)}

    @app.route("/trainers/git/status")
    def plugins_git_status():
        return plugins_sync.status()

    @app.route("/trainers/git/sync", methods=["POST"])
    def plugins_git_sync():
        body = request.get_json(silent=True) or {}
        actions = body.get("actions") if isinstance(body.get("actions"), dict) else None
        branch  = body.get("branch") if isinstance(body.get("branch"), str) else None
        return plugins_sync.sync(actions=actions, branch=branch)

    @app.route("/trainers/git/check", methods=["POST"])
    def plugins_git_check():
        return plugins_sync.check()

    @app.route("/trainers/git/files")
    def plugins_git_files():
        """Per-file table with three-state classification."""
        return plugins_sync.files()

    @app.route("/trainers/open_folder", methods=["POST"])
    def plugins_open_folder():
        return open_folder(plugins_reader.PLUGINS_ROOT)
