# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - trainer catalog, start/stop, repo sync, core-trainers index, open-folder.
#   Enforces fresh-run name collisions on /trainers/run. Every endpoint runs
#   through _safe so any exception lands in the dashboard log ring with a
#   parseable JSON error body (avoids the WebKit "string did not match the
#   expected pattern" symptom users hit when an HTML 500 reaches r.json()).
# veritate_mri/routes/trainers_routes.py
# ------------------------------------------------------------------------------------
# Imports:

from flask import request

from readers import models, trainers as trainers_reader
from runtime import logs as logmod
from training import trainer_runner
from training.sync import trainers_sync

from ._common import open_folder


def _safe(source, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logmod.error(source, msg)
        return ({"ok": False, "error": msg}, 500)


def register(app):
    @app.route("/trainers")
    def trainers_index():
        return _safe("trainers", lambda: {
            "trainers": trainers_reader.scan(),
            "running":  trainer_runner.state(),
        })

    @app.route("/trainers/run", methods=["POST"])
    def trainers_run():
        def _do():
            body = request.get_json(silent=True) or {}
            trainer_id = body.get("id")
            if not trainer_id:
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
            return trainer_runner.start(trainer_id, args)
        return _safe("trainers", _do)

    @app.route("/trainers/stop", methods=["POST"])
    def trainers_stop():
        return _safe("trainers", trainer_runner.stop)

    @app.route("/core_trainers")
    def core_trainers_index():
        def _do():
            from veritate_core import core_plugins as _cp
            flow = (request.args.get("flow") or "").strip() or None
            return {"trainers": _cp.all_plugins(flow=flow)}
        return _safe("trainers", _do)

    @app.route("/trainers/git/status")
    def trainers_git_status():
        return _safe("trainers-sync", trainers_sync.status)

    @app.route("/trainers/git/sync", methods=["POST"])
    def trainers_git_sync():
        def _do():
            body = request.get_json(silent=True) or {}
            actions = body.get("actions") if isinstance(body.get("actions"), dict) else None
            branch  = body.get("branch") if isinstance(body.get("branch"), str) else None
            return trainers_sync.sync(actions=actions, branch=branch)
        return _safe("trainers-sync", _do)

    @app.route("/trainers/git/check", methods=["POST"])
    def trainers_git_check():
        return _safe("trainers-sync", trainers_sync.check)

    @app.route("/trainers/git/files")
    def trainers_git_files():
        """Per-file table with three-state classification."""
        return _safe("trainers-sync", trainers_sync.files)

    @app.route("/trainers/open_folder", methods=["POST"])
    def trainers_open_folder():
        return _safe("trainers", lambda: open_folder(trainers_reader.PLUGINS_ROOT))
