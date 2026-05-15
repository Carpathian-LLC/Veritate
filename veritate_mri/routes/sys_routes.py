# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - system metrics, hardware specs, heartbeat, self-update routes, local
#   edits, versions.
# veritate_mri/routes/sys_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import os

from flask import Response, current_app, request

from runtime import heartbeat as heartbeat_mod
from runtime import lifecycle
from runtime import sys_metrics
from training.sync import app_sync as app_sync_mod

from ._common import user_error

# ------------------------------------------------------------------------------------
# Constants

MRI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSIONS_PATH = os.path.normpath(os.path.join(MRI_ROOT, "..", "versions.json"))

# ------------------------------------------------------------------------------------
# Functions

def register(app):
    @app.route("/sys_metrics")
    def sys_metrics_route():
        return sys_metrics.snapshot()

    @app.route("/sys/specs")
    def sys_specs_get():
        return sys_metrics.load_specs() or {"detected": False}

    @app.route("/sys/detect", methods=["POST"])
    def sys_specs_detect():
        return sys_metrics.detect_and_save()

    @app.route("/heartbeat/status")
    def heartbeat_status_route():
        return heartbeat_mod.status()

    @app.route("/heartbeat/send", methods=["POST"])
    def heartbeat_send_route():
        ok_send = heartbeat_mod.send_now()
        return {"ok": bool(ok_send), **heartbeat_mod.status()}

    @app.route("/app/update_status")
    def app_update_status_route():
        return app_sync_mod.status()

    @app.route("/app/update_check", methods=["POST"])
    def app_update_check_route():
        return app_sync_mod.check()

    @app.route("/app/update_pull", methods=["POST"])
    def app_update_pull_route():
        body = request.get_json(silent=True) or {}
        force            = bool(body.get("force"))
        ignore_training  = bool(body.get("ignore_training"))
        res = app_sync_mod.pull_update(force=force, ignore_training=ignore_training)
        if res.get("ok") and body.get("reload"):
            try:
                lifecycle.restart(current_app.config)
            except Exception as e:
                res["reload_error"] = user_error(e)
        return res

    @app.route("/app/local_edits")
    def app_local_edits_route():
        """List files that diverge from the last-pulled baseline."""
        return app_sync_mod.local_edits()

    @app.route("/app/update_channel", methods=["POST"])
    def app_update_channel_route():
        body = request.get_json(silent=True) or {}
        channel = (body.get("channel") or "").lower()
        return app_sync_mod.switch_channel(channel)

    @app.route("/versions")
    def versions_route():
        if not os.path.isfile(VERSIONS_PATH):
            return ({"error": f"versions file not found: {VERSIONS_PATH}"}, 404)
        with open(VERSIONS_PATH, "r", encoding="utf-8") as f:
            return Response(f.read(), mimetype="application/json")
