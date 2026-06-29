# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - mesh dashboard routes: status, token mgmt, test-connection, role switch.
# - settings keys read/written: mesh_role, mesh_hub_address, mesh_auth_token.
# - token is shown unmasked only to localhost callers.
# veritate_mri/routes/mesh_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import secrets
import time
import urllib.error
import urllib.request

from flask import request

from runtime import settings as settings_mod

from ._common import user_error

# ------------------------------------------------------------------------------------
# Constants

VALID_ROLES = ("off", "node", "hub", "both")
LOCAL_ADDRS = ("127.0.0.1", "::1", "localhost")
TEST_TIMEOUT_S = 5
TOKEN_BYTES = 32
NODES_PATH = "/mesh/hub/nodes"

# ------------------------------------------------------------------------------------
# Functions

def register(app):
    @app.route("/mesh/status")
    def mesh_status_route():
        s = settings_mod.get()
        role = s.get("mesh_role") or "off"
        hub_address = s.get("mesh_hub_address") or ""
        has_token = bool(s.get("mesh_auth_token"))

        try:
            from veritate_mesh import node as node_mod
            node_state = getattr(node_mod, "_STATE", None)
        except ImportError:
            node_state = None
        node_registered = bool(node_state and node_state.get("registered"))
        last_heartbeat  = node_state.get("last_heartbeat") if node_state else None
        current_job     = node_state.get("current_job")    if node_state else None

        try:
            from veritate_mesh import hub as hub_mod
            hub_nodes = hub_mod.get_registry().list_all()
        except Exception:
            hub_nodes = None
        return {
            "role": role,
            "hub_address": hub_address,
            "has_token": has_token,
            "node_registered": node_registered,
            "last_heartbeat": last_heartbeat,
            "current_job": current_job,
            "hub_nodes": hub_nodes,
        }

    @app.route("/mesh/token", methods=["GET"])
    def mesh_token_get_route():
        s = settings_mod.get()
        token = s.get("mesh_auth_token") or ""
        has_token = bool(token)
        if request.remote_addr not in LOCAL_ADDRS:
            return {"has_token": has_token, "token": None}
        return {"has_token": has_token, "token": token if has_token else None}

    @app.route("/mesh/token/regenerate", methods=["POST"])
    def mesh_token_regenerate_route():
        new_token = secrets.token_urlsafe(TOKEN_BYTES)
        try:
            settings_mod.update({"mesh_auth_token": new_token})
        except ValueError as ve:
            return {"ok": False, "error": user_error(ve)}, 400
        return {"ok": True, "token": new_token}

    @app.route("/mesh/test_connection", methods=["POST"])
    def mesh_test_connection_route():
        body = request.get_json(silent=True) or {}
        s = settings_mod.get()
        hub_address = (body.get("hub_address") or s.get("mesh_hub_address") or "").rstrip("/")
        auth_token  = body.get("auth_token")  or s.get("mesh_auth_token")  or ""
        if not hub_address:
            return {"ok": False, "status": None, "error": "hub_address is empty", "response_ms": 0}

        url = f"{hub_address}{NODES_PATH}"
        req = urllib.request.Request(url, method="GET")
        if auth_token:
            req.add_header("Authorization", f"Bearer {auth_token}")

        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=TEST_TIMEOUT_S) as resp:
                status = resp.getcode()
                elapsed_ms = int((time.monotonic() - start) * 1000)
                return {"ok": 200 <= int(status) < 300, "status": int(status), "error": None, "response_ms": elapsed_ms}
        except urllib.error.HTTPError as he:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return {"ok": False, "status": int(he.code), "error": user_error(he), "response_ms": elapsed_ms}
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return {"ok": False, "status": None, "error": user_error(e), "response_ms": elapsed_ms}

    @app.route("/mesh/role", methods=["POST"])
    def mesh_role_route():
        body = request.get_json(silent=True) or {}
        role = (body.get("role") or "").lower()
        if role not in VALID_ROLES:
            return {"ok": False, "error": f"invalid role: {role!r}; expected one of {VALID_ROLES}"}, 400
        try:
            settings_mod.update({"mesh_role": role})
        except ValueError as ve:
            return {"ok": False, "error": user_error(ve)}, 400
        return {"ok": True, "role": role, "restart_required": True}
