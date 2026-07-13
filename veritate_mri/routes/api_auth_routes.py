# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Optional API-key gate for the programmatic API surface (/v1/*, /generate,
#   /agent/stream). Independent of the dashboard password gate. Off unless a key
#   is set in settings, so a LAN box stays open by default. When set, protected
#   endpoints require Authorization: Bearer <key>; each authed hit bumps the
#   per-key request counter. Dashboard pages, /chat, /hybrid, /static, heartbeat
#   are never gated here.
# veritate_mri/routes/api_auth_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import hmac

from flask import request

from runtime import settings as settings_mod

# ------------------------------------------------------------------------------------
# Constants

PROTECTED_EXACT    = ("/generate", "/agent/stream")
PROTECTED_PREFIXES = ("/v1/",)
BEARER_PREFIX      = "Bearer "


# ------------------------------------------------------------------------------------
# Functions

def _is_protected(path):
    return path in PROTECTED_EXACT or any(path.startswith(p) for p in PROTECTED_PREFIXES)


def _bearer_token():
    header = request.headers.get("Authorization", "")
    if header.startswith(BEARER_PREFIX):
        return header[len(BEARER_PREFIX):].strip()
    return ""


def register(app):
    @app.before_request
    def _api_key_guard():
        key = settings_mod.get().get("api_key") or ""
        if not key or not _is_protected(request.path):
            return None
        if hmac.compare_digest(_bearer_token(), key):
            settings_mod.record_api_key_use()
            return None
        return ({"ok": False, "error": "invalid or missing api key"}, 401)
