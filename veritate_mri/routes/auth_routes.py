# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Optional password gate for the dashboard. Auth is OFF unless a password is set
#   (env VERITATE_DASHBOARD_PASSWORD), so a fresh install is never locked out. When
#   enabled, the public surface (chat, static) stays open; the dashboard
#   (/app) and management/training APIs require a session login.
# veritate_mri/routes/auth_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import hmac
import os

from flask import request, session, redirect, send_from_directory

# ------------------------------------------------------------------------------------
# Constants

PASSWORD_ENV = "VERITATE_DASHBOARD_PASSWORD"
SECRET_ENV   = "VERITATE_SECRET_KEY"
# Open without a session even when auth is enabled. Everything else is gated.
PUBLIC_EXACT    = ("/", "/login", "/logout", "/favicon.ico")
PUBLIC_PREFIXES = ("/static", "/chat", "/hybrid")


# ------------------------------------------------------------------------------------
# Functions

def _password():
    return os.environ.get(PASSWORD_ENV, "")


def enabled():
    return bool(_password())


def _is_public(path):
    return path in PUBLIC_EXACT or any(path.startswith(p) for p in PUBLIC_PREFIXES)


def register(app):
    if not app.secret_key:
        app.secret_key = os.environ.get(SECRET_ENV) or os.urandom(32)

    @app.before_request
    def _guard():
        if not enabled() or _is_public(request.path) or session.get("authed"):
            return None
        if request.method == "GET":
            return redirect("/login")
        return ({"ok": False, "error": "authentication required"}, 401)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            given = request.form.get("password") or ""
            if enabled() and hmac.compare_digest(given, _password()):
                session["authed"] = True
                return redirect("/app")
            return redirect("/login?e=1")
        return send_from_directory(app.static_folder, "login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect("/")
