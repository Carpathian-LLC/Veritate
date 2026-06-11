# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Optional dashboard auth: off when no password is set (never locks out), gates
#   /app and keeps the public surface open when a password is configured.
# tests/mri/test_auth.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import pytest

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
MRI_DIR   = os.path.join(REPO_ROOT, "veritate_mri")
for p in (REPO_ROOT, MRI_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from routes import auth_routes


# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture(scope="module")
def client():
    from veritate_mri.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_app_open_when_no_password(client, monkeypatch):
    """With no password configured, the dashboard is reachable (never locks out)."""
    monkeypatch.delenv(auth_routes.PASSWORD_ENV, raising=False)
    r = client.get("/app")
    assert r.status_code == 200


def test_app_gated_when_password_set(client, monkeypatch):
    """With a password set, an unauthenticated GET /app redirects to /login."""
    monkeypatch.setenv(auth_routes.PASSWORD_ENV, "secret123")
    r = client.get("/app")
    assert r.status_code == 302
    assert "/login" in r.headers.get("Location", "")


def test_public_surface_open_when_gated(client, monkeypatch):
    """Landing and chat stay public even when the dashboard is gated."""
    monkeypatch.setenv(auth_routes.PASSWORD_ENV, "secret123")
    assert client.get("/").status_code == 200
    assert client.get("/chat").status_code == 200


def test_login_grants_access(client, monkeypatch):
    """Correct password creates a session that unlocks /app."""
    monkeypatch.setenv(auth_routes.PASSWORD_ENV, "secret123")
    bad = client.post("/login", data={"password": "wrong"})
    assert bad.status_code == 302 and "e=1" in bad.headers.get("Location", "")
    ok = client.post("/login", data={"password": "secret123"})
    assert ok.status_code == 302 and ok.headers.get("Location", "").endswith("/app")
    assert client.get("/app").status_code == 200
