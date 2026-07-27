# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - dashboard password gate: gated GET redirects to /login, gated non-GET 401s, and
#   the public surface stays open. public cases are parametrized off the module's
#   own PUBLIC_EXACT / PUBLIC_PREFIXES so a newly added public path is covered.
# - the gate is env-driven; monkeypatch.setenv/delenv restores it on teardown, so
#   test ordering cannot leak an enabled or disabled gate into another module.
# tests/mri/test_auth_guard.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
from flask import Flask
from routes import auth_routes

# ------------------------------------------------------------------------------------
# Constants

PASSWORD    = "dashboard-test-password"
LOGIN_PATH  = "/login"
GATED_PATH  = "/app"

PROBE_SUFFIX = "/probe"
STATIC_DIR   = "static"
LOGIN_FILE   = "login.html"
PROBE_FILE   = "probe"

PUBLIC_PATHS = auth_routes.PUBLIC_EXACT + tuple(p + PROBE_SUFFIX for p in auth_routes.PUBLIC_PREFIXES)

REDIRECT_STATUSES = (301, 302, 303, 307, 308)
HTTP_UNAUTHORIZED = 401

# ------------------------------------------------------------------------------------
# Functions

def _ok(**_kwargs):
    return {"ok": True}


def _redirects_to_login(resp):
    return resp.status_code in REDIRECT_STATUSES and resp.headers.get("Location", "").endswith(LOGIN_PATH)


def _app(tmp_path):
    static = tmp_path / STATIC_DIR
    static.mkdir()
    (static / LOGIN_FILE).write_text("login", encoding="utf-8")
    (static / PROBE_FILE).write_text("probe", encoding="utf-8")
    app = Flask(__name__, static_folder=str(static))
    auth_routes.register(app)
    app.add_url_rule("/", "root", _ok)
    app.add_url_rule("/<path:rest>", "catchall", _ok, methods=("GET", "POST"))
    return app.test_client()


@pytest.fixture
def enabled_client(monkeypatch, tmp_path):
    """Dashboard app with the password gate enabled."""
    monkeypatch.setenv(auth_routes.PASSWORD_ENV, PASSWORD)
    return _app(tmp_path)


@pytest.fixture
def disabled_client(monkeypatch, tmp_path):
    """Dashboard app with no password set, so the gate is off."""
    monkeypatch.delenv(auth_routes.PASSWORD_ENV, raising=False)
    return _app(tmp_path)


def test_gate_enabled_when_password_set(monkeypatch):
    """enabled() reports True once the dashboard password env var is set."""
    monkeypatch.setenv(auth_routes.PASSWORD_ENV, PASSWORD)
    assert auth_routes.enabled() is True


def test_unauthenticated_gated_get_redirects_to_login(enabled_client):
    """An unauthenticated GET of a non-public path redirects to /login."""
    assert _redirects_to_login(enabled_client.get(GATED_PATH))


def test_unauthenticated_gated_post_returns_401(enabled_client):
    """An unauthenticated POST of a non-public path returns 401 rather than a redirect."""
    assert enabled_client.post(GATED_PATH).status_code == HTTP_UNAUTHORIZED


@pytest.mark.parametrize("path", PUBLIC_PATHS)
def test_public_path_not_redirected(enabled_client, path):
    """Every path the module treats as public stays open with the gate enabled."""
    assert not _redirects_to_login(enabled_client.get(path))


def test_gated_path_open_when_auth_disabled(disabled_client):
    """With no password set, a non-public path is not redirected."""
    assert not _redirects_to_login(disabled_client.get(GATED_PATH))


def test_authenticated_session_reaches_gated_path(enabled_client):
    """A session that logged in with the correct password reaches a gated path."""
    enabled_client.post(LOGIN_PATH, data={"password": PASSWORD})
    assert not _redirects_to_login(enabled_client.get(GATED_PATH))
