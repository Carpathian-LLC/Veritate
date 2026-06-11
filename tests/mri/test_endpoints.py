# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Smoke tests for the MRI Flask app. Uses Flask's test_client so no live
#   server is started. Verifies each read-only endpoint responds 200 with the
#   expected content-type. Mutating endpoints (POST) are not exercised here
#   to keep tests deterministic and side-effect-free.
# tests/mri/test_endpoints.py
# ------------------------------------------------------------------------------------
# Imports

import sys
import os

import pytest

# ------------------------------------------------------------------------------------
# Constants

REPO_ROOT  = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MRI_DIR    = os.path.join(REPO_ROOT, "veritate_mri")

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture(scope="module")
def client():
    """Flask test client for the MRI app. Module-scoped because import side
    effects (path setup, heartbeat thread spawn) are non-trivial."""
    if MRI_DIR not in sys.path:
        sys.path.insert(0, MRI_DIR)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from veritate_mri.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_root_serves_chat(client):
    """GET / returns 200 and the public chat front door."""
    r = client.get("/")
    assert r.status_code == 200
    assert b"/hybrid/chat" in r.data


def test_app_serves_dashboard(client):
    """GET /app returns 200 and the dashboard HTML payload."""
    r = client.get("/app")
    assert r.status_code == 200
    assert b"<html" in r.data.lower() or b"<!doctype" in r.data.lower()


def test_sys_metrics_returns_json(client):
    """GET /sys_metrics returns 200 + JSON. Powers the dashboard's resource bar."""
    r = client.get("/sys_metrics")
    assert r.status_code == 200
    assert "json" in r.content_type.lower()


def test_versions_returns_json(client):
    """GET /versions returns the versions.json contents as JSON."""
    r = client.get("/versions")
    assert r.status_code == 200
    assert "json" in r.content_type.lower()
    body = r.get_json()
    # versions.json is the source of truth; minimal shape check.
    assert isinstance(body, dict)


def test_heartbeat_status_returns_json(client):
    """GET /heartbeat/status returns the heartbeat thread's last known state."""
    r = client.get("/heartbeat/status")
    assert r.status_code == 200
    assert "json" in r.content_type.lower()


def test_engine_status_returns_json(client):
    """GET /engine/status reports whether the engine binary is built and reachable."""
    r = client.get("/engine/status")
    assert r.status_code == 200
    assert "json" in r.content_type.lower()


def test_plugins_endpoint_returns_json_list(client):
    """GET /trainers returns the list of installed training trainers."""
    r = client.get("/trainers")
    assert r.status_code == 200
    assert "json" in r.content_type.lower()


def test_settings_get_returns_json(client):
    """GET /settings returns the current MRI settings as JSON."""
    r = client.get("/settings")
    assert r.status_code == 200
    assert "json" in r.content_type.lower()


def test_logs_snapshot_returns_text(client):
    """GET /logs/snapshot returns the most recent log lines."""
    r = client.get("/logs/snapshot")
    assert r.status_code == 200


def test_unknown_endpoint_returns_404(client):
    """GET /no_such_route does NOT 200. Confirms the route table isn't a black hole."""
    r = client.get("/no_such_route_should_not_exist")
    assert r.status_code == 404
