# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - route tests for /generate input validation + the loopback path-param gate.
#   a minimal Flask app registers backends_routes; no model loads (rule 33) since
#   the assertions hit the param-parse and security gates before any backend runs.
# tests/mri/test_generate_route.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(REPO_ROOT, "veritate_mri") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "veritate_mri"))

from flask import Flask

from routes import backends_routes

# ------------------------------------------------------------------------------------
# Constants

LAN = {"REMOTE_ADDR": "192.168.0.99"}
LOCAL = {"REMOTE_ADDR": "127.0.0.1"}

# ------------------------------------------------------------------------------------
# Functions

def _client(brain=None, c_subprocess=None):
    app = Flask(__name__)
    app.config["BRAIN"] = brain
    app.config["C_SUBPROCESS"] = c_subprocess
    backends_routes.register(app)
    return app.test_client()


def test_malformed_param_returns_sse_error_not_500():
    """A non-numeric temperature yields a text/event-stream error frame, not a 500."""
    r = _client().get("/generate?backend=pytorch&temperature=abc")
    assert r.status_code == 200
    assert "text/event-stream" in r.content_type
    assert '"kind": "error"' in r.get_data(as_text=True)


def test_backend_not_loaded_yields_sse_error():
    """An unloaded pytorch backend returns an SSE error frame, not an exception."""
    r = _client(brain=None).get("/generate?backend=pytorch&prompt=hi")
    assert r.status_code == 200
    assert '"kind": "error"' in r.get_data(as_text=True)


def test_rag_path_from_lan_is_forbidden():
    """A rag= filesystem path from a non-loopback client is rejected with 403."""
    r = _client(brain=object()).get("/generate?backend=pytorch&prompt=x&rag=/etc/passwd",
                                    environ_base=LAN)
    assert r.status_code == 403


def test_rag_path_from_loopback_passes_the_gate():
    """The same rag= path from loopback passes the security gate (not 403)."""
    r = _client(brain=object()).get("/generate?backend=pytorch&prompt=x&rag=/nonexistent-xyz",
                                    environ_base=LOCAL)
    assert r.status_code != 403


def test_no_path_param_from_lan_is_not_gated():
    """A plain generate with no path param is not blocked for a LAN client."""
    r = _client(brain=None).get("/generate?backend=pytorch&prompt=hi", environ_base=LAN)
    assert r.status_code != 403
