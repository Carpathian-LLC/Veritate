# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Contract tests for the extension install/uninstall lifecycle: uninstall disables an extension
#   (canonical or installed) via disabled.json so discover() drops it, install re-enables it, and
#   uninstall preserves the data/ cache. Deterministic, no network, temp roots.
# extensions/tests/test_registry.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(REPO, "veritate_mri"), REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from extensions import registry

# ------------------------------------------------------------------------------------
# Helpers

_LOG = types.SimpleNamespace(ok=lambda *a: None, error=lambda *a: None, info=lambda *a: None)


def _roots(tmp_path, monkeypatch):
    can = str(tmp_path / "canonical")
    inst = str(tmp_path / "installed")
    os.makedirs(can)
    os.makedirs(inst)
    monkeypatch.setattr(registry, "CANONICAL_ROOT", can)
    monkeypatch.setattr(registry, "INSTALLED_ROOT", inst)
    monkeypatch.setattr(registry, "DISABLED_PATH", str(tmp_path / "disabled.json"))
    monkeypatch.setattr(registry, "logmod", _LOG)
    return can, inst


def _mk(root, ext_id):
    d = os.path.join(root, ext_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"id": ext_id, "name": ext_id}, f)
    return d

# ------------------------------------------------------------------------------------
# Tests

def test_uninstall_disables_a_builtin_then_install_reenables(tmp_path, monkeypatch):
    """uninstall records the id in disabled.json so discover() drops a canonical builtin; install clears it and the extension returns."""
    can, _ = _roots(tmp_path, monkeypatch)
    _mk(can, "foo")
    assert [m["id"] for m in registry.discover()] == ["foo"]
    registry.uninstall("foo")
    assert registry.discover() == []
    registry.install("foo")
    assert [m["id"] for m in registry.discover()] == ["foo"]


def test_uninstall_preserves_data_cache(tmp_path, monkeypatch):
    """uninstall removes installed code but keeps the data/ cache for a later reinstall."""
    _, inst = _roots(tmp_path, monkeypatch)
    d = os.path.join(inst, "foo")
    os.makedirs(os.path.join(d, "data"))
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        f.write("{}")
    with open(os.path.join(d, "data", "keep.csv"), "w", encoding="utf-8") as f:
        f.write("x")
    registry.uninstall("foo")
    assert os.path.isfile(os.path.join(d, "data", "keep.csv"))
    assert not os.path.isfile(os.path.join(d, "manifest.json"))


def test_uninstall_404s_live_routes_without_restart(tmp_path, monkeypatch):
    """the before_request gate 404s a disabled extension's already-registered routes, and reinstall restores them, with no re-registration or restart."""
    from flask import Flask
    can, _ = _roots(tmp_path, monkeypatch)
    d = _mk(can, "foo")
    os.makedirs(os.path.join(d, "server"))
    with open(os.path.join(d, "server", "reg.py"), "w", encoding="utf-8") as f:
        f.write("def register(app):\n    @app.route('/ext/foo/ping')\n    def _ping():\n        return 'pong'\n")
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"id": "foo", "name": "foo", "register": "server/reg.py"}, f)
    app = Flask(__name__)
    registry.register_all(app)
    client = app.test_client()
    assert client.get("/ext/foo/ping").status_code == 200
    registry.uninstall("foo")
    assert client.get("/ext/foo/ping").status_code == 404
    registry.install("foo")
    assert client.get("/ext/foo/ping").status_code == 200
