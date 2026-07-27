# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - route tests for the extension lifecycle over /extensions, /extensions/install
#   and /extensions/uninstall. the canonical, installed and disabled roots are all
#   redirected into tmp_path, so the real extensions/installed and disabled.json
#   are never touched.
# - the fixture extension starts disabled (the state a fresh install acts on) and
#   carries no data/ cache, so uninstall must leave nothing on disk.
# tests/mri/test_extensions_lifecycle.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import types

import pytest
from flask import Flask
from routes import extensions_routes

from extensions import registry

# ------------------------------------------------------------------------------------
# Constants

EXT_ID     = "fixture_ext"
UNKNOWN_ID = "absent_ext"
PAGE_HTML  = "<p>fixture</p>"

MANIFEST = {
    "id":          EXT_ID,
    "name":        "Fixture",
    "version":     "0.1.0",
    "author":      "tests",
    "kind":        "extension",
    "description": "Minimal page-only extension used by the lifecycle tests.",
    "page": {"route": f"/ext/{EXT_ID}", "file": "page/index.html", "nav_label": "Fixture"},
}

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture
def ext(monkeypatch, tmp_path):
    """Test client for /extensions with temp roots and one not-yet-installed fixture extension."""
    canonical = tmp_path / "canonical"
    installed = tmp_path / "installed"
    disabled  = tmp_path / "disabled.json"
    page_dir  = canonical / EXT_ID / "page"
    page_dir.mkdir(parents=True)
    (page_dir / "index.html").write_text(PAGE_HTML, encoding="utf-8")
    (canonical / EXT_ID / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    installed.mkdir()
    disabled.write_text(json.dumps([EXT_ID]), encoding="utf-8")
    monkeypatch.setattr(registry, "CANONICAL_ROOT", str(canonical))
    monkeypatch.setattr(registry, "INSTALLED_ROOT", str(installed))
    monkeypatch.setattr(registry, "DISABLED_PATH", str(disabled))
    app = Flask(__name__)
    extensions_routes.register(app)
    return types.SimpleNamespace(client=app.test_client(), installed=installed)


def _listed_ids(ext):
    return [e["id"] for e in ext.client.get("/extensions").get_json()["extensions"]]


def _install(ext, ext_id=EXT_ID):
    return ext.client.post("/extensions/install", json={"id": ext_id})


def _uninstall(ext, ext_id=EXT_ID):
    return ext.client.post("/extensions/uninstall", json={"id": ext_id})


def test_not_installed_extension_is_absent(ext):
    """GET /extensions omits an extension that has not been installed."""
    assert EXT_ID not in _listed_ids(ext)


def test_install_returns_200(ext):
    """POST /extensions/install with a known id returns 200."""
    assert _install(ext).status_code == 200


def test_install_lists_the_extension(ext):
    """After install the extension appears in GET /extensions."""
    _install(ext)
    assert EXT_ID in _listed_ids(ext)


def test_install_copies_the_extension_to_the_install_root(ext):
    """Install copies the canonical source into the install root."""
    _install(ext)
    assert os.path.isfile(str(ext.installed / EXT_ID / "manifest.json"))


def test_uninstall_delists_the_extension(ext):
    """After uninstall the extension is gone from GET /extensions."""
    _install(ext)
    _uninstall(ext)
    assert EXT_ID not in _listed_ids(ext)


def test_uninstall_leaves_no_directory_on_disk(ext):
    """Uninstall removes the installed directory outright, leaving no residue."""
    _install(ext)
    _uninstall(ext)
    assert not os.path.exists(str(ext.installed / EXT_ID))


def test_reinstall_after_uninstall_lists_the_extension(ext):
    """A reinstall clears the disabled flag and the extension is listed again."""
    _install(ext)
    _uninstall(ext)
    _install(ext)
    assert EXT_ID in _listed_ids(ext)


def test_install_unknown_id_returns_404(ext):
    """POST /extensions/install with an id that has no source returns 404."""
    assert _install(ext, UNKNOWN_ID).status_code == 404


def test_install_without_id_returns_400(ext):
    """POST /extensions/install with no id returns 400."""
    assert ext.client.post("/extensions/install", json={}).status_code == 400


def test_uninstall_without_id_returns_400(ext):
    """POST /extensions/uninstall with no id returns 400."""
    assert ext.client.post("/extensions/uninstall", json={}).status_code == 400
