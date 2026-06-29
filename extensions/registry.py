# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - owns the extension lifecycle: discover installed manifests, register their
#   pages + server routes onto the flask app, read the marketplace catalog,
#   install from the bundled canonical source, uninstall.
# - one failing extension is logged and skipped; it never aborts the others or
#   server startup.
# extensions/registry.py
# ------------------------------------------------------------------------------------
# Imports:

import importlib.util
import json
import os
import shutil
import sys

from flask import request, send_from_directory

from runtime import logs as logmod

# ------------------------------------------------------------------------------------
# Constants

EXTENSIONS_ROOT = os.path.dirname(os.path.abspath(__file__))
INSTALLED_ROOT  = os.path.join(EXTENSIONS_ROOT, "installed")
CANONICAL_ROOT  = os.path.join(EXTENSIONS_ROOT, "canonical")
CATALOG_PATH    = os.path.join(EXTENSIONS_ROOT, "catalog.json")
DISABLED_PATH   = os.path.join(EXTENSIONS_ROOT, "disabled.json")

LOG_SOURCE  = "extensions"
MANIFEST    = "manifest.json"
SERVER_DIR  = "server"
REGISTER_FN = "register"

_OWNED = {}                               # url rule string -> ext id, for the live disable gate

# ------------------------------------------------------------------------------------
# Functions

def _disabled():
    try:
        with open(DISABLED_PATH, "r", encoding="utf-8") as f:
            return set(json.load(f) or [])
    except (OSError, ValueError):
        return set()


def _write_disabled(ids):
    with open(DISABLED_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(ids), f)


def discover():
    """Active extensions, from both roots. Canonical (first-party, shipped, trusted) and installed/
    (downloaded third-party, overrides a canonical of the same id) are both active, EXCEPT ids the
    user has uninstalled (recorded in disabled.json) which are filtered out. One unreadable manifest
    is logged and skipped."""
    found = {}
    for root, source in ((CANONICAL_ROOT, "canonical"), (INSTALLED_ROOT, "installed")):
        if not os.path.isdir(root):
            continue
        for ext_id in sorted(os.listdir(root)):
            ext_dir = os.path.join(root, ext_id)
            manifest_path = os.path.join(ext_dir, MANIFEST)
            if not os.path.isfile(manifest_path):
                continue
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except (OSError, ValueError) as e:
                logmod.error(LOG_SOURCE, f"manifest read failed for {ext_id}: {e}")
                continue
            manifest["_dir"] = ext_dir
            manifest["_source"] = source
            found[manifest.get("id", ext_id)] = manifest
    disabled = _disabled()
    return [m for ext_id, m in found.items() if ext_id not in disabled]


def manifest_for(ext_id):
    return next((m for m in discover() if m.get("id") == ext_id), None)


def _register_one(app, manifest):
    ext_dir = manifest["_dir"]
    before = {r.rule for r in app.url_map.iter_rules()}
    server_dir = os.path.join(ext_dir, SERVER_DIR)
    if os.path.isdir(server_dir) and server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    register_rel = manifest.get(REGISTER_FN)
    if register_rel:
        register_path = os.path.join(ext_dir, register_rel)
        spec = importlib.util.spec_from_file_location(f"ext_{manifest['id']}_register", register_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.register(app)
    page = manifest.get("page") or {}
    route = page.get("route")
    page_file = page.get("file")
    if route and page_file:
        endpoint = f"ext_page_{manifest['id']}"
        app.add_url_rule(route, endpoint, _page_view(ext_dir, page_file))
    for rule in app.url_map.iter_rules():
        if rule.rule not in before:
            _OWNED[rule.rule] = manifest["id"]


def _gate():
    """Live disable: 404 any request whose matched route belongs to an uninstalled extension, so
    uninstall/reinstall take effect without a server restart (Flask routes cannot be unregistered)."""
    rule = request.url_rule
    if rule is None:
        return None
    ext_id = _OWNED.get(rule.rule)
    if ext_id is not None and ext_id in _disabled():
        return ("extension uninstalled", 404)
    return None


def _page_view(ext_dir, page_file):
    directory = os.path.join(ext_dir, os.path.dirname(page_file))
    filename = os.path.basename(page_file)
    return lambda: send_from_directory(directory, filename)


def register_all(app):
    app.before_request(_gate)
    registered = []
    for manifest in discover():
        try:
            _register_one(app, manifest)
            registered.append(manifest)
            logmod.ok(LOG_SOURCE, f"registered {manifest['id']}")
        except Exception as e:
            logmod.error(LOG_SOURCE, f"register failed for {manifest.get('id')}: {type(e).__name__}: {e}")
    return registered


def list_installed():
    out = []
    for manifest in discover():
        page = manifest.get("page") or {}
        out.append({
            "id":           manifest.get("id"),
            "name":         manifest.get("name"),
            "version":      manifest.get("version"),
            "nav_label":    page.get("nav_label"),
            "route":        page.get("route"),
            "experimental": manifest.get("experimental", False),
        })
    return out


def load_catalog():
    active = {m.get("id") for m in discover()}
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    entries = catalog.get("extensions") or []
    for entry in entries:
        entry["installed"] = entry.get("id") in active
    return entries


def install(ext_id):
    """Activate an extension: clear any disabled flag, and for a builtin copy its canonical source
    into installed/ (re-adding the code beside a kept data/ cache). Third-party extensions already
    present in installed/ just get re-enabled."""
    disabled = _disabled()
    if ext_id in disabled:
        disabled.discard(ext_id)
        _write_disabled(disabled)
    src = os.path.join(CANONICAL_ROOT, ext_id)
    if os.path.isdir(src):
        dst = os.path.join(INSTALLED_ROOT, ext_id)
        os.makedirs(INSTALLED_ROOT, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif not os.path.isdir(os.path.join(INSTALLED_ROOT, ext_id)):
        raise ValueError(f"no source for extension {ext_id!r}")
    logmod.ok(LOG_SOURCE, f"installed {ext_id}")
    return {"id": ext_id, "installed": True}


def uninstall(ext_id):
    """Deactivate an extension (recorded in disabled.json so it stays off, even a canonical builtin
    that cannot be physically deleted) and remove any installed/ code, preserving its data/ cache.
    A reinstall clears the flag and re-adds the code beside the kept data."""
    disabled = _disabled()
    disabled.add(ext_id)
    _write_disabled(disabled)
    dst = os.path.join(INSTALLED_ROOT, ext_id)
    if os.path.isdir(os.path.join(dst, "data")):
        for name in os.listdir(dst):
            if name == "data":
                continue
            p = os.path.join(dst, name)
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
    elif os.path.isdir(dst):
        shutil.rmtree(dst, ignore_errors=True)
    logmod.ok(LOG_SOURCE, f"uninstalled {ext_id}")
    return {"id": ext_id, "installed": False}
