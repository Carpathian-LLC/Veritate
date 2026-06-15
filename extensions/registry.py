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

from flask import send_from_directory

from runtime import logs as logmod

# ------------------------------------------------------------------------------------
# Constants

EXTENSIONS_ROOT = os.path.dirname(os.path.abspath(__file__))
INSTALLED_ROOT  = os.path.join(EXTENSIONS_ROOT, "installed")
CANONICAL_ROOT  = os.path.join(EXTENSIONS_ROOT, "canonical")
CATALOG_PATH    = os.path.join(EXTENSIONS_ROOT, "catalog.json")

LOG_SOURCE  = "extensions"
MANIFEST    = "manifest.json"
SERVER_DIR  = "server"
REGISTER_FN = "register"

# ------------------------------------------------------------------------------------
# Functions

def discover():
    """Installed extensions, from both roots. Canonical (first-party, shipped, trusted)
    is always active; installed/ holds downloaded third-party ones and overrides a
    canonical of the same id. One unreadable manifest is logged and skipped."""
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
    return list(found.values())


def manifest_for(ext_id):
    return next((m for m in discover() if m.get("id") == ext_id), None)


def _register_one(app, manifest):
    ext_dir = manifest["_dir"]
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


def _page_view(ext_dir, page_file):
    directory = os.path.join(ext_dir, os.path.dirname(page_file))
    filename = os.path.basename(page_file)
    return lambda: send_from_directory(directory, filename)


def register_all(app):
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
    src = os.path.join(CANONICAL_ROOT, ext_id)
    if not os.path.isdir(src):
        raise ValueError(f"no canonical source for extension {ext_id!r}")
    dst = os.path.join(INSTALLED_ROOT, ext_id)
    os.makedirs(INSTALLED_ROOT, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    logmod.ok(LOG_SOURCE, f"installed {ext_id}")
    return {"id": ext_id, "installed": True}


def uninstall(ext_id):
    """Remove the extension's code but preserve its data/ cache (downloaded datasets,
    recorded history). A reinstall re-adds the code beside the kept data."""
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
    else:
        shutil.rmtree(dst, ignore_errors=True)
    logmod.ok(LOG_SOURCE, f"uninstalled {ext_id}")
    return {"id": ext_id, "installed": False}
