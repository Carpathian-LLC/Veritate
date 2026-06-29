# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - http surface for the downloadable-extensions system. lists installed
#   extensions, reads the marketplace catalog, installs from the bundled
#   canonical source, uninstalls. the extensions package owns all disk work.
# veritate_mri/routes/extensions_routes.py
# ------------------------------------------------------------------------------------
# Imports:

from flask import request

from extensions import registry
from extensions import data as ext_data

# ------------------------------------------------------------------------------------
# Functions

def register(app):
    @app.route("/extensions")
    def extensions_list():
        return {"ok": True, "extensions": registry.list_installed()}

    @app.route("/extensions/catalog")
    def extensions_catalog():
        return {"ok": True, "catalog": registry.load_catalog()}

    @app.route("/extensions/install", methods=["POST"])
    def extensions_install():
        ext_id = (request.get_json(silent=True) or {}).get("id")
        if not ext_id:
            return ({"ok": False, "error": "id required"}, 400)
        try:
            extension = registry.install(ext_id)
        except ValueError as e:
            return ({"ok": False, "error": str(e)}, 404)
        return {"ok": True, "extension": extension}

    @app.route("/extensions/uninstall", methods=["POST"])
    def extensions_uninstall():
        ext_id = (request.get_json(silent=True) or {}).get("id")
        if not ext_id:
            return ({"ok": False, "error": "id required"}, 400)
        registry.uninstall(ext_id)
        return {"ok": True}

    @app.route("/extensions/<ext_id>/data")
    def extensions_data_catalog(ext_id):
        return {"ok": True, "datasets": ext_data.catalog(ext_id)}

    @app.route("/extensions/<ext_id>/data/download", methods=["POST"])
    def extensions_data_download(ext_id):
        source = (request.get_json(silent=True) or {}).get("source")
        if not source:
            return ({"ok": False, "error": "source required"}, 400)
        return ext_data.download(ext_id, source)

    @app.route("/extensions/<ext_id>/data/delete", methods=["POST"])
    def extensions_data_delete(ext_id):
        source = (request.get_json(silent=True) or {}).get("source")
        if not source:
            return ({"ok": False, "error": "source required"}, 400)
        return ext_data.delete(ext_id, source)
