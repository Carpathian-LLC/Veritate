# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - corpus library catalog, install/uninstall, deps, catalog url, user
#   sources add/remove, and open-folder.
# veritate_mri/routes/corpus_routes.py
# ------------------------------------------------------------------------------------
# Imports:

from flask import request

from readers import paths
from training.sync import corpus_sync

from ._common import open_folder

# ------------------------------------------------------------------------------------
# Constants


# ------------------------------------------------------------------------------------
# Functions

def register(app):
    @app.route("/corpus/library/catalog")
    def corpus_library_catalog():
        return corpus_sync.catalog()

    @app.route("/corpus/library/install", methods=["POST"])
    def corpus_library_install():
        body = request.get_json(silent=True) or {}
        return corpus_sync.install(body)

    @app.route("/corpus/library/install_deps", methods=["POST"])
    def corpus_library_install_deps():
        """Run pip install -r requirements.txt in a subprocess so import
        datasets succeeds without a server restart."""
        return corpus_sync.install_hf_deps()

    @app.route("/corpus/library/uninstall", methods=["POST"])
    def corpus_library_uninstall():
        body = request.get_json(silent=True) or {}
        return corpus_sync.uninstall(body.get("stem"))

    @app.route("/corpus/library/catalog_url", methods=["POST"])
    def corpus_library_catalog_url():
        body = request.get_json(silent=True) or {}
        return corpus_sync.set_catalog_url(body.get("url"))

    @app.route("/corpus/library/sources/add", methods=["POST"])
    def corpus_library_sources_add():
        body = request.get_json(silent=True) or {}
        return corpus_sync.add_user_source(body)

    @app.route("/corpus/library/sources/remove", methods=["POST"])
    def corpus_library_sources_remove():
        body = request.get_json(silent=True) or {}
        return corpus_sync.remove_user_source(body.get("stem"))

    @app.route("/corpus/open_folder", methods=["POST"])
    def corpus_open_folder():
        return open_folder(paths.CORPUS_ROOT)
