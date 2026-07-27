# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - corpus library catalog, install/uninstall, deps, catalog url, user
#   sources add/remove, open-folder, and the mix planner.
# - /corpus/mix/plan is the only validating layer in front of mix_planner: the
#   planner trusts its callers, so every body field is checked here.
# veritate_mri/routes/corpus_routes.py
# ------------------------------------------------------------------------------------
# Imports:

from flask import request

from readers import paths
from training import mix_planner
from training.sync import corpus_sync

from ._common import open_folder

# ------------------------------------------------------------------------------------
# Constants


# ------------------------------------------------------------------------------------
# Functions

def _positive_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def _plan_error(body):
    """First validation failure in a /corpus/mix/plan body, or None."""
    stems = body.get("stems")
    if not isinstance(stems, list) or not stems:
        return "stems must be a non-empty list of corpus stems"
    if not all(isinstance(s, str) and s.strip() for s in stems):
        return "every stem must be a non-empty string"
    if len(set(stems)) != len(stems):
        return "stems must be unique"
    if not _positive_number(body.get("target_bytes")):
        return "target_bytes must be a positive number"
    for key in ("max_epochs", "model_params"):
        if body.get(key) is not None and not _positive_number(body[key]):
            return f"{key} must be a positive number"
    if body.get("profile") is not None and not isinstance(body["profile"], str):
        return "profile must be a string"
    weights = body.get("weights")
    if weights is not None:
        if not isinstance(weights, dict):
            return "weights must be an object of stem -> weight"
        if any(s not in weights for s in stems):
            return "weights must cover every stem"
        if not all(_positive_number(weights[s]) for s in stems):
            return "every weight must be a positive number"
    return None


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

    @app.route("/corpus/mix/profiles")
    def corpus_mix_profiles():
        try:
            profiles = mix_planner.load_profiles()
        except (ValueError, KeyError, OSError) as e:
            return ({"ok": False, "error": str(e)}, 400)
        return {"ok": True, "path": mix_planner.profiles_path(), "profiles": [
            {"name": name, "label": prof.get("label") or name,
             "topics": prof.get("topics") or {}, "stems": prof.get("stems") or []}
            for name, prof in sorted(profiles.items())]}

    @app.route("/corpus/mix/plan", methods=["POST"])
    def corpus_mix_plan():
        body = request.get_json(silent=True) or {}
        error = _plan_error(body)
        if error:
            return ({"ok": False, "error": error}, 400)
        try:
            return mix_planner.plan(
                body["stems"], body["target_bytes"],
                profile=body.get("profile") or None, max_epochs=body.get("max_epochs"),
                model_params=body.get("model_params"), weights=body.get("weights"))
        except (ValueError, KeyError, OSError) as e:
            return ({"ok": False, "error": str(e)}, 400)

    @app.route("/corpus/open_folder", methods=["POST"])
    def corpus_open_folder():
        return open_folder(paths.CORPUS_ROOT)
