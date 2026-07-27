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

import threading

from flask import request
from readers import paths
from runtime import logs as logmod
from training import mix_planner
from training.sync import corpus_sync

from ._common import open_folder

# ------------------------------------------------------------------------------------
# Constants

LOG_SOURCE = "corpus"

STEM_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_")

_COMPOSE_LOCK = threading.Lock()
_COMPOSE = {"state": "idle"}


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

    @app.route("/corpus/mix/compose/status")
    def corpus_mix_compose_status():
        with _COMPOSE_LOCK:
            return dict(_COMPOSE)

    @app.route("/corpus/mix/compose", methods=["POST"])
    def corpus_mix_compose():
        """Materialize a plan into ONE unified corpus on disk. Runs in a thread
        because a chinchilla-scale compose moves gigabytes; poll
        /corpus/mix/compose/status for byte-level progress."""
        body = request.get_json(silent=True) or {}
        stem = (body.get("stem") or "").strip().lower()
        if not stem or set(stem) - STEM_CHARS:
            return ({"ok": False, "error": "stem must be lowercase letters, digits, underscores"}, 400)
        error = _plan_error(body)
        if error:
            return ({"ok": False, "error": error}, 400)
        for key, low, high in (("val_ratio", 0, 1), ("chunk_bytes", 1, None), ("seed", 0, None)):
            v = body.get(key)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return ({"ok": False, "error": f"{key} must be a number"}, 400)
            if v < low or (high is not None and v >= high):
                return ({"ok": False, "error": f"{key} is out of range"}, 400)
        with _COMPOSE_LOCK:
            if _COMPOSE.get("state") == "running":
                return ({"ok": False, "error": "a compose is already running"}, 409)
            _COMPOSE.clear()
            _COMPOSE.update({"state": "running", "stem": stem, "bytes_written": 0,
                             "bytes_total": 0, "phase": "planning"})

        def _run():
            try:
                plan_result = mix_planner.plan(
                    body["stems"], body["target_bytes"], profile=body.get("profile") or None,
                    max_epochs=body.get("max_epochs"), model_params=body.get("model_params"),
                    weights=body.get("weights"))

                def _progress(written, total):
                    with _COMPOSE_LOCK:
                        _COMPOSE.update({"bytes_written": written, "bytes_total": total,
                                         "phase": "writing train"})

                result = mix_planner.compose(
                    stem, plan_result, val_ratio=body.get("val_ratio"),
                    seed=body.get("seed"), chunk_bytes=body.get("chunk_bytes"),
                    progress=_progress)
                logmod.ok(LOG_SOURCE, f"composed corpus: stem={stem} "
                                      f"train={result['train_bytes']}B val={result['val_bytes']}B")
                with _COMPOSE_LOCK:
                    _COMPOSE.clear()
                    _COMPOSE.update({"state": "done", "stem": stem, "result": result})
            except (ValueError, KeyError, OSError) as e:
                logmod.warn(LOG_SOURCE, f"compose failed: stem={stem}: {type(e).__name__}: {e}")
                with _COMPOSE_LOCK:
                    _COMPOSE.clear()
                    _COMPOSE.update({"state": "error", "stem": stem, "error": str(e)})

        threading.Thread(target=_run, name=f"corpus-compose-{stem}", daemon=True).start()
        return {"ok": True, "stem": stem, "state": "running",
                "poll": "/corpus/mix/compose/status"}

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
