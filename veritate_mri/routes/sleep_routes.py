# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - sleep consolidation endpoints. Routes proxy to training.sleep; the
#   Generation-tab sleep panel polls /sleep and posts per-model /sleep/now and
#   /sleep/wake. Both POSTs take a "model" (JSON body or query); omitted, the
#   only enrolled model is assumed, and with several enrolled the request is a
#   400. /sleep/now bypasses only the idle gate (still requires the feature
#   enabled, an enrolled model, and an idle trainer).
# veritate_mri/routes/sleep_routes.py
# ------------------------------------------------------------------------------------
# Imports:

from flask import request
from runtime import settings as settings_mod
from training import sleep

from routes._common import safe_route as _safe

# ------------------------------------------------------------------------------------
# Constants


# ------------------------------------------------------------------------------------
# Functions

def _model_param():
    """(model, error): the request's model, defaulting to the only enrolled one."""
    body = request.get_json(silent=True) or {}
    m = str(body.get("model") or request.args.get("model") or "").strip()
    if m:
        return m, None
    names = sleep.enrolled(settings_mod.get())
    if len(names) == 1:
        return names[0], None
    return None, ("no model enrolled in sleep_models" if not names
                  else f"several models are enrolled ({', '.join(names)}); pass \"model\"")


def register(app):
    @app.route("/sleep")
    def sleep_status():
        return _safe("sleep", sleep.status)

    @app.route("/sleep/wake", methods=["POST"])
    def sleep_wake():
        model, err = _model_param()
        if err:
            return {"ok": False, "error": err}, 400
        return _safe("sleep", sleep.wake, model)

    @app.route("/sleep/now", methods=["POST"])
    def sleep_now():
        model, err = _model_param()
        if err:
            return {"ok": False, "error": err}, 400
        return _safe("sleep", lambda: {"ok": True,
                                       "result": sleep.maybe_sleep(force_idle=True, model=model)})
