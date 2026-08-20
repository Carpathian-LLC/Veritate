# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - sleep consolidation endpoints. Routes proxy to training.sleep; the
#   Generation-tab sleep panel polls /sleep and posts /sleep/wake.
#   /sleep/now bypasses the idle gate for a manual nap (still requires the
#   feature enabled, a configured model, and an idle trainer).
# veritate_mri/routes/sleep_routes.py
# ------------------------------------------------------------------------------------
# Imports:

from training import sleep

from routes._common import safe_route as _safe

# ------------------------------------------------------------------------------------
# Constants


# ------------------------------------------------------------------------------------
# Functions

def register(app):
    @app.route("/sleep")
    def sleep_status():
        return _safe("sleep", sleep.status)

    @app.route("/sleep/wake", methods=["POST"])
    def sleep_wake():
        return _safe("sleep", sleep.wake)

    @app.route("/sleep/now", methods=["POST"])
    def sleep_now():
        return _safe("sleep", lambda: {"ok": True, "result": sleep.maybe_sleep(force_idle=True)})
