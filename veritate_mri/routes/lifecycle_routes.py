# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - server lifecycle endpoints. routes proxy to runtime.lifecycle.
# veritate_mri/routes/lifecycle_routes.py
# ------------------------------------------------------------------------------------
# Imports:

from flask import current_app

from runtime import lifecycle

# ------------------------------------------------------------------------------------
# Constants


# ------------------------------------------------------------------------------------
# Functions

def register(app):
    @app.route("/lifecycle/restart", methods=["POST"])
    def lifecycle_restart():
        return lifecycle.restart(current_app.config)

    @app.route("/lifecycle/kill", methods=["POST"])
    def lifecycle_kill():
        return lifecycle.kill(current_app.config)

    @app.route("/lifecycle/soft_reload", methods=["POST"])
    def lifecycle_soft_reload():
        return lifecycle.soft_reload(current_app.config)
