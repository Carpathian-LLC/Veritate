# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - standalone process launcher for the mesh subsystem. for users who want a
#   headless node or hub without the full MRI dashboard.
# - role chosen by argv: `python -m veritate_mesh hub`. when absent, falls back
#   to runtime/settings.py::mesh_role.
# - this entrypoint puts the repo root on sys.path so runtime/* and readers/*
#   resolve regardless of cwd.
# veritate_mesh/__main__.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
MRI_ROOT  = os.path.join(REPO_ROOT, "veritate_mri")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, MRI_ROOT)

from flask import Flask
from runtime import logs as logmod
from runtime import settings as settings_mod

from veritate_mesh import hub as hub_mod
from veritate_mesh import node as node_mod
from veritate_mesh.protocol import ROLE_BOTH, ROLE_HUB, ROLE_NODE, ROLE_OFF, VALID_ROLES

# ------------------------------------------------------------------------------------
# Constants

DEFAULT_PORT_HUB  = 8201
DEFAULT_PORT_NODE = 8101
DEFAULT_BIND_HOST = "0.0.0.0"
# 0 means "pick the port matching the resolved role".
ROLE_DEFAULT_PORT = 0
# Role sentinel meaning "read mesh_role from settings" rather than force one.
ROLE_AUTO = "auto"

SETTING_ROLE = "mesh_role"
LOG_SOURCE = "mesh.main"

# ------------------------------------------------------------------------------------
# Functions

def _build_app(role: str) -> Flask:
    app = Flask(__name__)
    if role in (ROLE_HUB, ROLE_BOTH):
        hub_mod.register(app)
        hub_mod.start_workers()
        logmod.info(LOG_SOURCE, "hub routes registered")
    if role in (ROLE_NODE, ROLE_BOTH):
        node_mod.register(app)
        node_mod.start_workers()
        logmod.info(LOG_SOURCE, "node routes registered")
    return app


def main():
    ap = argparse.ArgumentParser(prog="veritate_mesh")
    ap.add_argument("role", nargs="?", default=ROLE_AUTO,
                    help=f"role: {' / '.join((*VALID_ROLES, ROLE_AUTO))}. "
                         f"{ROLE_AUTO} reads {SETTING_ROLE} from settings.")
    ap.add_argument("--host", default=DEFAULT_BIND_HOST)
    ap.add_argument("--port", type=int, default=ROLE_DEFAULT_PORT,
                    help=f"{ROLE_DEFAULT_PORT} = role default "
                         f"({ROLE_HUB}={DEFAULT_PORT_HUB}, {ROLE_NODE}={DEFAULT_PORT_NODE})")
    args = ap.parse_args()

    cli_role = args.role
    if cli_role and cli_role != ROLE_AUTO:
        if cli_role not in VALID_ROLES:
            raise SystemExit(f"invalid role: {cli_role!r}; expected one of {VALID_ROLES}")
        role = cli_role
    else:
        role = (settings_mod.get().get(SETTING_ROLE) or ROLE_OFF).lower()
    if role == ROLE_OFF:
        raise SystemExit("mesh_role is off; nothing to run. pass `hub`, `node`, or `both`.")
    port = args.port or (DEFAULT_PORT_HUB if role == ROLE_HUB else DEFAULT_PORT_NODE)
    app = _build_app(role)
    print(f"veritate_mesh role={role} http://{args.host}:{port}")
    app.run(host=args.host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
