# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - flask test client over every cold read-only endpoint. no live server, no
#   network. mutating POSTs are not exercised here; backend load/unload happens
#   in check_backend_*.
# tests/selftest/checks/check_mri_endpoints.py
# ------------------------------------------------------------------------------------
# Imports

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA            = "mri"

ENDPOINTS = (
    ("/",                       "html"),
    ("/sys_metrics",            "json"),
    ("/sys/specs",              "json"),
    ("/versions",               "json"),
    ("/heartbeat/status",       "json"),
    ("/engine/status",          "json"),
    ("/trainers",               "json"),
    ("/core_trainers",          "json"),
    ("/settings",               "json"),
    ("/settings/notices",       "json"),
    ("/logs/snapshot",          "any"),
    ("/wiki",                   "json"),
    ("/meta",                   "json"),
    ("/c-engines",              "json"),
    ("/c-models",               "json"),
    ("/pytorch-models",         "json"),
    ("/backends",               "json"),
    ("/runs",                   "json"),
    ("/addons",                 "json"),
    ("/app/local_edits",        "json"),
    ("/no_such_route_404",      "404"),
)

OK_STATUS  = 200
NOT_FOUND  = 404

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """every cold-state read-only endpoint responds 200 with the expected
    content-type (or 404 for the negative case)."""
    try:
        from app import app
    except Exception as exc:
        return _status.fail("mri_endpoints", f"flask app import failed: {exc}")

    client = app.test_client()
    failures = []
    for path, kind in ENDPOINTS:
        r = client.get(path)
        if kind == "404":
            if r.status_code != NOT_FOUND:
                failures.append(f"{path}: expected 404, got {r.status_code}")
            continue
        if r.status_code != OK_STATUS:
            failures.append(f"{path}: status {r.status_code}")
            continue
        ct = r.headers.get("Content-Type", "")
        if kind == "json" and "json" not in ct:
            failures.append(f"{path}: content-type {ct!r}, expected json")
        elif kind == "html" and "html" not in ct:
            failures.append(f"{path}: content-type {ct!r}, expected html")
    if failures:
        return _status.fail("mri_endpoints", failures[0], {"failures": failures})
    return _status.ok("mri_endpoints", f"{len(ENDPOINTS)} endpoints ok")
