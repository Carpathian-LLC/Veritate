# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - read-only catalog reachability check: the corpus library catalog endpoint
#   must respond. nothing is downloaded.
# tests/selftest/checks/check_corpus_catalog.py
# ------------------------------------------------------------------------------------
# Imports

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA     = "mri"
ENDPOINT = "/corpus/library/catalog"

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """flask /corpus/library/catalog returns a JSON payload (cached if present)."""
    try:
        from app import app
    except Exception as exc:
        return _status.fail("corpus_catalog", f"flask app import failed: {exc}")
    client = app.test_client()
    r = client.get(ENDPOINT)
    if r.status_code >= 500:
        return _status.fail("corpus_catalog", f"status {r.status_code}",
                            {"body": r.get_data(as_text=True)[:300]})
    return _status.ok("corpus_catalog", f"status {r.status_code}")
