# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - shared HTTPS SSL context helper the sync/teacher/runtime modules import.
# tests/mri/test_net.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import ssl
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "veritate_mri")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from runtime import net

# ------------------------------------------------------------------------------------
# Functions

def test_ssl_context_returns_verifying_context():
    """net.ssl_context() returns a hostname-verifying ssl.SSLContext."""
    ctx = net.ssl_context()
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_REQUIRED
