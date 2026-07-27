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

import ssl

from runtime import net

# ------------------------------------------------------------------------------------
# Functions

def test_ssl_context_returns_verifying_context():
    """net.ssl_context() returns a hostname-verifying ssl.SSLContext."""
    ctx = net.ssl_context()
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_REQUIRED
