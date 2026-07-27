# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - unit test for the loopback route guard (routes._common.is_loopback). The
#   former system-dependency registry this file also covered was consolidated
#   out; only the loopback guard remains here.
# tests/mri/test_system_deps.py
# ------------------------------------------------------------------------------------
# Imports:



from routes._common import is_loopback

# ------------------------------------------------------------------------------------
# Functions

def test_loopback_accepts_localhost_and_rejects_lan():
    """is_loopback allows 127.0.0.1 and ::1, rejects a LAN address and garbage."""
    assert is_loopback("127.0.0.1") is True
    assert is_loopback("::1") is True
    assert is_loopback("192.168.0.43") is False
    assert is_loopback("not-an-ip") is False
