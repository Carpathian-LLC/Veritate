# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - veritate_mri/inference/decode/* import + a sanity peek at expected classes.
# tests/selftest/checks/check_decode_imports.py
# ------------------------------------------------------------------------------------
# Imports

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA          = "inference"
DECODE_MODS   = (
    "kv_cache", "mtp_decode", "constraints",
)
EXPECTED_NAMES = {
    "kv_cache":   ("KVCachedDecoder",),
    "mtp_decode": ("MTPDecoder",),
}

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """every decode module imports; the documented public classes are present
    where known."""
    miss = []
    for name in DECODE_MODS:
        try:
            mod = __import__("inference.decode." + name, fromlist=["*"])
        except Exception as exc:
            miss.append(f"{name}: import failed: {exc}")
            continue
        for cls in EXPECTED_NAMES.get(name, ()):
            if not hasattr(mod, cls):
                miss.append(f"{name}: missing {cls}")
    if miss:
        return _status.fail("decode_imports", miss[0], {"errors": miss})
    return _status.ok("decode_imports", f"{len(DECODE_MODS)} decode modules ok")
