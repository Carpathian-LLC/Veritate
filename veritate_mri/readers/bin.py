# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - read precision tag and bin format version from a model's veritate.bin header.
# veritate_mri/readers/bin.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import struct

from . import paths

# ------------------------------------------------------------------------------------
# Constants

VERITATE_MODEL_MAGIC = b"VRTE"

# Mirror of the C engine's accepted versions in veritate_engine/v1/src/model.c
# (model_load + model_load_int4 dispatch). The engine accepts every entry here
# at load time with no runtime cost — version handling is load-time-only;
# decode kernels are shared. Keep this list in lockstep with the C dispatch.
VERSION_LABELS = {
    3:  "INT8",
    4:  "INT4-packed",
    5:  "INT8-percol",
    6:  "INT8-MoD",
    8:  "INT8-norm",
    9:  "INT8-boost",
    11: "QAT-unified",
    12: "MTP",
}

# v10 was assigned twice on different branches (MoE-on-dev vs ternary-on-
# experimental) and was retired during the v11 unification. Any .bin with
# version=10 must be re-exported from its most recent .pt checkpoint.
RETIRED_VERSIONS = frozenset({10})

# ------------------------------------------------------------------------------------
# Functions

def header(name):
    """Return (precision_label, version_int) for the model's bin, or ('?', 0) if absent."""
    p = paths.bin_path(name)
    if not os.path.isfile(p):
        return ("?", 0)
    try:
        with open(p, "rb") as f:
            magic = f.read(4)
            if magic != VERITATE_MODEL_MAGIC:
                return ("?", 0)
            (version,) = struct.unpack("<I", f.read(4))
    except (OSError, struct.error):
        return ("?", 0)
    label = VERSION_LABELS.get(int(version), f"v{version}")
    return (label, int(version))


def act_boost(name):
    """Return act_boost int from a v9+ bin (None for older versions or missing)."""
    p = paths.bin_path(name)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "rb") as f:
            magic = f.read(4)
            if magic != VERITATE_MODEL_MAGIC:
                return None
            (version,) = struct.unpack("<I", f.read(4))
            if int(version) < 9:
                return None
            f.seek(struct.calcsize("<4sIIIIIII"))
            (boost,) = struct.unpack("<i", f.read(4))
            return int(boost)
    except (OSError, struct.error):
        return None


def exists(name):
    return os.path.isfile(paths.bin_path(name))


