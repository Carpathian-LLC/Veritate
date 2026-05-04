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

VERSION_LABELS = {
    3: "INT8",
    4: "INT8-percol",
    5: "INT8-percol-v5",
    6: "INT8-mod",
    7: "BitNet",
    8: "INT8-norm",
    9: "INT8-boost",
}

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
