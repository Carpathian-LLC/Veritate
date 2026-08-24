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

# Mirror of the C engine's accepted versions in veritate_engine/src/model.c
# (model_load + model_load_int4 dispatch). The engine accepts every entry here
# at load time with no runtime cost: version handling is load-time-only;
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
    13: "HYBRID-fp",
}

# Bin format version 13 carries no act_boost: the first extension field is the weight dtype and
# the numeric path is fp32/fp16, so the boost/QAT gibberish heuristic does not
# apply. act_boost() returns None for these versions.
NO_ACT_BOOST_VERSIONS = frozenset({13})

# Bin format version 10 was assigned twice on different branches (MoE-on-dev vs
# ternary-on-experimental) and was retired when version 11 unified them. Any .bin with
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


# v13 header extension field 0. Mirrors export.HYBRID_DTYPES; the engine reads
# the same field to pick its kernel.
HYBRID_DTYPE_LABELS = {0: "fp32", 1: "fp16", 2: "int8"}


def weight_dtype(name):
    """Weight dtype a v13 hybrid bin was exported at ("fp32"/"fp16"/"int8"), or
    None for any other version. A re-export has to keep whatever the box is
    serving: silently moving an int8 box to the fp16 default doubles its bin and
    changes its decode speed."""
    p = paths.bin_path(name)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "rb") as f:
            if f.read(4) != VERITATE_MODEL_MAGIC:
                return None
            (version,) = struct.unpack("<I", f.read(4))
            if int(version) != 13:
                return None
            f.seek(struct.calcsize("<4sIIIIIII"))
            (code,) = struct.unpack("<i", f.read(4))
    except (OSError, struct.error):
        return None
    return HYBRID_DTYPE_LABELS.get(int(code))


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
            if int(version) < 9 or int(version) in NO_ACT_BOOST_VERSIONS:
                return None
            f.seek(struct.calcsize("<4sIIIIIII"))
            (boost,) = struct.unpack("<i", f.read(4))
            return int(boost)
    except (OSError, struct.error):
        return None


def exists(name):
    return os.path.isfile(paths.bin_path(name))


