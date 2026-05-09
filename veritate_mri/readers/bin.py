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
    3:  "INT8",
    4:  "INT8-percol",
    5:  "INT8-percol-v5",
    6:  "INT8-mod",
    7:  "BitNet",
    8:  "INT8-norm",
    9:  "INT8-boost",
    10: "RETIRED-v10",
    11: "QAT-unified",
}

# Versions the engine still loads. v10 was retired during the v11 merge: any
# .bin still on disk with version=10 must be re-exported from its .pt
# checkpoint to v11 before it can be loaded.
SUPPORTED_VERSIONS = frozenset({3, 4, 5, 6, 7, 8, 9, 11})
RETIRED_VERSIONS   = frozenset({10})

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


def health(name):
    """Return a structured health record for a model's .bin.

    Shape: {present, version, label, stale, reason}.
    `stale=True` means the engine will refuse to load this .bin and the user
    should re-export from the most recent .pt checkpoint."""
    p = paths.bin_path(name)
    out = {"present": False, "version": 0, "label": "?", "stale": False, "reason": None}

    # Some plugins (e.g. distill_teacher) write a sibling bin named
    # `veritate_v2.bin` instead of the canonical `veritate.bin`. Check both.
    candidates = [p, os.path.join(os.path.dirname(p), "veritate_v2.bin")]
    actual = next((c for c in candidates if os.path.isfile(c)), None)
    if actual is None:
        return out

    out["present"] = True
    try:
        with open(actual, "rb") as f:
            magic = f.read(4)
            if magic != VERITATE_MODEL_MAGIC:
                out["stale"]  = True
                out["reason"] = "magic mismatch (corrupted or wrong file)"
                return out
            (version,) = struct.unpack("<I", f.read(4))
    except (OSError, struct.error) as e:
        out["stale"]  = True
        out["reason"] = f"header read failed: {e}"
        return out

    out["version"] = int(version)
    out["label"]   = VERSION_LABELS.get(int(version), f"v{version}")

    if int(version) in RETIRED_VERSIONS:
        out["stale"]  = True
        out["reason"] = (f"v{version} was retired (assigned twice on different "
                         f"branches before the v11 merge). re-export from the "
                         f"most recent .pt checkpoint.")
    elif int(version) not in SUPPORTED_VERSIONS:
        out["stale"]  = True
        out["reason"] = f"unsupported version v{version}. re-export from the most recent .pt checkpoint."

    return out


def is_stale(name):
    """True if this model's .bin won't load against the current engine."""
    return health(name)["stale"]
