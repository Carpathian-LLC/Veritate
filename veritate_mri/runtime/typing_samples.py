# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - store for recorded typing sessions: the per-keystroke evidence the draft trigger
#   is tuned against.
# - a session is a list of keystrokes, each carrying its gap, the character, and where
#   in the text it fell. Keystrokes ending a question are labelled done=true, which is
#   the ground truth that makes a candidate rule scorable instead of a matter of taste.
# - raw only. No medians, no thresholds, no recommendations: a summary computed here
#   would be one more thing to disagree with the data.
# - one json file per session under data/typing_samples/, machine-local (gitignored
#   like the rest of data/).
# veritate_mri/runtime/typing_samples.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os

from readers.paths import REPO_ROOT

# ------------------------------------------------------------------------------------
# Constants

SAMPLES_DIR   = os.path.join(REPO_ROOT, "data", "typing_samples")
FILE_PREFIX   = "session_"
FILE_SUFFIX   = ".json"
# A session that would not fit in memory is a bug in the recorder, not a long sitting.
MAX_KEYSTROKES = 20000
MAX_LIST       = 200

# ------------------------------------------------------------------------------------
# Functions

def path(name):
    return os.path.join(SAMPLES_DIR, f"{FILE_PREFIX}{name}{FILE_SUFFIX}")


def save(session):
    """Write one recorded session. `session` carries `keys` (the per-keystroke records)
    and whatever context the recorder captured. Returns the stored name."""
    keys = session.get("keys")
    if not isinstance(keys, list) or not keys:
        raise ValueError("session has no keystrokes")
    if len(keys) > MAX_KEYSTROKES:
        raise ValueError(f"session exceeds {MAX_KEYSTROKES} keystrokes")
    name = str(session.get("name") or "").strip()
    if not name or not name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("session name must be alphanumeric with - or _")
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    with open(path(name), "w", encoding="utf-8") as f:
        json.dump(session, f, indent=1)
    return name


def load(name):
    with open(path(name), encoding="utf-8") as f:
        return json.load(f)


def listing():
    """Stored sessions, newest first: name, keystroke count, and how many keystrokes
    carry the done label."""
    if not os.path.isdir(SAMPLES_DIR):
        return []
    out = []
    for fn in os.listdir(SAMPLES_DIR):
        if not (fn.startswith(FILE_PREFIX) and fn.endswith(FILE_SUFFIX)):
            continue
        path = os.path.join(SAMPLES_DIR, fn)
        name = fn[len(FILE_PREFIX):-len(FILE_SUFFIX)]
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            keys = data.get("keys") or []
            out.append({
                "name":      name,
                "keystrokes": len(keys),
                "questions": sum(1 for k in keys if k.get("done")),
                "mtime":     os.path.getmtime(path),
            })
        except (OSError, ValueError):
            continue
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out[:MAX_LIST]
