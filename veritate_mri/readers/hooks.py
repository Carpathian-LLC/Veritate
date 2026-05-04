# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - single ingestion entrypoint for per-step hook artifacts.
# - artifact set defined in paths.HOOK_ARTIFACTS.
# - mtime-keyed cache. callers receive parsed dicts/arrays, never paths.
# veritate_mri/readers/hooks.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os

from . import paths

# ------------------------------------------------------------------------------------
# Constants

_CACHE = {}

# ------------------------------------------------------------------------------------
# Functions

def _load_json(p):
    try:
        st = os.stat(p)
    except OSError:
        return None
    hit = _CACHE.get(p)
    if hit and hit[0] == st.st_mtime:
        return hit[1]
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    _CACHE[p] = (st.st_mtime, data)
    return data


def _load_npz(p):
    try:
        import numpy as np
        return dict(np.load(p, allow_pickle=False))
    except (OSError, ValueError):
        return None


def list_steps(name):
    d = paths.hooks_dir(name)
    if not os.path.isdir(d):
        return []
    steps = []
    for entry in os.listdir(d):
        m = paths.HOOK_STEP_RE.match(entry)
        if m and os.path.isdir(os.path.join(d, entry)):
            steps.append(int(m.group(1)))
    return sorted(steps)


def load_artifact(name, step, artifact):
    if artifact not in paths.HOOK_ARTIFACTS:
        return None
    p = paths.hook_artifact_path(name, step, artifact)
    kind = paths.HOOK_ARTIFACTS[artifact][1]
    if kind == "json":
        return _load_json(p)
    if kind == "npz":
        return _load_npz(p)
    return None


def load_series(name, artifact):
    out = []
    for step in list_steps(name):
        d = load_artifact(name, step, artifact)
        if d is not None:
            out.append((step, d))
    return out


def artifact_exists(name, step, artifact):
    if artifact not in paths.HOOK_ARTIFACTS:
        return False
    return os.path.isfile(paths.hook_artifact_path(name, step, artifact))
