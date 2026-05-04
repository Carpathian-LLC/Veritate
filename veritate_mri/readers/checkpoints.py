# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - list a model's pytorch checkpoints; load one as a state dict.
# veritate_mri/readers/checkpoints.py
# ------------------------------------------------------------------------------------
# Imports:

import os

from . import paths

# ------------------------------------------------------------------------------------
# Constants

# ------------------------------------------------------------------------------------
# Functions

def list_steps(name):
    d = paths.checkpoints_dir(name)
    if not os.path.isdir(d):
        return []
    steps = []
    for entry in os.listdir(d):
        m = paths.CHECKPOINT_RE.match(entry)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def latest_step(name):
    steps = list_steps(name)
    return steps[-1] if steps else None


def path_for(name, step):
    return paths.checkpoint_path(name, step)
