# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - list and validate model directories under models/.
# - naming convention: <name>_<param>_<precision>_<version>.
# veritate_mri/readers/models.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import re

from . import paths

# ------------------------------------------------------------------------------------
# Constants

# Model dir name = <corpus>_<size>_<precision>_<version>[_<variant>]. The corpus
# segment may itself contain underscores (e.g. "children_classics",
# "general_fiction"); the next three segments are unambiguous because <size> is
# digits+m|b, <precision> is [a-z0-9]+, and <version> starts with v. The
# optional trailing <variant> tags adapter/QAT derivatives (e.g. "_qat", "_m1",
# "_m3"); it must start with a letter so it can't be confused with a version
# segment. Greedy matching backtracks until the fixed trailing segments fit,
# leaving the rest as <corpus>.
NAME_RE = re.compile(
    r"^[a-z0-9]+(?:_[a-z0-9]+)*_[0-9]+[mb]_[a-z0-9]+_v[0-9]+[a-z]?(?:_[a-z][a-z0-9]*)?$"
)

# ------------------------------------------------------------------------------------
# Functions

def is_valid_name(name):
    return bool(NAME_RE.match(name or ""))


def exists(name):
    return os.path.isdir(paths.model_dir(name)) and os.path.isfile(paths.config_path(name))


def list_models():
    if not os.path.isdir(paths.MODELS_ROOT):
        return []
    out = []
    for entry in sorted(os.listdir(paths.MODELS_ROOT)):
        if exists(entry):
            out.append(entry)
    return out
