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

# Names must be filesystem-safe: lowercase letters, digits, and underscores
# only; no leading or trailing underscore; at least one char. Anything else
# (size tokens, version tags, corpus prefixes) is descriptive and lives in
# config.json, not in the directory name.
NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_]*[a-z0-9])?$")

# ------------------------------------------------------------------------------------
# Functions

def is_valid_name(name):
    return bool(NAME_RE.match(name or ""))


def slugify_user_name(text):
    """Lowercase, strip diacritics-ish, collapse whitespace/dashes to '_',
    keep only [a-z0-9_], collapse repeats, trim leading/trailing '_'."""
    s = (text or "").strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_", "."):
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


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
