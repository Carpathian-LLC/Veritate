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

# Two accepted shapes:
#   (legacy) <corpus>_<size>_<precision>_<version>[_<variant>] — kept so existing
#            models like "fineweb_edu_800m_bf16_v1" still validate.
#   (new)    <user_slug>_<size> — the user picks any slug; the form auto-appends
#            <size> (e.g. "chatty_otter_85m"). Spec details (corpus, precision,
#            version, variant) move into the description.
# Size token: <digits><m|b>[optional digits], so 85m, 1b, 1b3 (1.3B), 1b5 all pass.
NAME_RE_LEGACY = re.compile(
    r"^[a-z0-9]+(?:_[a-z0-9]+)*_[0-9]+[mb][0-9]*_[a-z0-9]+_v[0-9]+[a-z]?(?:_[a-z][a-z0-9]*)?$"
)
NAME_RE_USER = re.compile(
    r"^[a-z0-9]+(?:_[a-z0-9]+)*_[0-9]+[mb][0-9]*$"
)

# ------------------------------------------------------------------------------------
# Functions

def is_valid_name(name):
    n = name or ""
    return bool(NAME_RE_USER.match(n) or NAME_RE_LEGACY.match(n))


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
