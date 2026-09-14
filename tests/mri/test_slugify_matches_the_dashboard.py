# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the model name a user types is slugified TWICE, once in the browser to show the
#   composed name on the Training form (index.js `_trSlugify`) and once in python to create
#   the directory (readers/models.slugify_user_name). They must agree character for
#   character or the form promises a name the trainer does not make, and the collision
#   check that guards an existing model looks at the wrong directory. This runs both over
#   the same inputs and compares. Skips without node, like the other javascript tests.
# tests/mri/test_slugify_matches_the_dashboard.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import shutil
import subprocess

import pytest
from conftest import REPO_ROOT
from readers.models import slugify_user_name

# ------------------------------------------------------------------------------------
# Constants

INDEX_JS = os.path.join(REPO_ROOT, "veritate_mri", "web", "index.js")

CASES = [
    "Plain Name",
    "chat200m",
    "  --Odd..Name--  ",
    "UPPER CASE",
    "many___underscores",
    "trailing_",
    "_leading",
    "dots.and-dashes mixed",
    "digits 123 456",
    "café mocha",          # non-ascii letters: str.isalnum kept them, the browser never did
    "naïve model",
    "中文",
    "emoji 🙂 name",
    "",
    "   ",
    "!!!",
]

# ------------------------------------------------------------------------------------
# Functions


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_both_sides_slugify_a_name_the_same_way():
    """Every divergence here is a name the form shows and the trainer does not create."""
    with open(INDEX_JS, encoding="utf-8") as handle:
        src = handle.read()
    start = src.index("function _trSlugify")
    body = src[start:src.index("\nfunction ", start + 1)]
    script = body + "\nconsole.log(JSON.stringify(" + json.dumps(CASES) + ".map(_trSlugify)));"
    proc = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == [slugify_user_name(c) for c in CASES]


def test_a_name_that_slugifies_to_nothing_is_empty_not_an_error():
    """The route checks the slug before composing; an exception here would 500 the form."""
    assert slugify_user_name("中文") == ""
    assert slugify_user_name("") == ""
    assert slugify_user_name(None) == ""
