# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - asserts the tutorial mode web assets exist, are wired into index.html, carry
#   the step-data array and the tutorial_completed POST, and parse under node.
#   no torch, no network. node --check skipped when node is absent.
# tests/mri/test_tutorial_assets.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import shutil
import subprocess

import pytest

# ------------------------------------------------------------------------------------
# Constants

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
WEB_DIR   = os.path.join(REPO_ROOT, "veritate_mri", "web")
TUT_JS    = os.path.join(WEB_DIR, "tutorial.js")
TUT_CSS   = os.path.join(WEB_DIR, "tutorial.css")
INDEX     = os.path.join(WEB_DIR, "index.html")

# ------------------------------------------------------------------------------------
# Functions

def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_assets_exist_non_empty():
    """tutorial.js and tutorial.css exist and are non-empty."""
    assert os.path.getsize(TUT_JS) > 0
    assert os.path.getsize(TUT_CSS) > 0


def test_index_references_css():
    """index.html links tutorial.css."""
    assert "/static/tutorial.css" in _read(INDEX)


def test_index_references_js():
    """index.html includes tutorial.js after index.js."""
    html = _read(INDEX)
    assert "/static/tutorial.js" in html
    assert html.index("/static/index.js") < html.index("/static/tutorial.js")


def test_js_has_step_array():
    """tutorial.js defines the STEPS data array."""
    assert "const STEPS = [" in _read(TUT_JS)


def test_js_posts_completed():
    """tutorial.js POSTs tutorial_completed to /settings."""
    js = _read(TUT_JS)
    assert "tutorial_completed: true" in js
    assert "/settings" in js


def test_js_node_check():
    """tutorial.js parses without syntax error under node --check."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    r = subprocess.run([node, "--check", TUT_JS], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
