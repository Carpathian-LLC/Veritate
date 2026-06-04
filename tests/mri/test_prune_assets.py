# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - asserts the neuron-pruning web assets exist, are wired into index.html, call
#   the pruning endpoints, and parse under node. no torch, no network. node
#   --check skipped when node is absent.
# tests/mri/test_prune_assets.py
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
PRUNE_JS  = os.path.join(WEB_DIR, "prune.js")
PRUNE_CSS = os.path.join(WEB_DIR, "prune.css")
INDEX     = os.path.join(WEB_DIR, "index.html")

# ------------------------------------------------------------------------------------
# Functions

def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_assets_exist_non_empty():
    """prune.js and prune.css exist and are non-empty."""
    assert os.path.getsize(PRUNE_JS) > 0
    assert os.path.getsize(PRUNE_CSS) > 0


def test_index_references_css():
    """index.html links prune.css."""
    assert "/static/prune.css" in _read(INDEX)


def test_index_references_js_after_index():
    """index.html includes prune.js after index.js."""
    html = _read(INDEX)
    assert "/static/prune.js" in html
    assert html.index("/static/index.js") < html.index("/static/prune.js")


def test_js_calls_pruning_endpoints():
    """prune.js calls the report and generate_plugin endpoints."""
    js = _read(PRUNE_JS)
    assert "/pruning/report" in js
    assert "/pruning/generate_plugin" in js


def test_js_mounts_into_learning_tab():
    """prune.js mounts its panel into the Models (learning) tab."""
    assert 'data-tab="learning"' in _read(PRUNE_JS)


def test_js_parses_under_node():
    """prune.js is syntactically valid JavaScript."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    r = subprocess.run([node, "--check", PRUNE_JS], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
