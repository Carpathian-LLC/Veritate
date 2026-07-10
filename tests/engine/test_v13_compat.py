# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - canonical-compat regression for the v13 engine changes: a v9 fixture must
#   greedy-decode byte-identically to the golden transcript recorded from the
#   pre-v13 engine build. plus a v13 fixture load + generate smoke.
# - requires the built engine binary; skips when absent.
# tests/engine/test_v13_compat.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(REPO_ROOT, "veritate_mri") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "veritate_mri"))

from readers import paths

# ------------------------------------------------------------------------------------
# Constants

FIXTURES     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
V9_FIXTURE   = os.path.join(FIXTURES, "canon_v9_fixture.bin")
V9_GOLDEN    = os.path.join(FIXTURES, "canon_v9_greedy_golden.bin")
V13_FIXTURE  = os.path.join(FIXTURES, "hybrid_v13_fixture.bin")
PROMPTS      = b"Hello world\nabc def\nOnce upon a time\n"
BUDGET       = "32"
N_TURNS      = 3

# ------------------------------------------------------------------------------------
# Functions


def _engine():
    exe = paths.engine_binary_path()
    if not os.path.isfile(exe):
        pytest.skip(f"engine binary not built: {exe}")
    return exe


def _greedy(exe, bin_path):
    env = dict(os.environ, VERITATE_MODEL_PATH=bin_path)
    p = subprocess.run([exe, "chat_greedy", BUDGET], input=PROMPTS,
                       capture_output=True, env=env, timeout=300)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return p.stdout


def test_v9_greedy_matches_golden():
    """v9 fixture greedy transcript is byte-identical to the pre-v13 recording."""
    with open(V9_GOLDEN, "rb") as f:
        golden = f.read()
    assert _greedy(_engine(), V9_FIXTURE) == golden


def test_v13_fixture_loads_and_generates():
    """v13 fixture loads and emits the full greedy budget for every turn."""
    out = _greedy(_engine(), V13_FIXTURE)
    assert len(out) == N_TURNS * (int(BUDGET) + 1)
