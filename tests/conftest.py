# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - shared pytest fixtures + path setup. pulled in automatically by every test.
# tests/conftest.py
# ------------------------------------------------------------------------------------
# Imports

import os
import sys

import pytest

# ------------------------------------------------------------------------------------
# Constants

REPO_ROOT     = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR    = os.path.join(REPO_ROOT, "models")
ENGINE_BINARY = os.path.join(REPO_ROOT, "veritate_engine", "v1", "bin", "macos", "arm64", "veritate")

# put the repo root on sys.path so `from veritate_mri.readers import paths` etc work.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture(scope="session")
def repo_root():
    """absolute path to the Veritate repo root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def models_dir():
    """absolute path to models/. may be empty on a fresh clone; tests should skip if so."""
    return MODELS_DIR


@pytest.fixture(scope="session")
def engine_binary_or_skip():
    """path to the built engine binary, or skip if not built. tests that need the
    binary depend on this fixture instead of importing the path directly."""
    if not os.path.isfile(ENGINE_BINARY):
        pytest.skip(f"engine binary not built: {ENGINE_BINARY}. run veritate_engine/v1/build/build.sh first.")
    return ENGINE_BINARY


@pytest.fixture
def model_bins():
    """list of every veritate*.bin in models/<name>/. empty list if no models exist."""
    out = []
    if not os.path.isdir(MODELS_DIR):
        return out
    for name in sorted(os.listdir(MODELS_DIR)):
        sub = os.path.join(MODELS_DIR, name)
        if not os.path.isdir(sub):
            continue
        for fname in os.listdir(sub):
            if fname.startswith("veritate") and fname.endswith(".bin"):
                out.append(os.path.join(sub, fname))
    return out
