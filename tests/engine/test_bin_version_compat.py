# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Locks the Python-side .bin version surface to what the C engine actually
#   dispatches (veritate_engine/v1/src/model.c, model_load + model_load_int4).
#   If a new engine version ships and these tests pass without updates, the
#   Python-side label table is drifting from the engine and users will see
#   stale labels in the dashboard.
# tests/engine/test_bin_version_compat.py
# ------------------------------------------------------------------------------------
# Imports

import struct

import pytest

from veritate_mri.readers import bin as binr
from veritate_mri.inference.backends import c_engine

# ------------------------------------------------------------------------------------
# Constants

ENGINE_ACCEPTED_VERSIONS = (3, 4, 5, 6, 8, 9, 11, 12)
RETIRED_VERSIONS = (10,)
HEADER_FMT = "<4sIIIIIII"


def _write_min_header(path, version):
    """Minimal 32-byte header: magic + version + dummy shape ints."""
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, b"VRTE", version, 256, 64, 2, 128, 4, 256))


# ------------------------------------------------------------------------------------
# Functions

@pytest.mark.parametrize("version", ENGINE_ACCEPTED_VERSIONS)
def test_python_loader_accepts_every_engine_version(tmp_path, version):
    """_read_bin_shape parses every version the C engine dispatches on."""
    p = tmp_path / "veritate.bin"
    _write_min_header(p, version)
    shape = c_engine._read_bin_shape(str(p))
    assert shape["version"] == version


def test_python_loader_rejects_retired_v10_with_clear_message(tmp_path):
    """v10 was retired during the v11 merge; users get a re-export hint."""
    p = tmp_path / "veritate.bin"
    _write_min_header(p, 10)
    with pytest.raises(RuntimeError, match="v10 is retired"):
        c_engine._read_bin_shape(str(p))


@pytest.mark.parametrize("version", ENGINE_ACCEPTED_VERSIONS)
def test_version_label_table_covers_every_engine_version(version):
    """VERSION_LABELS must name every version the engine accepts so the
    dashboard never shows a bare 'v12'-style fallback for a real release."""
    assert version in binr.VERSION_LABELS


def test_retired_versions_match_documented_set():
    """The retired set is the contract with users for re-export warnings."""
    assert binr.RETIRED_VERSIONS == frozenset(RETIRED_VERSIONS)
