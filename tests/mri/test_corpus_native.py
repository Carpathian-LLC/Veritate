# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Tests for the corpus library 'native' format: install copies repo-bundled
#   bins from veritate_mri/data/corpus/ into trainers/corpus/, uninstall only
#   removes the user copy. Also pins the shipped mcp_docs corpus contract.
# tests/mri/test_corpus_native.py
# ------------------------------------------------------------------------------------
# Imports

import json
import os
import sys

# ------------------------------------------------------------------------------------
# Constants

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
MRI_DIR   = os.path.join(REPO_ROOT, "veritate_mri")
if MRI_DIR not in sys.path:
    sys.path.insert(0, MRI_DIR)

from training.sync import corpus_sync as cs

STEM        = "nat_test"
TRAIN_BYTES = b"train corpus bytes<|endoftext|>more"
VAL_BYTES   = b"val corpus bytes"

# ------------------------------------------------------------------------------------
# Functions

def _setup_dirs(tmp_path, monkeypatch, with_val=True):
    native = tmp_path / "native"
    user   = tmp_path / "user"
    native.mkdir()
    (native / f"{STEM}_train.bin").write_bytes(TRAIN_BYTES)
    if with_val:
        (native / f"{STEM}_val.bin").write_bytes(VAL_BYTES)
    monkeypatch.setattr(cs, "NATIVE_CORPUS_DIR", str(native))
    monkeypatch.setattr(cs, "CORPUS_DIR", str(user))
    return native, user


def test_native_install_copies_train_and_val(tmp_path, monkeypatch):
    """install(format=native) copies both bins into the user corpus dir."""
    _, user = _setup_dirs(tmp_path, monkeypatch)
    res = cs.install({"stem": STEM, "format": "native"})
    assert res["ok"] is True
    assert (user / f"{STEM}_train.bin").read_bytes() == TRAIN_BYTES
    assert (user / f"{STEM}_val.bin").read_bytes() == VAL_BYTES


def test_native_install_reports_train_bytes(tmp_path, monkeypatch):
    """install(format=native) returns bytes_train = train file size."""
    _setup_dirs(tmp_path, monkeypatch)
    res = cs.install({"stem": STEM, "format": "native"})
    assert res["bytes_train"] == len(TRAIN_BYTES)


def test_native_install_train_only(tmp_path, monkeypatch):
    """No bundled val bin -> install copies train only, still ok."""
    _, user = _setup_dirs(tmp_path, monkeypatch, with_val=False)
    res = cs.install({"stem": STEM, "format": "native"})
    assert res["ok"] is True
    assert (user / f"{STEM}_train.bin").is_file()
    assert not (user / f"{STEM}_val.bin").exists()


def test_native_install_missing_source_fails(tmp_path, monkeypatch):
    """Stem not bundled in the native dir -> install refuses."""
    _setup_dirs(tmp_path, monkeypatch)
    res = cs.install({"stem": "ghost", "format": "native"})
    assert res["ok"] is False


def test_native_uninstall_keeps_repo_copy(tmp_path, monkeypatch):
    """uninstall removes the user copy; the repo-bundled bins stay."""
    native, user = _setup_dirs(tmp_path, monkeypatch)
    cs.install({"stem": STEM, "format": "native"})
    res = cs.uninstall(STEM)
    assert res["ok"] is True
    assert not (user / f"{STEM}_train.bin").exists()
    assert (native / f"{STEM}_train.bin").is_file()


def test_catalog_annotates_native_entry(tmp_path, monkeypatch):
    """catalog() marks native entries available and fills sizes from the bundled files."""
    _setup_dirs(tmp_path, monkeypatch)
    monkeypatch.setattr(cs.settings_mod, "get", lambda: {})
    monkeypatch.setattr(cs, "_load_local_catalog", lambda: [{"stem": STEM, "format": "native"}])
    entry = next(c for c in cs.catalog()["corpora"] if c["stem"] == STEM)
    assert entry["native_available"] is True
    assert entry["size_train"] == len(TRAIN_BYTES)
    assert entry["size_val"] == len(VAL_BYTES)


def test_mcp_docs_bins_shipped():
    """mcp_docs train and val bins exist in veritate_mri/data/corpus/ and are non-empty."""
    from readers import paths
    train = paths.native_corpus_train_path("mcp_docs")
    val   = paths.native_corpus_val_path("mcp_docs")
    assert os.path.getsize(train) > 0
    assert os.path.getsize(val) > 0


def test_mcp_docs_registered_as_native():
    """mcp_docs is in the shipped catalog with format=native."""
    with open(cs.LOCAL_CATALOG_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)["corpora"]
    entry = next(e for e in entries if e["stem"] == "mcp_docs")
    assert entry["format"] == "native"
