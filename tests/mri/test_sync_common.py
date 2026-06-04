# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Unit tests for sync_common.classify_one covering every branch of the
#   three-state diff (local vs remote vs last_sha). Uses a fake sha_fn so the
#   tests are deterministic and never touch a real hash function.
# tests/mri/test_sync_common.py
# ------------------------------------------------------------------------------------
# Imports

import os
import random
import sys

import pytest

# ------------------------------------------------------------------------------------
# Constants

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
MRI_DIR   = os.path.join(REPO_ROOT, "veritate_mri")
if MRI_DIR not in sys.path:
    sys.path.insert(0, MRI_DIR)

from training.sync import sync_common as sc

REL_NAME      = "file.bin"
SHA_LOCAL     = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
SHA_REMOTE    = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
SHA_LAST_SYNC = "cccccccccccccccccccccccccccccccccccccccc"

# ------------------------------------------------------------------------------------
# Functions

def _write(tmp_path, name=REL_NAME, body=b"x"):
    p = tmp_path / name
    p.write_bytes(body)
    return str(p)


def _fake_sha(value):
    return lambda _path: value


def test_missing_local_none_remote_untracked_is_missing(tmp_path):
    """No local, no remote, no state -> MISSING."""
    random.seed(0)
    st, _ = sc.classify_one(str(tmp_path / "ghost.bin"), None, None, sha_fn=_fake_sha(None))
    assert st == sc.STATE_MISSING


def test_missing_local_none_remote_tracked_is_orphan(tmp_path):
    """No local, no remote, but tracked in state -> ORPHAN."""
    random.seed(0)
    st, _ = sc.classify_one(str(tmp_path / "ghost.bin"), None, {"synced_sha": SHA_LAST_SYNC}, sha_fn=_fake_sha(None))
    assert st == sc.STATE_ORPHAN


def test_missing_local_with_remote_is_missing(tmp_path):
    """No local, remote present -> MISSING."""
    random.seed(0)
    st, _ = sc.classify_one(str(tmp_path / "ghost.bin"), SHA_REMOTE, None, sha_fn=_fake_sha(None))
    assert st == sc.STATE_MISSING


def test_local_present_none_remote_is_orphan(tmp_path):
    """Local on disk, remote unknown -> ORPHAN."""
    random.seed(0)
    p = _write(tmp_path)
    st, _ = sc.classify_one(p, None, None, sha_fn=_fake_sha(SHA_LOCAL))
    assert st == sc.STATE_ORPHAN


def test_local_matches_remote_is_current(tmp_path):
    """Local sha == remote sha -> CURRENT."""
    random.seed(0)
    p = _write(tmp_path)
    st, _ = sc.classify_one(p, SHA_LOCAL, None, sha_fn=_fake_sha(SHA_LOCAL))
    assert st == sc.STATE_CURRENT


def test_local_differs_untracked_is_modified(tmp_path):
    """Local differs from remote, never tracked -> MODIFIED (adopt baseline)."""
    random.seed(0)
    p = _write(tmp_path)
    st, _ = sc.classify_one(p, SHA_REMOTE, None, sha_fn=_fake_sha(SHA_LOCAL))
    assert st == sc.STATE_MODIFIED


def test_local_matches_last_remote_moved_is_update_available(tmp_path):
    """Local == last_sha, remote moved -> UPDATE_AVAILABLE."""
    random.seed(0)
    p = _write(tmp_path)
    st, _ = sc.classify_one(p, SHA_REMOTE, {"synced_sha": SHA_LOCAL}, sha_fn=_fake_sha(SHA_LOCAL))
    assert st == sc.STATE_UPDATE_AVAILABLE


def test_both_moved_is_conflict(tmp_path):
    """Local != last_sha AND remote != last_sha -> CONFLICT."""
    random.seed(0)
    p = _write(tmp_path)
    st, _ = sc.classify_one(p, SHA_REMOTE, {"synced_sha": SHA_LAST_SYNC}, sha_fn=_fake_sha(SHA_LOCAL))
    assert st == sc.STATE_CONFLICT


def test_local_moved_remote_unchanged_is_modified(tmp_path):
    """Local != last_sha, remote == last_sha -> MODIFIED."""
    random.seed(0)
    p = _write(tmp_path)
    st, _ = sc.classify_one(p, SHA_LAST_SYNC, {"synced_sha": SHA_LAST_SYNC}, sha_fn=_fake_sha(SHA_LOCAL))
    assert st == sc.STATE_MODIFIED
