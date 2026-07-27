# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers the updater's conflict gate: local_edits() compares the tree against the
#   files a pull is about to write, not against the last-pull snapshot alone.
# tests/mri/test_app_sync_edits.py
# ------------------------------------------------------------------------------------
# Imports:

import hashlib
import json
import os

import pytest
from training.sync import app_sync

# ------------------------------------------------------------------------------------
# Constants

BASELINE_TEXT = "baseline\n"
UPSTREAM_TEXT = "upstream\n"
EDITED_TEXT   = "edited\n"
TRACKED_REL   = "veritate_core/model.py"

# ------------------------------------------------------------------------------------
# Functions

def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _write(root, rel, text):
    path = os.path.join(root, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Point the updater at an empty throwaway tree with a writable baseline."""
    root = str(tmp_path)
    monkeypatch.setattr(app_sync, "REPO_DIR", root)
    monkeypatch.setattr(app_sync, "BASELINE_PATH", os.path.join(root, "data", "baseline.json"))
    return root


def _set_baseline(files):
    os.makedirs(os.path.dirname(app_sync.BASELINE_PATH), exist_ok=True)
    with open(app_sync.BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "written_at": 0, "branch": "dev", "files": files}, f)


def test_local_file_matching_incoming_is_not_a_conflict(repo):
    """A file that differs from the baseline but equals the incoming version reports clean."""
    _write(repo, TRACKED_REL, UPSTREAM_TEXT)
    _set_baseline({TRACKED_REL: _sha(BASELINE_TEXT)})
    edits = app_sync.local_edits(incoming={TRACKED_REL: _sha(UPSTREAM_TEXT)})
    assert edits["counts"]["modified"] == 0


def test_local_edit_conflicting_with_incoming_is_reported(repo):
    """A file whose bytes differ from the incoming version is reported as modified."""
    _write(repo, TRACKED_REL, EDITED_TEXT)
    _set_baseline({TRACKED_REL: _sha(BASELINE_TEXT)})
    edits = app_sync.local_edits(incoming={TRACKED_REL: _sha(UPSTREAM_TEXT)})
    assert [m["path"] for m in edits["modified"]] == [TRACKED_REL]


def test_path_absent_from_incoming_is_not_reported(repo):
    """A tracked path the tarball no longer ships cannot be overwritten, so it never gates."""
    _write(repo, TRACKED_REL, EDITED_TEXT)
    _set_baseline({TRACKED_REL: _sha(BASELINE_TEXT)})
    edits = app_sync.local_edits(incoming={"veritate_core/load.py": _sha(UPSTREAM_TEXT)})
    assert edits["counts"]["modified"] == 0


def test_deleted_file_the_pull_restores_is_reported_missing(repo):
    """A baseline file deleted locally but present in the tarball is reported as missing."""
    _set_baseline({TRACKED_REL: _sha(BASELINE_TEXT)})
    edits = app_sync.local_edits(incoming={TRACKED_REL: _sha(UPSTREAM_TEXT)})
    assert [m["path"] for m in edits["missing"]] == [TRACKED_REL]


def test_mass_divergence_discards_the_baseline(repo):
    """A baseline diverging past the stale threshold is dropped instead of gating the pull."""
    count = app_sync.STALE_BASELINE_MIN_FILES
    rels = [f"veritate_core/mod_{i}.py" for i in range(count)]
    for rel in rels:
        _write(repo, rel, EDITED_TEXT)
    _set_baseline({rel: _sha(BASELINE_TEXT) for rel in rels})
    edits = app_sync.local_edits(incoming={rel: _sha(UPSTREAM_TEXT) for rel in rels})
    assert edits["has_baseline"] is False
