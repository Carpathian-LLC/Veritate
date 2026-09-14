# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the three-state file sync every sync lane (app, corpus, models) rests on: the
#   state file round trip and its corruption fallback, the two hash formats (git's blob
#   sha must match what the GitHub tree API reports), path-traversal rejection, and the
#   six classifications from local / remote / last-synced hashes with the bulk action
#   each maps to.
# tests/mri/test_sync_common.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import pytest
from training.sync import sync_common as sc

# ------------------------------------------------------------------------------------
# Constants

GIT_BLOB_SHA_OF_HELLO = "ce013625030ba8dba906f756967f9e9ca394464a"   # `git hash-object` of b"hello\n"

# ------------------------------------------------------------------------------------
# Functions


def _content_sha(path):
    """A stand-in hash: the file's own text, so the states read off the content."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_state_round_trips_and_a_corrupt_or_foreign_file_reads_as_empty(tmp_path):
    root = str(tmp_path)
    assert sc.load_state(root) == {}
    sc.save_state(root, {"a.txt": {"synced_sha": "x"}}, remote_branch="main")
    assert sc.load_state(root) == {"a.txt": {"synced_sha": "x"}}
    with open(sc.state_path(root), "w", encoding="utf-8") as f:
        f.write("{not json")
    assert sc.load_state(root) == {}
    sc.save_state(root, {"a.txt": {}})
    with open(sc.state_path(root), "w", encoding="utf-8") as f:
        f.write('{"version": 99, "files": {"a.txt": {}}}')
    assert sc.load_state(root) == {}


def test_git_blob_sha_matches_git_and_missing_files_hash_to_none(tmp_path):
    p = tmp_path / "h.txt"
    p.write_bytes(b"hello\n")
    assert sc.git_blob_sha1_file(str(p)) == GIT_BLOB_SHA_OF_HELLO
    assert sc.sha256_file(str(p)) == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
    assert sc.sha256_file(str(tmp_path / "nope")) is None and sc.git_blob_sha1_file(str(tmp_path / "nope")) is None


@pytest.mark.parametrize("rel", ["../x", "a/../../x", ".", ""])
def test_a_destination_outside_the_root_is_refused(tmp_path, rel):
    assert sc.safe_dest(rel, str(tmp_path)) is None


def test_a_destination_inside_the_root_is_absolute(tmp_path):
    assert sc.safe_dest("a/b.bin", str(tmp_path)) == os.path.join(str(tmp_path), "a", "b.bin")


def test_the_six_classifications(tmp_path):
    local = tmp_path / "f"
    sha = _content_sha
    path = str(local)
    assert sc.classify_one(path, "r1", None, sha) == (sc.STATE_MISSING, None)
    assert sc.classify_one(path, None, {"synced_sha": "r0"}, sha) == (sc.STATE_ORPHAN, None)
    local.write_text("r1")
    assert sc.classify_one(path, "r1", None, sha) == (sc.STATE_CURRENT, "r1")
    assert sc.classify_one(path, None, None, sha) == (sc.STATE_ORPHAN, "r1")
    assert sc.classify_one(path, "r2", None, sha) == (sc.STATE_MODIFIED, "r1")            # never tracked
    assert sc.classify_one(path, "r2", {"synced_sha": "r1"}, sha) == (sc.STATE_UPDATE_AVAILABLE, "r1")
    assert sc.classify_one(path, "r0", {"synced_sha": "r0"}, sha) == (sc.STATE_MODIFIED, "r1")   # only local moved
    assert sc.classify_one(path, "r2", {"synced_sha": "r0"}, sha) == (sc.STATE_CONFLICT, "r1")   # both moved


def test_classify_set_covers_remote_files_and_tracked_orphans_sorted(tmp_path):
    (tmp_path / "b").write_text("r1")
    rows = sc.classify_set(str(tmp_path), {"b": "r1", "c": "r9"}, {"a": {"synced_sha": "old"}},
                           sha_fn=_content_sha)
    assert [(r["path"], r["state"]) for r in rows] == [("a", sc.STATE_ORPHAN), ("b", sc.STATE_CURRENT),
                                                         ("c", sc.STATE_MISSING)]
    assert sc.summarize_states(rows) == {sc.STATE_ORPHAN: 1, sc.STATE_CURRENT: 1, sc.STATE_MISSING: 1}


def test_the_bulk_button_only_installs_and_updates():
    """Anything the user may have touched needs an explicit per-file decision."""
    assert sc.default_action_for_state(sc.STATE_MISSING) == sc.ACTION_INSTALL
    assert sc.default_action_for_state(sc.STATE_UPDATE_AVAILABLE) == sc.ACTION_UPDATE
    for st in (sc.STATE_CURRENT, sc.STATE_MODIFIED, sc.STATE_CONFLICT, sc.STATE_ORPHAN, "unknown"):
        assert sc.default_action_for_state(st) == sc.ACTION_SKIP
