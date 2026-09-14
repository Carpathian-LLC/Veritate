# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the models sync lane over a stubbed remote tree: provenance splits local-trained dirs
#   (invisible to sync) from remote-pulled ones, files() classifies against git blob shas
#   and flags large files, check() keeps the banner's shape, and an unreachable remote is
#   an ok:false answer, not an exception.
# tests/mri/test_models_sync.py
# ------------------------------------------------------------------------------------
# Imports:

import urllib.error

import pytest
from training.sync import models_sync as ms
from training.sync import sync_common as sc

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def rig(tmp_path, monkeypatch):
    models = tmp_path / "models"
    (models / "wren").mkdir(parents=True)
    (models / "wren" / "model.bin").write_bytes(b"weights")
    (models / "mine").mkdir()
    (models / "mine" / "model.bin").write_bytes(b"trained here")
    monkeypatch.setattr(ms, "MODELS_DIR", str(models))
    monkeypatch.setattr(ms, "LARGE_FILE_BYTES", 5)
    tree = {"wren/model.bin": {"sha": sc.git_blob_sha1_file(str(models / "wren" / "model.bin")), "size": 7},
            "wren/config.json": {"sha": "abc", "size": 2}}
    monkeypatch.setattr(ms, "_fetch_remote_tree", lambda branch: tree)
    monkeypatch.setattr(ms.logmod, "ok", lambda *a: None)
    monkeypatch.setattr(ms.logmod, "error", lambda *a: None)
    return ms, models


def test_provenance_hides_local_trained_dirs_and_surfaces_remote_only_ones(rig):
    m, _models = rig
    assert m._provenance_table({"wren/model.bin": {}, "other/x.pt": {}}) == {
        "mine": "local-trained", "wren": "remote-pulled", "other": "remote-pulled"}


def test_files_classifies_against_git_blob_shas_and_flags_large_files(rig):
    m, _models = rig
    r = m.files()
    assert r["ok"] and r["provenance"]["mine"] == "local-trained"
    by_path = {row["path"]: row for row in r["files"]}
    assert by_path["wren/model.bin"]["state"] == sc.STATE_CURRENT and by_path["wren/model.bin"]["large"]
    assert by_path["wren/config.json"]["state"] == sc.STATE_MISSING and not by_path["wren/config.json"]["large"]
    assert r["counts"] == {sc.STATE_CURRENT: 1, sc.STATE_MISSING: 1}
    assert r["status"]["local_files"] == 2 and r["status"]["last"]["ok"] is True


def test_check_keeps_the_banner_shape(rig):
    m, _models = rig
    c = m.check()
    assert c["action"] == "check" and c["remote_files"] == 2 and c["new_files"] == 1 and c["modified"] == 0


def test_an_unreachable_remote_is_an_error_answer_not_an_exception(rig, monkeypatch):
    m, _models = rig

    def down(branch):
        raise urllib.error.URLError("no route to host")
    monkeypatch.setattr(m, "_fetch_remote_tree", down)
    r = m.files()
    assert r["ok"] is False and "no route to host" in r["error"] and r["status"]["last"]["ok"] is False
    assert m.check()["ok"] is False
