# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - A virtualenv named anything other than venv/.venv was walked by the "added"
#   pass of local_edits, so every .py inside it was reported as a file the user
#   added. One .veritate_venv turned a clean tree into 8,837 "changes", which
#   reads as catastrophic drift and invites someone to delete the checkout.
# - The test for a venv is pyvenv.cfg, not a name list, so a venv called env3 or
#   build_env is caught too. Dot-dirs go as tooling caches.
# tests/mri/test_local_noise_dirs.py
# ------------------------------------------------------------------------------------
# Imports:

import os

from training.sync import app_sync

# ------------------------------------------------------------------------------------
# Helpers


def _mkdir(root, *parts):
    p = os.path.join(root, *parts)
    os.makedirs(p, exist_ok=True)
    return p


def _touch(path, name, text="x\n"):
    with open(os.path.join(path, name), "w", encoding="utf-8", newline="") as f:
        f.write(text)


def _repo(tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(app_sync, "REPO_DIR", root)
    return root


# ------------------------------------------------------------------------------------
# What counts as noise


def test_a_venv_under_any_name_is_noise(tmp_path, monkeypatch):
    root = _repo(tmp_path, monkeypatch)
    for name in (".veritate_venv", "build_env", "env3"):
        d = _mkdir(root, name)
        _touch(d, "pyvenv.cfg")
    assert app_sync._local_noise_dirs() >= {".veritate_venv", "build_env", "env3"}


def test_dot_dirs_are_noise(tmp_path, monkeypatch):
    root = _repo(tmp_path, monkeypatch)
    for name in (".pytest_cache", ".ruff_cache", ".idea"):
        _mkdir(root, name)
    assert app_sync._local_noise_dirs() >= {".pytest_cache", ".ruff_cache", ".idea"}


def test_real_source_dirs_are_not_noise(tmp_path, monkeypatch):
    root = _repo(tmp_path, monkeypatch)
    for name in ("veritate_mri", "veritate_core", "tests", "extensions"):
        _mkdir(root, name)
    assert app_sync._local_noise_dirs() == set()


def test_a_plain_dir_named_like_a_venv_without_the_marker_is_kept(tmp_path, monkeypatch):
    """No pyvenv.cfg means it is somebody's source dir, not an interpreter."""
    root = _repo(tmp_path, monkeypatch)
    _mkdir(root, "environment")
    assert "environment" not in app_sync._local_noise_dirs()


def test_files_are_never_reported_as_dirs(tmp_path, monkeypatch):
    root = _repo(tmp_path, monkeypatch)
    _touch(root, ".plugin_pid.json")
    assert ".plugin_pid.json" not in app_sync._local_noise_dirs()


def test_missing_repo_dir_returns_empty_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(app_sync, "REPO_DIR", str(tmp_path / "gone"))
    assert app_sync._local_noise_dirs() == set()


# ------------------------------------------------------------------------------------
# The effect on the added list


def test_venv_contents_stay_out_of_the_added_list(tmp_path, monkeypatch):
    root = _repo(tmp_path, monkeypatch)
    monkeypatch.setattr(app_sync, "BASELINE_PATH",
                        os.path.join(root, "data", "baseline.json"))
    venv = _mkdir(root, ".veritate_venv", "lib", "site-packages")
    for i in range(50):
        _touch(venv, f"mod{i}.py")
    src = _mkdir(root, "veritate_core")
    _touch(src, "model.py")
    _touch(src, "mine.py")

    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    with open(app_sync.BASELINE_PATH, "w", encoding="utf-8") as f:
        f.write('{"version":1,"written_at":0,"branch":"dev",'
                '"files":{"veritate_core/model.py":"deadbeef"}}')

    added = [e["path"] for e in app_sync.local_edits()["added"]]
    assert added == ["veritate_core/mine.py"]
