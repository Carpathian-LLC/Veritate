# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - DEFAULT_SKIP_DIRS is matched against the FIRST path segment only. The whole
#   trainer consolidation rests on that: repo-root data/corpus/ must survive an
#   update untouched (100 GB of bins), while veritate_mri/data/trainer_sizes.json
#   must ship WITH the update like any other tracked file. Same directory name,
#   opposite outcomes, decided purely by depth.
# - If a future edit prunes by basename at any depth, trainer_sizes.json stops
#   arriving and every install silently keeps whatever size table it already had.
# tests/mri/test_updater_skip_depth.py
# ------------------------------------------------------------------------------------
# Imports:

import os

from training.sync import app_sync

# ------------------------------------------------------------------------------------
# Helpers


def _write(root, rel, text="x\n"):
    path = os.path.join(root, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _tracked(root, skip=None):
    skip = set(skip if skip is not None else app_sync.DEFAULT_SKIP_DIRS)
    return {rel for rel, _abs in app_sync._iter_tracked(str(root), skip)}


# ------------------------------------------------------------------------------------
# The two halves of the invariant


def test_repo_root_data_is_skipped(tmp_path):
    _write(tmp_path, "data/corpus/fineweb_edu_train.bin")
    _write(tmp_path, "data/mri_settings.json")
    assert _tracked(tmp_path) == set()


def test_nested_data_dir_is_not_skipped(tmp_path):
    """veritate_mri/data/trainer_sizes.json must ride the update."""
    _write(tmp_path, "veritate_mri/data/trainer_sizes.json")
    assert _tracked(tmp_path) == {"veritate_mri/data/trainer_sizes.json"}


def test_both_at_once(tmp_path):
    _write(tmp_path, "data/corpus/the_pile_train.bin")
    _write(tmp_path, "veritate_mri/data/trainer_sizes.json")
    _write(tmp_path, "veritate_mri/training/veritate_trainer.py")
    assert _tracked(tmp_path) == {
        "veritate_mri/data/trainer_sizes.json",
        "veritate_mri/training/veritate_trainer.py",
    }


# ------------------------------------------------------------------------------------
# The rest of the skip set behaves the same way


def test_every_skip_name_is_top_level_only(tmp_path):
    for name in app_sync.DEFAULT_SKIP_DIRS:
        _write(tmp_path, f"{name}/skipped.txt")
        _write(tmp_path, f"veritate_mri/{name}/kept.txt")
    got = _tracked(tmp_path)
    assert not any(r.endswith("skipped.txt") for r in got)
    assert len(got) == len(app_sync.DEFAULT_SKIP_DIRS)


def test_legacy_trainers_corpus_is_skipped_at_the_root(tmp_path):
    """Installs predating 2026-07-29 still hold their bins under trainers/."""
    _write(tmp_path, "trainers/corpus/openwebtext10g_train.bin")
    _write(tmp_path, "trainers/.gitignore")
    assert _tracked(tmp_path) == set()


def test_rels_are_forward_slashed(tmp_path):
    _write(tmp_path, "veritate_mri/readers/trainers.py")
    assert _tracked(tmp_path) == {"veritate_mri/readers/trainers.py"}
