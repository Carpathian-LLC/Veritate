# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Corpus bins moved from trainers/corpus/ to data/corpus/ on 2026-07-29, when the
#   trainer became platform code and `trainers/` stopped holding any. Existing
#   installs still have their 71-100 GB in the old place, so BOTH roots are read
#   and the old one must never go blind.
# - The invariants: an existing bin resolves wherever it lives; a brand-new stem
#   resolves to the canonical root so downloads land in the new location; listing
#   sees both roots and never double-counts a stem present in both.
# tests/mri/test_corpus_roots.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

MRI_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "veritate_mri")
if MRI_ROOT not in sys.path:
    sys.path.insert(0, MRI_ROOT)

from readers import corpus as corpus_reader
from readers import paths

# ------------------------------------------------------------------------------------
# Helpers


def _write(root, name, payload=b"\x00" * 32):
    os.makedirs(root, exist_ok=True)
    p = os.path.join(root, name)
    with open(p, "wb") as f:
        f.write(payload)
    return p


def _point_roots(monkeypatch, canonical, legacy):
    monkeypatch.setattr(paths, "CORPUS_ROOT", canonical)
    monkeypatch.setattr(paths, "LEGACY_CORPUS_ROOT", legacy)


# ------------------------------------------------------------------------------------
# Search order


def test_canonical_root_is_searched_first(monkeypatch, tmp_path):
    canonical, legacy = str(tmp_path / "data"), str(tmp_path / "trainers")
    _point_roots(monkeypatch, canonical, legacy)
    assert paths.corpus_search_dirs() == (canonical, legacy)


def test_corpus_dir_is_the_canonical_write_target(monkeypatch, tmp_path):
    canonical, legacy = str(tmp_path / "data"), str(tmp_path / "trainers")
    _point_roots(monkeypatch, canonical, legacy)
    assert paths.corpus_dir() == canonical


# ------------------------------------------------------------------------------------
# Resolution


def test_resolves_a_bin_that_only_exists_in_the_legacy_root(monkeypatch, tmp_path):
    canonical, legacy = str(tmp_path / "data"), str(tmp_path / "trainers")
    _point_roots(monkeypatch, canonical, legacy)
    want = _write(legacy, "old_train.bin")
    assert paths.corpus_train_path("old") == want


def test_prefers_the_canonical_root_when_a_stem_exists_in_both(monkeypatch, tmp_path):
    canonical, legacy = str(tmp_path / "data"), str(tmp_path / "trainers")
    _point_roots(monkeypatch, canonical, legacy)
    _write(legacy, "both_train.bin")
    want = _write(canonical, "both_train.bin")
    assert paths.corpus_train_path("both") == want


def test_unknown_stem_resolves_to_the_canonical_root(monkeypatch, tmp_path):
    """So a download of a corpus nobody has yet lands in data/corpus/."""
    canonical, legacy = str(tmp_path / "data"), str(tmp_path / "trainers")
    _point_roots(monkeypatch, canonical, legacy)
    got = paths.corpus_train_path("brand_new")
    assert got == os.path.join(canonical, "brand_new_train.bin")
    assert not os.path.exists(got)


def test_val_path_follows_the_same_two_root_rule(monkeypatch, tmp_path):
    canonical, legacy = str(tmp_path / "data"), str(tmp_path / "trainers")
    _point_roots(monkeypatch, canonical, legacy)
    want = _write(legacy, "old_val.bin")
    assert paths.corpus_val_path("old") == want
    assert paths.corpus_val_path("nope") == os.path.join(canonical, "nope_val.bin")


# ------------------------------------------------------------------------------------
# Listing


def test_list_stems_sees_both_roots_without_duplicating(monkeypatch, tmp_path):
    canonical, legacy = str(tmp_path / "data"), str(tmp_path / "trainers")
    _point_roots(monkeypatch, canonical, legacy)
    monkeypatch.setattr(corpus_reader.plugins_reader, "scan", list)
    _write(canonical, "fresh_train.bin")
    _write(legacy, "legacy_only_train.bin")
    _write(canonical, "shared_train.bin")
    _write(legacy, "shared_train.bin")

    stems = [e["stem"] for e in corpus_reader.list_stems()]
    assert sorted(stems) == ["fresh", "legacy_only", "shared"]
    assert stems.count("shared") == 1


def test_resolve_paths_returns_none_when_the_bin_is_in_neither_root(monkeypatch, tmp_path):
    canonical, legacy = str(tmp_path / "data"), str(tmp_path / "trainers")
    _point_roots(monkeypatch, canonical, legacy)
    assert corpus_reader.resolve_paths("ghost") == (None, None)


def test_resolve_paths_finds_a_legacy_train_bin_and_its_val(monkeypatch, tmp_path):
    canonical, legacy = str(tmp_path / "data"), str(tmp_path / "trainers")
    _point_roots(monkeypatch, canonical, legacy)
    _write(legacy, "old_train.bin")
    _write(legacy, "old_val.bin")
    train, val = corpus_reader.resolve_paths("old")
    assert train == os.path.join(legacy, "old_train.bin")
    assert val == os.path.join(legacy, "old_val.bin")
