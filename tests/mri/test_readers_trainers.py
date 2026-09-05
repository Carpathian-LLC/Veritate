# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - direct unit tests for readers/trainers.py's size-table plumbing: load_sizes_doc,
#   size_defaults, all_size_defaults, and the native trainer manifest/record. Uses
#   native_sizes_path() as the override seam so the merge logic is tested against a
#   small fixture doc instead of the real (34-size) trainer_sizes.json.
# tests/mri/test_readers_trainers.py
# ------------------------------------------------------------------------------------
# Imports:

import json

from readers import trainers

# ------------------------------------------------------------------------------------
# Helpers

SIZES_DOC = {
    "shared_defaults": {"seq": 512, "batch_size": 8, "base_lr": 0.001},
    "sizes": {
        "10m": {"shape": {"layers": 4, "hidden": 128, "ffn": 512, "heads": 4},
                "defaults": {"batch_size": 16}},
        "200m": {"shape": {"layers": 12, "hidden": 768, "ffn": 3072, "heads": 12}},
    },
    "default_size": "10m",
}


def _patched(monkeypatch, tmp_path, doc=SIZES_DOC):
    path = tmp_path / "trainer_sizes.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(trainers, "native_sizes_path", lambda: str(path))


# ------------------------------------------------------------------------------------
# load_sizes_doc


def test_load_sizes_doc_returns_the_doc_at_the_resolved_path(tmp_path, monkeypatch):
    """load_sizes_doc reads and parses whatever native_sizes_path() points at."""
    _patched(monkeypatch, tmp_path)
    assert trainers.load_sizes_doc() == SIZES_DOC


# ------------------------------------------------------------------------------------
# size_defaults


def test_size_defaults_merges_shared_defaults_into_the_size(tmp_path, monkeypatch):
    """size_defaults('200m') carries shared_defaults keys the size did not override."""
    _patched(monkeypatch, tmp_path)
    out = trainers.size_defaults("200m")
    assert out["seq"] == 512
    assert out["base_lr"] == 0.001
    assert out["size"] == "200m"


def test_size_defaults_size_wins_over_shared_on_conflict(tmp_path, monkeypatch):
    """A size's own defaults override the shared default for the same key."""
    _patched(monkeypatch, tmp_path)
    out = trainers.size_defaults("10m")
    assert out["batch_size"] == 16


def test_default_size_key_selects_the_prefilled_size(tmp_path, monkeypatch):
    """The doc's default_size key names which size a fresh training form prefills."""
    _patched(monkeypatch, tmp_path)
    doc = trainers.load_sizes_doc()
    out = trainers.size_defaults(doc[trainers.DEFAULT_SIZE_KEY])
    assert out["size"] == "10m"
    assert out["batch_size"] == 16


# ------------------------------------------------------------------------------------
# all_size_defaults


def test_all_size_defaults_covers_every_size_in_the_doc(tmp_path, monkeypatch):
    """all_size_defaults returns exactly one merged entry per size in the doc."""
    _patched(monkeypatch, tmp_path)
    out = trainers.all_size_defaults()
    assert set(out.keys()) == set(SIZES_DOC["sizes"].keys())


def test_all_size_defaults_each_entry_is_independently_merged(tmp_path, monkeypatch):
    """Each size's merged entry in all_size_defaults keeps that size's own override."""
    _patched(monkeypatch, tmp_path)
    out = trainers.all_size_defaults()
    assert out["10m"]["batch_size"] == 16
    assert out["200m"]["batch_size"] == 8


# ------------------------------------------------------------------------------------
# native trainer manifest / record


def test_native_trainer_manifest_sizes_covers_every_size():
    """NATIVE_TRAINER_MANIFEST's sizes match the real shipped size -> shape table."""
    assert set(trainers.NATIVE_TRAINER_MANIFEST["sizes"].keys()) == set(trainers.load_native_sizes().keys())


def test_native_record_exposes_the_native_trainer_id():
    """_native_record() carries NATIVE_TRAINER_ID and is flagged native."""
    rec = trainers._native_record()
    assert rec["id"] == trainers.NATIVE_TRAINER_ID
    assert rec["native"] is True


# ------------------------------------------------------------------------------------
# image trainer record


def test_scan_lists_the_image_trainer_beside_the_native_one():
    """The Training tab's picker is built from scan(); the image trainer must be in it."""
    ids = [p["id"] for p in trainers.scan()]
    assert trainers.NATIVE_TRAINER_ID in ids
    assert trainers.IMAGE_TRAINER_ID in ids


def test_image_record_is_an_image_flow_trainer_with_the_full_size_table():
    rec = trainers._image_record()
    assert rec["id"] == trainers.IMAGE_TRAINER_ID
    assert rec["native"] is True
    assert rec["manifest"]["flow"] == ["image"]
    assert rec["manifest"]["kind"] == "trainer"
    assert set(rec["manifest"]["sizes"]) == set(trainers.load_native_sizes())


def test_image_manifest_carries_no_per_size_text_tuning():
    """The text table's per-size defaults (seq 512 at 10m, QAT on) would override the
    image geometry; the image manifest must not inherit them."""
    assert trainers.IMAGE_TRAINER_MANIFEST["size_defaults"] == {}


def test_image_defaults_fix_what_an_image_model_is():
    d = trainers.IMAGE_TRAINER_DEFAULTS
    assert d["trunk"] == "dense"
    assert d["objective"] == "masked_grid"
    assert d["hooks"] == "off"
    assert d["seq"] == 0                       # derived from geometry
    assert d["height"] % d["patch"] == 0 and d["width"] % d["patch"] == 0
    assert d["planes"] * (d["height"] // d["patch"]) * (d["width"] // d["patch"]) <= 2048


def test_image_trainer_is_not_bench_capable():
    """No auto-tune on the image trainer: the bench measures the text loop."""
    assert trainers.IMAGE_TRAINER_MANIFEST["bench"] is False
