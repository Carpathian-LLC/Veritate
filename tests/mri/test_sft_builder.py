# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Regression: build_sft_idk_corpus.py must produce byte-deterministic ChatML
#   .bin files from JSONL family inputs (idk, chit-chat, jokes, in-domain) so
#   the sha256 of every rebuild matches. Non-determinism silently changes the
#   corpus every training run and breaks reproducibility.
# - Arch-agnostic: file I/O + json + hashlib, identical on all OS.
# tests/mri/test_sft_builder.py
# ------------------------------------------------------------------------------------
# Imports

import hashlib
import json
import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
BUILDER_DIR = os.path.join(REPO_ROOT, "veritate_mri", "tools")
if BUILDER_DIR not in sys.path:
    sys.path.insert(0, BUILDER_DIR)

import importlib.util

_builder_path = os.path.join(BUILDER_DIR, "build_sft_idk_corpus.py")
_spec = importlib.util.spec_from_file_location("build_sft_idk_corpus", _builder_path)
build_sft_idk_corpus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_sft_idk_corpus)


# ------------------------------------------------------------------------------------
# Constants

FAMILIES = ("idk_abstention.jsonl", "chitchat_greetings.jsonl",
            "jokes.jsonl", "in_domain_qa.jsonl")


# ------------------------------------------------------------------------------------
# Functions

def _make_fixture(in_dir, per_family=3):
    """Write a tiny 4-family SFT input tree the builder can consume."""
    for fname in FAMILIES:
        p = os.path.join(in_dir, fname)
        with open(p, "w", encoding="utf-8") as f:
            for i in range(per_family):
                f.write(json.dumps({"user": f"{fname[:3]} q{i}",
                                    "assistant": f"{fname[:3]} a{i}"}) + "\n")


def test_build_produces_chatml_frame(tmp_path):
    """Every pair renders as one ChatML turn separated by <|endoftext|>."""
    in_dir  = tmp_path / "in";  in_dir.mkdir()
    out_dir = tmp_path / "out"
    _make_fixture(str(in_dir), per_family=2)
    manifest = build_sft_idk_corpus.build(str(in_dir), str(out_dir),
                                          seed=42, val_ratio=0.1)
    train_path = out_dir / build_sft_idk_corpus.TRAIN_FILENAME
    body = train_path.read_bytes()
    assert b"<|im_start|>user\n"       in body
    assert b"<|im_end|>\n<|im_start|>assistant\n" in body
    assert b"<|endoftext|>\n"          in body
    assert manifest["total_pairs"] == 8   # 4 families x 2 each
    assert manifest["train_pairs"] + manifest["val_pairs"] == 8


def test_build_is_deterministic(tmp_path):
    """Same seed + same input == identical bytes. Catches nondeterministic
    shuffles or timestamp leaks in the future."""
    in_dir = tmp_path / "in"; in_dir.mkdir()
    _make_fixture(str(in_dir), per_family=3)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    m1 = build_sft_idk_corpus.build(str(in_dir), str(out1), seed=7, val_ratio=0.2)
    m2 = build_sft_idk_corpus.build(str(in_dir), str(out2), seed=7, val_ratio=0.2)

    b1 = (out1 / build_sft_idk_corpus.TRAIN_FILENAME).read_bytes()
    b2 = (out2 / build_sft_idk_corpus.TRAIN_FILENAME).read_bytes()
    assert hashlib.sha256(b1).hexdigest() == hashlib.sha256(b2).hexdigest()
    assert m1["train_sha256"] == m2["train_sha256"]
    assert m1["val_sha256"]   == m2["val_sha256"]


def test_build_rejects_bad_jsonl(tmp_path):
    """A malformed JSONL line surfaces as a clear ValueError, not a silent skip."""
    in_dir = tmp_path / "in"; in_dir.mkdir()
    _make_fixture(str(in_dir), per_family=1)
    # Corrupt one family file
    bad = os.path.join(str(in_dir), FAMILIES[0])
    with open(bad, "a", encoding="utf-8") as f:
        f.write("not valid json\n")

    try:
        build_sft_idk_corpus.build(str(in_dir), str(tmp_path / "out"),
                                   seed=1, val_ratio=0.1)
    except ValueError as e:
        assert "invalid JSON" in str(e)
    else:
        raise AssertionError("expected ValueError on malformed JSONL")


def test_build_rejects_empty_fields(tmp_path):
    """Empty user or assistant field raises rather than shipping a broken pair."""
    in_dir = tmp_path / "in"; in_dir.mkdir()
    # write minimal valid content to 3 families, and a bad line to the 4th
    _make_fixture(str(in_dir), per_family=1)
    bad = os.path.join(str(in_dir), FAMILIES[1])
    with open(bad, "w", encoding="utf-8") as f:
        f.write(json.dumps({"user": "hi", "assistant": ""}) + "\n")
    try:
        build_sft_idk_corpus.build(str(in_dir), str(tmp_path / "out"),
                                   seed=1, val_ratio=0.1)
    except ValueError as e:
        assert "missing/empty" in str(e)
    else:
        raise AssertionError("expected ValueError on empty assistant field")


def test_manifest_records_family_counts(tmp_path):
    """Manifest carries per-family counts and sha256 for downstream verification."""
    in_dir = tmp_path / "in"; in_dir.mkdir()
    _make_fixture(str(in_dir), per_family=4)
    m = build_sft_idk_corpus.build(str(in_dir), str(tmp_path / "out"),
                                   seed=99, val_ratio=0.25)
    assert set(m["family_counts"].keys()) == set(FAMILIES)
    for fam in FAMILIES:
        assert m["family_counts"][fam] == 4
    assert m["train_sha256"] and len(m["train_sha256"]) == 64
    assert m["val_sha256"]   and len(m["val_sha256"])   == 64
    assert m["license"] == build_sft_idk_corpus.LICENSE_FILENAME


def test_license_file_written(tmp_path):
    """LICENSE.md is written alongside the .bin (COS-upload requirement)."""
    in_dir = tmp_path / "in"; in_dir.mkdir()
    out_dir = tmp_path / "out"
    _make_fixture(str(in_dir), per_family=1)
    build_sft_idk_corpus.build(str(in_dir), str(out_dir), seed=1, val_ratio=0.5)
    lic = out_dir / build_sft_idk_corpus.LICENSE_FILENAME
    assert lic.exists()
    body = lic.read_text(encoding="utf-8")
    assert "Veritate" in body
    assert "abstention" in body.lower() or "I don't know" in body


def test_optional_corpus_dir_writes_stem(tmp_path):
    """When corpus_dir is given, the builder also writes sft_idk_{train,val}.bin
    there so multicorpus can resolve the stem without an extra copy step."""
    in_dir  = tmp_path / "in"; in_dir.mkdir()
    out_dir = tmp_path / "out"
    corp    = tmp_path / "corp"; corp.mkdir()
    _make_fixture(str(in_dir), per_family=2)
    m = build_sft_idk_corpus.build(str(in_dir), str(out_dir),
                                   seed=1, val_ratio=0.2, corpus_dir=str(corp))
    assert m["corpus_paths"] is not None
    assert (corp / "sft_idk_train.bin").exists()
    assert (corp / "sft_idk_val.bin").exists()
