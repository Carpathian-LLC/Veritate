# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Regression: build_sft_corpus.py must produce byte-deterministic ChatML bins
#   from single-turn AND multi-turn JSONL rows across arches (rule 34d). Sibling
#   coverage to test_sft_builder.py (which pins the IDK-specific builder).
# tests/mri/test_sft_corpus_builder.py
# ------------------------------------------------------------------------------------
# Imports

import hashlib
import importlib.util
import json
import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
_BUILDER_PATH = os.path.join(REPO_ROOT, "veritate_mri", "tools", "build_sft_corpus.py")
_spec = importlib.util.spec_from_file_location("build_sft_corpus", _BUILDER_PATH)
build_sft_corpus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_sft_corpus)


# ------------------------------------------------------------------------------------
# Constants

STEM         = "sft_probe_v1"
FAMILY_ONE   = "family_a.jsonl"
FAMILY_TWO   = "family_b.jsonl"
PURPOSE      = "Test purpose."
LICENSE_TEXT = "# Test License\n\nFor use with Veritate.\n"


# ------------------------------------------------------------------------------------
# Functions


def _write_single_turn(path, n):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"user": f"q{i}", "assistant": f"a{i}"}) + "\n")


def _write_multi_turn(path, n):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"turns": [
                {"role": "user", "content": f"hello {i}"},
                {"role": "assistant", "content": f"hi {i}"},
                {"role": "user", "content": f"follow-up {i}"},
                {"role": "assistant", "content": f"answer {i}"},
            ]}) + "\n")


def test_builder_emits_chatml_frame(tmp_path):
    """Every row renders as a full ChatML conversation ending in <|endoftext|>."""
    in_dir  = tmp_path / "in";  in_dir.mkdir()
    out_dir = tmp_path / "out"
    _write_single_turn(str(in_dir / FAMILY_ONE), 3)
    _write_multi_turn(str(in_dir / FAMILY_TWO), 2)
    m = build_sft_corpus.build(STEM, str(in_dir), str(out_dir),
                                (FAMILY_ONE, FAMILY_TWO), PURPOSE, LICENSE_TEXT,
                                seed=1, val_ratio=0.2)
    body = (out_dir / f"{STEM}_train.bin").read_bytes()
    assert b"<|im_start|>user\n"                   in body
    assert b"<|im_end|>\n<|im_start|>assistant\n"  in body
    assert b"<|endoftext|>\n"                      in body
    assert m["family_counts"] == {FAMILY_ONE: 3, FAMILY_TWO: 2}


def test_builder_is_deterministic(tmp_path):
    """Same seed + same input == identical bytes across rebuilds."""
    in_dir = tmp_path / "in"; in_dir.mkdir()
    _write_single_turn(str(in_dir / FAMILY_ONE), 4)
    _write_multi_turn(str(in_dir / FAMILY_TWO), 3)
    o1 = tmp_path / "o1"; o2 = tmp_path / "o2"
    m1 = build_sft_corpus.build(STEM, str(in_dir), str(o1),
                                 (FAMILY_ONE, FAMILY_TWO), PURPOSE, LICENSE_TEXT,
                                 seed=99, val_ratio=0.25)
    m2 = build_sft_corpus.build(STEM, str(in_dir), str(o2),
                                 (FAMILY_ONE, FAMILY_TWO), PURPOSE, LICENSE_TEXT,
                                 seed=99, val_ratio=0.25)
    b1 = (o1 / f"{STEM}_train.bin").read_bytes()
    b2 = (o2 / f"{STEM}_train.bin").read_bytes()
    assert hashlib.sha256(b1).hexdigest() == hashlib.sha256(b2).hexdigest()
    assert m1["train_sha256"] == m2["train_sha256"]


def test_prose_row_renders_without_a_chatml_frame(tmp_path):
    """A {"text": ...} row emits bare prose plus <|endoftext|>, no fabricated user turn."""
    in_dir = tmp_path / "in"; in_dir.mkdir()
    with open(in_dir / FAMILY_ONE, "w", encoding="utf-8") as f:
        for i in range(4):
            f.write(json.dumps({"text": f"the ferry was late again on day {i}"}) + "\n")
    build_sft_corpus.build(STEM, str(in_dir), str(tmp_path / "out"),
                            (FAMILY_ONE,), PURPOSE, LICENSE_TEXT,
                            seed=1, val_ratio=0.25)
    body = (tmp_path / "out" / f"{STEM}_train.bin").read_bytes()
    assert b"<|im_start|>" not in body
    assert body.count(b"<|endoftext|>\n") == 3
    assert body.startswith(b"the ferry was late again on day ")


def test_builder_rejects_bad_jsonl(tmp_path):
    """Malformed JSONL surfaces as a clear ValueError, not a silent skip."""
    in_dir = tmp_path / "in"; in_dir.mkdir()
    _write_single_turn(str(in_dir / FAMILY_ONE), 1)
    with open(in_dir / FAMILY_ONE, "a", encoding="utf-8") as f:
        f.write("not valid json\n")
    try:
        build_sft_corpus.build(STEM, str(in_dir), str(tmp_path / "out"),
                                (FAMILY_ONE,), PURPOSE, LICENSE_TEXT,
                                seed=1, val_ratio=0.1)
    except ValueError as e:
        assert "invalid JSON" in str(e)
    else:
        raise AssertionError("expected ValueError on malformed JSONL")


def test_builder_rejects_bad_turns(tmp_path):
    """Multi-turn row with an unknown role raises rather than shipping garbage."""
    in_dir = tmp_path / "in"; in_dir.mkdir()
    with open(in_dir / FAMILY_ONE, "w", encoding="utf-8") as f:
        f.write(json.dumps({"turns": [{"role": "system", "content": "x"}]}) + "\n")
    try:
        build_sft_corpus.build(STEM, str(in_dir), str(tmp_path / "out"),
                                (FAMILY_ONE,), PURPOSE, LICENSE_TEXT,
                                seed=1, val_ratio=0.1)
    except ValueError as e:
        assert "invalid turn" in str(e)
    else:
        raise AssertionError("expected ValueError on unknown role")


def test_optional_corpus_dir_writes_stem(tmp_path):
    """corpus_dir installs <stem>_{train,val}.bin so the trainer resolves the stem."""
    in_dir = tmp_path / "in"; in_dir.mkdir()
    _write_single_turn(str(in_dir / FAMILY_ONE), 4)
    out_dir = tmp_path / "out"
    corp    = tmp_path / "corp"; corp.mkdir()
    m = build_sft_corpus.build(STEM, str(in_dir), str(out_dir),
                                (FAMILY_ONE,), PURPOSE, LICENSE_TEXT,
                                seed=1, val_ratio=0.25, corpus_dir=str(corp))
    assert (corp / f"{STEM}_train.bin").exists()
    assert (corp / f"{STEM}_val.bin").exists()
    assert m["corpus_paths"]["stem"] == STEM


def test_manifest_shape(tmp_path):
    """Manifest carries every field a downstream verifier needs."""
    in_dir = tmp_path / "in"; in_dir.mkdir()
    _write_single_turn(str(in_dir / FAMILY_ONE), 5)
    m = build_sft_corpus.build(STEM, str(in_dir), str(tmp_path / "out"),
                                (FAMILY_ONE,), PURPOSE, LICENSE_TEXT,
                                seed=42, val_ratio=0.2)
    assert m["name"] == f"veritate_{STEM}"
    assert m["byte_template"] == "ChatML"
    assert len(m["train_sha256"]) == 64
    assert len(m["val_sha256"])   == 64
    assert m["train_bytes"] > 0 and m["val_bytes"] > 0
