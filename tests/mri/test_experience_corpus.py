# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers tools/build_experience_corpus.py, sleep's "index its thoughts" step
#   (IDEA 20 E4). Pins: exchanges reconstruct as prompt+reply+closing marker,
#   exact duplicates dedupe, sub-minimum replies drop, torn JSONL lines are
#   skipped without killing the build, the train/val split interleaves, and the
#   --days window selects only the newest files.
# tests/mri/test_experience_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import base64
import json

import pytest
from tools import build_experience_corpus as bec

# ------------------------------------------------------------------------------------
# Functions


def _rec(prompt, output):
    return json.dumps({"ts": 0, "model": "m",
                       "prompt_b64": base64.b64encode(prompt).decode(),
                       "output_b64": base64.b64encode(output).decode()})


@pytest.fixture()
def roots(tmp_path, monkeypatch):
    exp = tmp_path / "experience"
    exp.mkdir()
    out = tmp_path / "corpus"
    monkeypatch.setattr(bec, "EXPERIENCE_ROOT", str(exp))
    return exp, out


def test_build_reconstructs_and_dedupes(roots):
    exp, out = roots
    lines = [_rec(b"<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", b"hello there"),
             _rec(b"<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", b"hello there"),
             _rec(b"P2", b"reply two goes here")]
    (exp / "20260819.jsonl").write_text("\n".join(lines) + "\n")
    n, _tb, vb = bec.build(out_dir=str(out), val_frac=0.0)
    assert n == 2                                    # duplicate collapsed
    data = (out / "experience_train.bin").read_bytes()
    assert b"hello there<|im_end|>\n" in data        # closing marker appended
    assert data.count(b"hello there") == 1
    assert b"P2reply two goes here<|im_end|>\n" in data
    assert vb == 0


def test_short_replies_and_torn_lines_skipped(roots):
    exp, out = roots
    (exp / "20260819.jsonl").write_text(
        _rec(b"P", b"ok") + "\n" +                   # under min-reply
        "{torn json\n" +
        _rec(b"P", b"a full length reply") + "\n")
    n, _tb, _ = bec.build(out_dir=str(out), val_frac=0.0)
    assert n == 1
    assert b"a full length reply" in (out / "experience_train.bin").read_bytes()


def test_val_split_interleaves(roots):
    exp, out = roots
    lines = [_rec(f"P{i}".encode(), f"reply number {i} padded".encode()) for i in range(40)]
    (exp / "20260819.jsonl").write_text("\n".join(lines) + "\n")
    n, _tb, vb = bec.build(out_dir=str(out), val_frac=0.10)
    assert n == 40 and vb > 0
    assert (out / "experience_val.bin").read_bytes().count(b"reply number") == 4


def test_days_window(roots):
    exp, out = roots
    (exp / "20260817.jsonl").write_text(_rec(b"OLD", b"old day reply here") + "\n")
    (exp / "20260819.jsonl").write_text(_rec(b"NEW", b"new day reply here") + "\n")
    bec.build(out_dir=str(out), days=1, val_frac=0.0)
    data = (out / "experience_train.bin").read_bytes()
    assert b"new day reply" in data and b"old day reply" not in data


def test_min_val_bytes_floors_the_val_bin(roots):
    """A night too small for the val_frac stride still fills val to the draw window."""
    exp, out = roots
    lines = [_rec(f"P{i}".encode(), (f"reply number {i} " + "x" * 90).encode())
             for i in range(10)]
    (exp / "20260819.jsonl").write_text("\n".join(lines) + "\n")
    n, tb, vb = bec.build(out_dir=str(out), val_frac=0.05, min_val_bytes=300)
    assert n == 10 and tb > 0
    assert vb >= 300
