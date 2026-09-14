# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - tools/jsonl_to_bin: every accepted trace form (byte list, b64:, utf-8) lands in the
#   bin joined by the separator, garbage lines are counted not fatal, the trailing
#   fraction goes to the val bin, max_records caps the read, and the refusals name
#   their cause.
# tests/corpus/test_jsonl_to_bin.py
# ------------------------------------------------------------------------------------
# Imports:

import base64
import json

import pytest
from jsonl_to_bin import jsonl_to_bin

# ------------------------------------------------------------------------------------
# Functions


def _write(path, rows):
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


def test_every_trace_form_lands_in_the_bin_joined_by_the_separator(tmp_path):
    src = _write(tmp_path / "t.jsonl", [
        json.dumps({"trace_bytes": [104, 105]}),
        json.dumps({"trace_bytes": "b64:" + base64.b64encode(b"yo").decode()}),
        json.dumps({"trace_bytes": "hé"}),
        "not json at all",
        json.dumps({"other": 1}),
    ])
    out = tmp_path / "x_train.bin"
    stats = jsonl_to_bin(src, str(out), separator=b"|")
    assert out.read_bytes() == b"hi|yo|h\xc3\xa9"
    assert (stats["n_records"], stats["n_skipped"], stats["val_bin"]) == (3, 2, None)


def test_the_trailing_fraction_goes_to_the_val_bin_and_max_records_caps_the_read(tmp_path):
    src = _write(tmp_path / "t.jsonl", [json.dumps({"trace_bytes": [65 + i]}) for i in range(10)])
    out = tmp_path / "y_train.bin"
    stats = jsonl_to_bin(src, str(out), separator=b"", val_split_ratio=0.2)
    assert out.read_bytes() == b"ABCDEFGH" and (tmp_path / "y_val.bin").read_bytes() == b"IJ"
    assert (stats["n_train"], stats["n_val"]) == (8, 2)
    stats = jsonl_to_bin(src, str(out), separator=b"", max_records=3)
    assert out.read_bytes() == b"ABC" and stats["n_records"] == 3


def test_refusals_name_their_cause(tmp_path):
    with pytest.raises(FileNotFoundError):
        jsonl_to_bin(str(tmp_path / "missing.jsonl"), str(tmp_path / "o.bin"))
    src = _write(tmp_path / "t.jsonl", [json.dumps({"trace_bytes": [1]})])
    with pytest.raises(ValueError, match="val_split_ratio"):
        jsonl_to_bin(src, str(tmp_path / "o.bin"), val_split_ratio=0.9)
    empty = _write(tmp_path / "e.jsonl", ["{}", ""])
    with pytest.raises(RuntimeError, match="no usable records"):
        jsonl_to_bin(empty, str(tmp_path / "o.bin"))
