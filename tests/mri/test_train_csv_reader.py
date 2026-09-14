# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - readers/train_csv is the data-access layer under every loss curve the dashboard draws
#   and every val row a stop rule or a sleep gate reads. It must never raise on a file
#   being appended to by a live trainer: a torn last line, a missing file, a column added
#   by a newer trainer. Pinned here: numeric coercion, a row kept when a value is blank,
#   the mtime cache invalidating on a write, and empty results rather than exceptions.
# tests/mri/test_train_csv_reader.py
# ------------------------------------------------------------------------------------
# Imports:

import os

from readers import train_csv

# ------------------------------------------------------------------------------------
# Constants

HEADER = "step,split,loss,lr,grad_norm,tok_per_s,wall_s,seed"

# ------------------------------------------------------------------------------------
# Functions


def _csv(tmp_path, monkeypatch, body):
    p = tmp_path / "train.csv"
    p.write_text(body, encoding="utf-8")
    monkeypatch.setattr(train_csv.paths, "train_csv_path", lambda _n: str(p))
    train_csv._CACHE.clear()
    return p


def test_rows_come_back_typed(tmp_path, monkeypatch):
    _csv(tmp_path, monkeypatch, f"{HEADER}\n70200,train,0.4063,3e-05,0.82,258,111.0,0\n")
    row = train_csv.load("m")[0]
    assert row["step"] == 70200 and isinstance(row["step"], int)
    assert row["loss"] == 0.4063 and row["lr"] == 3e-05
    assert row["split"] == "train"


def test_a_blank_value_becomes_none_and_keeps_the_row(tmp_path, monkeypatch):
    """A val row carries no grad_norm or throughput. Dropping it would erase the val
    curve, which is the one a stop rule reads."""
    _csv(tmp_path, monkeypatch, f"{HEADER}\n70400,val,0.3827,3e-05,,,22783.3,0\n")
    row = train_csv.load("m")[0]
    assert row["split"] == "val" and row["loss"] == 0.3827
    assert row["grad_norm"] is None and row["tok_per_s"] is None


def test_a_torn_last_line_is_skipped_not_raised(tmp_path, monkeypatch):
    """The file is being appended to by a live run; the reader may see half a line."""
    _csv(tmp_path, monkeypatch, f"{HEADER}\n1,train,0.5,3e-05,0.1,10,1.0,0\n2,train,0.4,3e")
    rows = train_csv.load("m")
    assert [r["step"] for r in rows] == [1]


def test_an_unparseable_step_drops_the_row(tmp_path, monkeypatch):
    _csv(tmp_path, monkeypatch, f"{HEADER}\nNaNstep,train,0.5,3e-05,0.1,10,1.0,0\n")
    assert train_csv.load("m") == []


def test_a_bad_number_becomes_none_rather_than_dropping_the_row(tmp_path, monkeypatch):
    _csv(tmp_path, monkeypatch, f"{HEADER}\n5,train,notanumber,3e-05,0.1,10,1.0,0\n")
    row = train_csv.load("m")[0]
    assert row["step"] == 5 and row["loss"] is None


def test_a_missing_file_reads_as_empty(tmp_path, monkeypatch):
    """readers return None or empty rather than raising."""
    monkeypatch.setattr(train_csv.paths, "train_csv_path", lambda _n: str(tmp_path / "absent.csv"))
    train_csv._CACHE.clear()
    assert train_csv.load("m") == []
    assert train_csv.raw_text("m") == ""
    assert train_csv.is_present("m") is False
    assert train_csv.file_stat("m") is None


def test_a_header_only_file_reads_as_empty(tmp_path, monkeypatch):
    _csv(tmp_path, monkeypatch, HEADER + "\n")
    assert train_csv.load("m") == []


def test_an_empty_file_reads_as_empty(tmp_path, monkeypatch):
    _csv(tmp_path, monkeypatch, "")
    assert train_csv.load("m") == []


def test_the_cache_invalidates_when_the_run_appends(tmp_path, monkeypatch):
    """Cached on mtime: a live run's new rows must appear without a restart."""
    p = _csv(tmp_path, monkeypatch, f"{HEADER}\n1,train,0.5,3e-05,0.1,10,1.0,0\n")
    assert len(train_csv.load("m")) == 1
    with open(p, "a", encoding="utf-8") as f:
        f.write("2,train,0.4,3e-05,0.1,10,2.0,0\n")
    os.utime(p, (1_800_000_000, 1_800_000_000))
    assert [r["step"] for r in train_csv.load("m")] == [1, 2]


def test_a_row_with_the_wrong_column_count_is_skipped(tmp_path, monkeypatch):
    """A trainer that added a column writes rows this reader must not misalign."""
    _csv(tmp_path, monkeypatch, f"{HEADER}\n1,train,0.5,3e-05,0.1,10,1.0,0,extra\n")
    assert train_csv.load("m") == []
