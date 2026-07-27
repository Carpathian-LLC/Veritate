# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - GET /run/<name>/csv over a train.csv written by save.append_train_row (rule 21):
#   the served header pins the per-step schema and its column order, rows keep the
#   header's field count, and missing runs 404 instead of serving an empty body.
# tests/mri/test_runs_csv.py
# ------------------------------------------------------------------------------------
# Imports:

import json

import pytest
from flask import Flask
from readers import paths
from routes import runs_routes
from training import save

# ------------------------------------------------------------------------------------
# Constants

NAME = "csv_model"

# The per-step schema every dashboard reader parses. A change to what
# append_train_row writes must break this list, not silently reshape the CSV.
EXPECTED_COLUMNS = ["step", "split", "loss", "lr", "grad_norm", "tok_per_s", "wall_s", "seed"]

FULL_ROW = {"step": 7, "split": "train", "loss": 1.5, "lr": 0.0003,
            "grad_norm": 0.25, "tok_per_s": 1234.5, "wall_s": 2.0, "seed": 11}
FULL_ROW_SERIALIZED = ["7", "train", "1.500000", "3.000000e-04", "0.250000",
                       "1234.50", "2.000", "11"]
OPTIONAL_COLUMNS = ["lr", "grad_norm", "tok_per_s", "wall_s", "seed"]

CSV_MIMETYPE = "text/csv"

# ------------------------------------------------------------------------------------
# Functions

def _client(monkeypatch, tmp_path, rows=(FULL_ROW,), make_dir=True):
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path))
    if make_dir:
        (tmp_path / NAME).mkdir(parents=True)
        with open(tmp_path / NAME / "config.json", "w", encoding="utf-8") as f:
            json.dump({"description": "csv fixture"}, f)
    for row in rows:
        save.append_train_row(NAME, **row)
    app = Flask(__name__)
    runs_routes.register(app)
    return app.test_client()


def _lines(resp):
    return [ln for ln in resp.get_data(as_text=True).splitlines() if ln]


@pytest.fixture
def served(monkeypatch, tmp_path):
    return _client(monkeypatch, tmp_path).get(f"/run/{NAME}/csv")


def test_csv_header_lists_every_column_in_order(served):
    """The served header is exactly the columns append_train_row writes, in order."""
    assert _lines(served)[0].split(",") == EXPECTED_COLUMNS


def test_csv_row_field_count_matches_header(served):
    """Each data row carries one field per header column."""
    assert len(_lines(served)[1].split(",")) == len(EXPECTED_COLUMNS)


def test_csv_row_values_land_in_their_columns(served):
    """A fully populated row serializes to the documented per-column formats."""
    assert _lines(served)[1].split(",") == FULL_ROW_SERIALIZED


def test_csv_response_mimetype_is_csv(served):
    """The endpoint serves text/csv."""
    assert served.mimetype == CSV_MIMETYPE


def test_csv_omitted_optionals_are_empty_fields(monkeypatch, tmp_path):
    """Optional metrics left None serialize as empty fields, keeping the schema width."""
    client = _client(monkeypatch, tmp_path, rows=({"step": 1, "split": "val", "loss": 2.0},))
    fields = _lines(client.get(f"/run/{NAME}/csv"))[1].split(",")
    row = dict(zip(EXPECTED_COLUMNS, fields, strict=True))
    assert {row[c] for c in OPTIONAL_COLUMNS} == {""}


def test_csv_header_written_once_across_appends(monkeypatch, tmp_path):
    """Two appends produce one header line plus two data lines."""
    rows = (FULL_ROW, {"step": 8, "split": "val", "loss": 1.25})
    client = _client(monkeypatch, tmp_path, rows=rows)
    assert len(_lines(client.get(f"/run/{NAME}/csv"))) == len(rows) + 1


def test_csv_unknown_model_returns_404(monkeypatch, tmp_path):
    """A model dir that does not exist 404s."""
    client = _client(monkeypatch, tmp_path)
    assert client.get("/run/nosuchmodel/csv").status_code == 404


def test_csv_missing_file_returns_404(monkeypatch, tmp_path):
    """A model with no train.csv 404s rather than serving an empty body."""
    client = _client(monkeypatch, tmp_path, rows=())
    assert client.get(f"/run/{NAME}/csv").status_code == 404
