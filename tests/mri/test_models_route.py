# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - route tests for /v1/models (OpenAI-compatible list). readers are stubbed so no
#   filesystem models or checkpoint loads are needed (rule 33); assertions cover the
#   payload envelope, extra-field passthrough, and newest-first order.
# tests/mri/test_models_route.py
# ------------------------------------------------------------------------------------
# Imports:



from flask import Flask
from routes import models_routes

# ------------------------------------------------------------------------------------
# Constants

ROWS = [
    {"name": "old_model", "step": 5, "is_current": False, "plugin": "",
     "n_params": 800000000, "hidden": 1536, "layers": 24, "description": "",
     "mtime": 100, "capabilities": {"chat": True}, "engine": "c_engine"},
    {"name": "new_model", "step": 9, "is_current": True, "plugin": "",
     "n_params": None, "hidden": 512, "layers": 8, "description": "",
     "mtime": 200, "capabilities": {}, "engine": "pytorch"},
]

# A row missing every optional field except name/mtime: the endpoint must
# degrade each extra to null rather than 500 (regression guard for the hard
# r["engine"] subscript that KeyError'd on rows predating the `engine` field).
SPARSE_ROW = {"name": "sparse_model", "mtime": 300}

# ------------------------------------------------------------------------------------
# Functions

def _client(monkeypatch, rows):
    """Minimal app with models_routes registered and _model_rows stubbed."""
    monkeypatch.setattr(models_routes, "_model_rows", lambda: sorted(rows, key=lambda r: -r["mtime"]))
    app = Flask(__name__)
    models_routes.register(app)
    return app.test_client()


def test_v1_models_envelope(monkeypatch):
    """GET /v1/models returns the OpenAI list envelope with object:'list'."""
    body = _client(monkeypatch, ROWS).get("/v1/models").get_json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)


def test_v1_models_entry_shape(monkeypatch):
    """Each entry has id/object/created/owned_by and the preserved extras."""
    entry = _client(monkeypatch, ROWS).get("/v1/models").get_json()["data"][0]
    assert entry["object"] == "model"
    assert entry["owned_by"] == models_routes.OWNER
    for k in ("id", "created", "n_params", "hidden", "layers", "capabilities", "is_current", "engine"):
        assert k in entry


def test_v1_models_tolerates_row_missing_optional_fields(monkeypatch):
    """A row with only name/mtime must yield a 200 with the extras nulled out,
    never a 500 from a hard subscript on a missing key."""
    resp = _client(monkeypatch, [SPARSE_ROW]).get("/v1/models")
    assert resp.status_code == 200
    entry = resp.get_json()["data"][0]
    assert entry["id"] == "sparse_model"
    assert entry["created"] == 300
    for k in ("n_params", "hidden", "layers", "capabilities", "is_current", "engine"):
        assert entry[k] is None


def test_v1_models_id_and_created_from_row(monkeypatch):
    """id maps from name and created is the int mtime of the model."""
    top = _client(monkeypatch, ROWS).get("/v1/models").get_json()["data"][0]
    assert top["id"] == "new_model"
    assert top["created"] == 200


def test_v1_models_sorted_newest_first(monkeypatch):
    """Entries are ordered newest-first by created, matching /pytorch-models."""
    data = _client(monkeypatch, ROWS).get("/v1/models").get_json()["data"]
    assert [e["id"] for e in data] == ["new_model", "old_model"]


# ------------------------------------------------------------------------------------
# Discovery: an engine-only model must be listed.
#
# _model_rows() skipped any model without a PyTorch checkpoint, so a serving box that
# holds only the exported veritate.bin advertised nothing. That is the normal shape of
# a deployment to a weak box: 610 MB of int8 bin instead of a 4.8 GB .pt. Deleting the
# superseded checkpoint on cardinal-01 (2026-09-14) made the deployed model invisible
# to the picker while the C engine had it loaded and serving.

def _rows_with(monkeypatch, steps, bins):
    """_model_rows() over stubbed readers. steps/bins map name -> step or bin presence."""
    from readers import bin as binr
    from readers import capabilities as caps_reader
    from readers import checkpoints as ck
    from readers import config as cfg_reader
    from readers import models as models_reader
    from routes import hybrid_routes

    monkeypatch.setattr(models_reader, "list_models", lambda: sorted(steps))
    monkeypatch.setattr(ck, "latest_step", lambda n: steps[n])
    monkeypatch.setattr(ck, "path_for", lambda n, s: f"/nonexistent/{n}/{s}.pt")
    monkeypatch.setattr(binr, "exists", lambda n: bins[n])
    monkeypatch.setattr(cfg_reader, "load", lambda n: {"n_params_total": 1, "shape": {"hidden": 8, "layers": 2}})
    monkeypatch.setattr(cfg_reader, "description", lambda n: "")
    monkeypatch.setattr(caps_reader, "read", lambda n: {})
    monkeypatch.setattr(hybrid_routes, "_default_local_backend", lambda n: "c" if bins[n] else "pytorch")

    app = Flask(__name__)
    with app.test_request_context():
        return {r["name"]: r for r in models_routes._model_rows()}


def test_engine_only_model_is_listed(monkeypatch):
    """A model with veritate.bin and no .pt is a served model, not an absent one."""
    rows = _rows_with(monkeypatch,
                      steps={"engine_only": None, "pytorch_only": 5, "no_weights": None},
                      bins={"engine_only": True, "pytorch_only": False, "no_weights": False})
    assert "engine_only" in rows, "engine-only model dropped from discovery"
    assert rows["engine_only"]["engine"] == "c"
    assert "pytorch_only" in rows


def test_model_with_neither_weights_is_skipped(monkeypatch):
    """config.json alone is a directory, not a model."""
    rows = _rows_with(monkeypatch,
                      steps={"no_weights": None, "pytorch_only": 5},
                      bins={"no_weights": False, "pytorch_only": False})
    assert "no_weights" not in rows
