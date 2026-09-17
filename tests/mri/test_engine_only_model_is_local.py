# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - a model deployed to a serving box holds the exported veritate.bin and no PyTorch
#   checkpoint: 610 MB of int8 against the 4.8 GB .pt it came from. is_local_model()
#   required a checkpoint, so /v1/chat/completions answered "model not found" for a
#   model the C engine could load and serve (cardinal-01, 2026-09-14).
# - _ensure_c() already needs only the bin; this pins the gate in front of it.
# tests/mri/test_engine_only_model_is_local.py
# ------------------------------------------------------------------------------------
# Imports:

from readers import bin as binr
from readers import checkpoints, models
from routes import hybrid_routes

# ------------------------------------------------------------------------------------
# Functions

def _stub(monkeypatch, step, has_bin):
    monkeypatch.setattr(models, "exists", lambda n: True)
    monkeypatch.setattr(checkpoints, "latest_step", lambda n: step)
    monkeypatch.setattr(binr, "exists", lambda n: has_bin)


def test_bin_only_model_is_local(monkeypatch):
    """The normal shape of a deployment to a weak box."""
    _stub(monkeypatch, step=None, has_bin=True)
    assert hybrid_routes.is_local_model("wren2")


def test_checkpoint_only_model_is_local(monkeypatch):
    """A training box that has not exported yet still serves on pytorch."""
    _stub(monkeypatch, step=146000, has_bin=False)
    assert hybrid_routes.is_local_model("wren2")


def test_model_with_no_weights_is_not_local(monkeypatch):
    """config.json alone is a directory, not a servable model."""
    _stub(monkeypatch, step=None, has_bin=False)
    assert not hybrid_routes.is_local_model("wren2")


def test_cloud_id_is_never_local(monkeypatch):
    _stub(monkeypatch, step=146000, has_bin=True)
    assert not hybrid_routes.is_local_model(hybrid_routes.CLOUD_ID)
