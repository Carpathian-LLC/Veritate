# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the refusals of /pruning/report and /pruning/generate_plugin (unknown model, no
#   checkpoint, a checkpoint without dense FFNs) and the /export/<name> contract (404
#   unknown, 400 no checkpoint or bad request, ok with the exporter's result). Readers
#   and the exporter are stubbed; the one checkpoint written is a real tiny torch file.
# tests/mri/test_pruning_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
import torch
from flask import Flask
from routes import pruning_routes

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def rig(monkeypatch, tmp_path):
    ckpt = tmp_path / "step_5.pt"
    torch.save({"model": {"blocks.0.router.weight": torch.zeros(1)}, "args": {"plugin": "moe_trainer"}}, ckpt)
    monkeypatch.setattr(pruning_routes.models, "exists", lambda n: n in ("moe", "empty"))
    monkeypatch.setattr(pruning_routes.checkpoints, "latest_step", lambda n: 5 if n == "moe" else None)
    monkeypatch.setattr(pruning_routes.checkpoints, "path_for", lambda n, s: str(ckpt))
    app = Flask(__name__)
    pruning_routes.register(app)
    return app.test_client()


def test_report_refuses_an_unknown_model_and_a_model_without_checkpoints(rig):
    assert rig.get("/pruning/report?model=nope").status_code == 400
    res = rig.get("/pruning/report?model=empty")
    assert res.status_code == 400 and "no checkpoints" in res.get_json()["error"]


def test_report_refuses_a_checkpoint_without_dense_ffns_and_names_the_plugin(rig):
    res = rig.get("/pruning/report?model=moe")
    assert res.status_code == 400
    assert "not enabled" in res.get_json()["error"] and "moe_trainer" in res.get_json()["error"]


def test_generate_plugin_needs_a_model_and_a_step(rig):
    assert rig.post("/pruning/generate_plugin", json={"model": "moe"}).status_code == 400
    assert rig.post("/pruning/generate_plugin", json={"model": "nope", "step": 5}).status_code == 400
    res = rig.post("/pruning/generate_plugin", json={"model": "moe", "step": 5})
    assert res.status_code == 400 and "not enabled" in res.get_json()["error"]


def test_export_contract(rig, monkeypatch):
    from training import export as export_mod
    seen = {}

    def fake_export(name, step, dtype=None):
        seen["call"] = (name, step, dtype)
        if dtype == "bad":
            raise ValueError("unknown dtype")
        return {"path": "/x/model.bin", "bytes": 12}
    monkeypatch.setattr(export_mod, "export_checkpoint", fake_export)
    assert rig.post("/export/nope", json={}).status_code == 404
    assert rig.post("/export/empty", json={}).status_code == 400
    body = rig.post("/export/moe", json={}).get_json()
    assert body == {"ok": True, "path": "/x/model.bin", "bytes": 12} and seen["call"] == ("moe", 5, None)
    res = rig.post("/export/moe", json={"step": 3, "dtype": "bad"})
    assert res.status_code == 400 and seen["call"] == ("moe", 3, "bad")
