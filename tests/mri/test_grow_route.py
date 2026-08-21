# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - POST /models/grow + /models/grow/status + /models/grow/options. Pins: the
#   happy path writes the exact layout the continue flow resumes (step_0.pt +
#   config.json with updated shape/name/size/description), validation failures
#   are clean 400 JSON (never 500), name collisions and unsupported variants are
#   refused before any work, and options enumerates only reachable sizes.
# tests/mri/test_grow_route.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import time

import pytest
import torch
from flask import Flask
from readers import paths
from routes import grow_routes

from veritate_core.model_patched import VeritatePatched

# ------------------------------------------------------------------------------------
# Constants

SRC   = "toy_src"
SEQ   = 128
SHAPE = {"layers": 2, "hidden": 64, "ffn": 128, "heads": 4}
# "5m" in trainer_sizes.json (6L/256h/1024f/4heads) strictly dominates SHAPE and
# keeps the grown toy small enough for a seconds-fast test.
TARGET_SIZE  = "5m"
JOB_WAIT_S   = 60

# ------------------------------------------------------------------------------------
# Functions


def _write_source(trunk="hybrid", state_rule="gla"):
    torch.manual_seed(3)
    model = VeritatePatched(vocab=256, hidden=SHAPE["hidden"], layers=SHAPE["layers"],
                            ffn=SHAPE["ffn"], heads=SHAPE["heads"], seq=SEQ,
                            global_mixer="recurrent")
    os.makedirs(paths.checkpoints_dir(SRC))
    torch.save({"model": model.state_dict(), "step": 700},
               paths.checkpoint_path(SRC, 700))
    cfg = {
        "name": SRC,
        "description": "toy",
        "shape": dict(SHAPE, seq=SEQ, vocab=256),
        "n_params_total": 234344,
        "training_args": {"trunk": trunk, "state_rule": state_rule, "size": "5m",
                          "corpus_bin": "/pinned/train.bin", "val_bin": "/pinned/val.bin"},
    }
    with open(paths.config_path(SRC), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return model


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path / "models"))
    grow_routes._STATE.update(running=False, source=None, name=None, phase="idle",
                              error=None, result=None)
    app = Flask(__name__)
    grow_routes.register(app)
    return app.test_client()


def _wait_done(client):
    deadline = time.time() + JOB_WAIT_S
    while time.time() < deadline:
        st = client.get("/models/grow/status").get_json()
        if not st["running"]:
            return st
        time.sleep(0.05)
    raise AssertionError("grow job did not finish in time")


def test_happy_path_writes_resumable_layout(client):
    """Growing to a trainer_sizes key writes step_0.pt + a config.json the
    continue flow resumes, and the grown weights load into the target shape."""
    _write_source()
    r = client.post("/models/grow", json={"source": SRC, "step": 700,
                                          "target_size": TARGET_SIZE, "name": "toy_big"})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["target"] == {"layers": 6, "hidden": 256, "ffn": 1024,
                                      "heads": 4, "seq": SEQ}
    st = _wait_done(client)
    assert st["phase"] == "done" and st["error"] is None
    ckpt_path = paths.checkpoint_path("toy_big", 0)
    assert os.path.isfile(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["step"] == 0 and "optimizer" not in ckpt
    grown = VeritatePatched(vocab=256, hidden=256, layers=6, ffn=1024, heads=4,
                            seq=SEQ, global_mixer="recurrent")
    grown.load_state_dict(ckpt["model"], strict=True)


def test_grown_config_carries_shape_name_description(client):
    """config.json: name/shape/size/description/training_args all updated; the
    source's pinned corpus paths are cleared so the form's corpus choice wins."""
    _write_source()
    client.post("/models/grow", json={"source": SRC, "target_size": TARGET_SIZE,
                                      "name": "toy_big"})
    _wait_done(client)
    with open(paths.config_path("toy_big"), encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["name"] == "toy_big"
    assert cfg["step"] == 0
    assert {f: cfg["shape"][f] for f in SHAPE} == {"layers": 6, "hidden": 256,
                                                   "ffn": 1024, "heads": 4}
    assert cfg["shape"]["seq"] == SEQ
    assert cfg["grown_from"] == {"source": SRC, "step": 700}
    assert "Grown from toy_src@700 (function-preserving)" in cfg["description"]
    assert "warmup_steps > 0" in cfg["description"]
    ta = cfg["training_args"]
    assert ta["size"] == TARGET_SIZE
    assert ta["resume"] is True
    assert ta["warmup_steps"] > 0
    assert ta["total_steps"] > 0
    assert ta["corpus_bin"] == "" and ta["val_bin"] == ""
    assert cfg["n_params_total"] > 234344


def test_name_collision_is_refused(client):
    """A target name that already exists is a 400 before any work."""
    _write_source()
    os.makedirs(paths.model_dir("taken"))
    r = client.post("/models/grow", json={"source": SRC, "target_size": TARGET_SIZE,
                                          "name": "taken"})
    assert r.status_code == 400
    assert "already exists" in r.get_json()["error"]


def test_shrinking_shape_is_refused(client):
    """An explicit target smaller than the source on any axis is a clean 400."""
    _write_source()
    r = client.post("/models/grow", json={
        "source": SRC, "name": "toy_small",
        "target_size": {"layers": 2, "hidden": 32, "ffn": 128, "heads": 4}})
    assert r.status_code == 400
    assert "growth only" in r.get_json()["error"]
    assert not os.path.isdir(paths.model_dir("toy_small"))


def test_head_only_growth_is_refused(client):
    """heads up at fixed hidden would shrink head_dim: refused with the reason."""
    _write_source()
    r = client.post("/models/grow", json={
        "source": SRC, "name": "toy_heads",
        "target_size": {"layers": 2, "hidden": 64, "ffn": 128, "heads": 8}})
    assert r.status_code == 400
    assert "head_dim" in r.get_json()["error"]


def test_unsupported_variant_is_refused(client):
    """A source trained with an unsupported trunk or state rule is a clean 400."""
    _write_source(trunk="hybrid_moe")
    r = client.post("/models/grow", json={"source": SRC, "target_size": TARGET_SIZE,
                                          "name": "toy_moe"})
    assert r.status_code == 400
    assert "trunk" in r.get_json()["error"]


def test_options_lists_only_reachable_targets(client):
    """options enumerates sizes strictly larger than and reachable from the
    source, with param counts; the source's own size is absent."""
    _write_source()
    r = client.get("/models/grow/options", query_string={"source": SRC})
    body = r.get_json()
    assert r.status_code == 200 and body["ok"]
    assert body["step"] == 700 and body["steps"] == [700]
    keys = [t["size"] for t in body["targets"]]
    assert TARGET_SIZE in keys
    for t in body["targets"]:
        assert all(t[f] >= SHAPE[f] for f in SHAPE)
        assert any(t[f] > SHAPE[f] for f in SHAPE)
        assert t["hidden"] % t["heads"] == 0
        assert t["hidden"] // t["heads"] >= SHAPE["hidden"] // SHAPE["heads"]
        assert t["params"] > body["params"]


def test_seq_only_growth_writes_extended_tables(client):
    """target_seq without target_size grows context alone: position tables
    extend, config carries the new seq in shape and training_args."""
    _write_source()
    r = client.post("/models/grow", json={"source": SRC, "target_seq": 256,
                                          "name": "toy_ctx"})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["target"]["seq"] == 256
    st = _wait_done(client)
    assert st["phase"] == "done" and st["error"] is None
    ckpt = torch.load(paths.checkpoint_path("toy_ctx", 0), map_location="cpu",
                      weights_only=False)
    assert ckpt["model"]["pos_emb.weight"].shape[0] == 256
    assert ckpt["model"]["slot_pos_emb.weight"].shape[0] == 64
    with open(paths.config_path("toy_ctx"), encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["shape"]["seq"] == 256
    assert cfg["training_args"]["seq"] == 256


def test_invalid_seq_is_refused(client):
    """A smaller seq and a non-stride-multiple seq are clean 400s."""
    _write_source()
    r = client.post("/models/grow", json={"source": SRC, "target_seq": 64,
                                          "name": "toy_ctx"})
    assert r.status_code == 400
    assert "growth only" in r.get_json()["error"]
    r = client.post("/models/grow", json={"source": SRC, "target_seq": 130,
                                          "name": "toy_ctx"})
    assert r.status_code == 400
    assert "stride" in r.get_json()["error"]
    assert not os.path.isdir(paths.model_dir("toy_ctx"))


def test_options_offer_seq_choices(client):
    """options exposes 1x/2x/4x context choices with per-choice param counts."""
    _write_source()
    r = client.get("/models/grow/options", query_string={"source": SRC})
    body = r.get_json()
    assert body["seq"] == SEQ
    assert body["seq_choices"] == [SEQ, 2 * SEQ, 4 * SEQ]
    assert body["source_params_seq"][str(2 * SEQ)] > body["params"]
    for t in body["targets"]:
        assert t["params_seq"][str(SEQ)] == t["params"]
        assert t["params_seq"][str(4 * SEQ)] > t["params"]


def test_missing_source_is_a_400_not_a_500(client):
    r = client.post("/models/grow", json={"source": "nope", "target_size": TARGET_SIZE,
                                          "name": "toy_big"})
    assert r.status_code == 400
    assert "not found" in r.get_json()["error"]
