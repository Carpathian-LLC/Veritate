# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - POST /images/generate against a tiny image model written straight into a temp
#   models root (config.json stamped image + step_1.pt + its codec). Pins: every mode
#   returns a PNG at the model's frame, a text model is refused, a source mode without
#   a source is a 400, and /images/models lists only image models.
# tests/mri/test_image_generate_route.py
# ------------------------------------------------------------------------------------
# Imports:

import base64
import io
import json
import os

import pytest
import torch
from flask import Flask
from PIL import Image
from readers import paths
from routes import image_routes

from veritate_core.model import Veritate
from veritate_core.plugin import hardware, image_codec

# ------------------------------------------------------------------------------------
# Constants

H = W = 40
PATCH, PLANES, SEQ = 20, 2, 64
NAME = "pix_tiny"

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def client(tmp_path, monkeypatch):
    for root in ("MODELS_ROOT", "CODEC_ROOT", "IMAGES_ROOT", "CORPUS_ROOT"):
        monkeypatch.setattr(paths, root, str(tmp_path / root.lower()))
    monkeypatch.setenv(hardware.DEVICE_ENV, "cpu")
    image_routes._MODEL_CACHE.clear()
    torch.manual_seed(0)
    codec = image_codec.ImageCodec(planes=PLANES, latent_dim=8, patch=PATCH, dec_hidden=32)
    image_codec.save(codec, paths.codec_path(NAME + "_codec"))
    model = Veritate(256, 32, 1, 64, 2, SEQ, causal=False)
    os.makedirs(paths.checkpoints_dir(NAME), exist_ok=True)
    torch.save({"model": model.state_dict()}, paths.checkpoint_path(NAME, 1))
    cfg = {"name": NAME, "training": "image", "kind": "trainer",
           "shape": {"vocab": 256, "hidden": 32, "layers": 1, "ffn": 64, "heads": 2, "seq": SEQ},
           "training_args": {"codec": NAME + "_codec", "height": H, "width": W,
                             "image_code_bytes": PLANES * (H // PATCH) * (W // PATCH)}}
    with open(paths.config_path(NAME), "w", encoding="utf-8") as handle:
        json.dump(cfg, handle)
    # a text model beside it, to be excluded
    os.makedirs(paths.model_dir("prose"), exist_ok=True)
    with open(paths.config_path("prose"), "w", encoding="utf-8") as handle:
        json.dump({"name": "prose", "training": "", "shape": {}, "training_args": {}}, handle)
    os.makedirs(paths.checkpoints_dir("prose"), exist_ok=True)
    torch.save({"model": {}}, paths.checkpoint_path("prose", 1))
    app = Flask(__name__)
    image_routes.register(app)
    return app.test_client()


def _png_b64():
    buf = io.BytesIO()
    Image.new("RGB", (80, 50), (40, 90, 200)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def test_models_lists_only_image_models(client):
    body = client.get("/images/models").get_json()
    assert [m["name"] for m in body["models"]] == [NAME]
    assert body["models"][0]["steps"] == [1]


@pytest.mark.parametrize("mode", ["text", "unconditional", "variation", "inpaint", "expand"])
def test_every_mode_returns_a_png_at_the_frame(client, mode):
    body = {"model": NAME, "mode": mode, "caption": "a blue sky", "passes": 2, "seed": 3,
            "rect": [0.0, 0.0, 0.5, 0.5]}
    if mode in ("variation", "inpaint", "expand"):
        body["image"] = _png_b64()
    res = client.post("/images/generate", json=body)
    assert res.status_code == 200, res.get_json()
    out = res.get_json()
    assert out["ok"] is True and out["mode"] == mode and out["step"] == 1
    image = Image.open(io.BytesIO(base64.b64decode(out["png"])))
    assert image.size == (W, H)


def test_a_text_model_is_refused(client):
    res = client.post("/images/generate", json={"model": "prose", "mode": "text"})
    assert res.status_code == 400
    assert "not an image model" in res.get_json()["error"]


def test_a_source_mode_without_a_source_is_a_400(client):
    res = client.post("/images/generate", json={"model": NAME, "mode": "variation"})
    assert res.status_code == 400


def test_the_loaded_model_stays_resident(client):
    client.post("/images/generate", json={"model": NAME, "mode": "text", "passes": 1})
    assert len(image_routes._MODEL_CACHE) == 1
    client.post("/images/generate", json={"model": NAME, "mode": "text", "passes": 1})
    assert len(image_routes._MODEL_CACHE) == 1


def test_mri_lists_probe_steps_and_serves_their_pictures(client):
    """The Models tab's image view: metrics per checkpoint and the PNGs behind them."""
    from veritate_core.plugin import image_codec as ic
    from veritate_core.plugin import image_probe
    model = Veritate(256, 32, 1, 64, 2, SEQ, causal=False).eval()
    torch.manual_seed(0)
    codec = ic.ImageCodec(planes=PLANES, latent_dim=8, patch=PATCH, dec_hidden=32).eval()
    geometry = {"height": H, "width": W, "seq": SEQ, "code_bytes": PLANES * 4}
    image_probe.dump(model, codec, geometry, NAME, 1, None, "cpu")
    body = client.get("/images/mri/" + NAME).get_json()
    assert body["ok"] is True and body["training"] == "image"
    assert [s["step"] for s in body["steps"]] == [1]
    assert body["geometry"]["height"] == H
    res = client.get(f"/images/mri/{NAME}/1/samples.png")
    assert res.status_code == 200 and res.data[:4] == b"\x89PNG"
    assert client.get(f"/images/mri/{NAME}/1/fill.png").status_code == 404      # no val bin -> not written
    assert client.get(f"/images/mri/{NAME}/1/evil.txt").status_code == 404
    assert client.get("/images/mri/nope").status_code == 404



def test_live_reports_the_stage_before_config_exists_and_404s_for_an_unknown_run(client):
    """A first run spends most of its wall clock before config.json is written; the
    Training tab must still see what it is doing."""
    from veritate_core.plugin import image_progress
    model_dir = paths.model_dir("fresh_20m")
    prog = image_progress.Progress(model_dir, "mps", total_steps=500)
    prog.stage("decode", 3800, 139719, "decoding pictures 3,800 / 139,719")
    r = client.get("/images/live/fresh_20m")
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] and d["training"] == "image"
    assert d["progress"]["device"] == "mps"
    assert d["progress"]["current"] == "decode"
    assert d["progress"]["stages"]["decode"]["done"] == 3800
    assert d["progress"]["stages"]["train"]["total"] == 500
    assert d["checkpoint_steps"] == [] and d["latest_probe"] is None
    assert client.get("/images/live/never_there").status_code == 404


def test_live_lists_checkpoints_and_the_latest_probe_for_a_trained_model(client):
    from veritate_core.plugin import image_codec as _ic
    from veritate_core.plugin import image_probe
    model = Veritate(256, 32, 1, 64, 2, SEQ, causal=False)
    codec = _ic.load(paths.codec_path(NAME + "_codec"))
    geometry = {"height": H, "width": W, "seq": SEQ, "code_bytes": PLANES * (H // PATCH) * (W // PATCH)}
    image_probe.dump(model, codec, geometry, NAME, 1, None, "cpu")
    d = client.get("/images/live/" + NAME).get_json()
    assert d["checkpoint_steps"] == [1]
    assert d["last_checkpoint_at"] is not None
    assert d["latest_probe"]["step"] == 1 and "samples.png" in d["latest_probe"]["files"]
    assert d["running"] is False and d["log_tail"] == []
