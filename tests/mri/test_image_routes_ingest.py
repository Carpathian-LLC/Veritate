# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The GUI's way of getting pictures in: POST /images/ingest collects a folder on this
#   machine into a set in the background, /images/ingest/status reports it, and
#   /images/sets + /train/discovery list what can be trained on. Pins the validation
#   (a bad set name or a missing folder is a 400, never a thread), the one-at-a-time
#   rule, and that a real ingest lands pictures the discovery route then sees.
# tests/mri/test_image_routes_ingest.py
# ------------------------------------------------------------------------------------
# Imports:

import time

import pytest
from flask import Flask
from PIL import Image
from readers import paths
from routes import image_routes, train_routes

# ------------------------------------------------------------------------------------
# Constants

WAIT_S = 15.0

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def client(tmp_path, monkeypatch):
    for root in ("IMAGES_ROOT", "CODEC_ROOT", "MODELS_ROOT", "CORPUS_ROOT"):
        monkeypatch.setattr(paths, root, str(tmp_path / root.lower()))
    with image_routes._INGEST_LOCK:
        image_routes._INGEST.update(status=image_routes.INGEST_IDLE, set=None, sources=None,
                                    report=None, error=None, started_at=None, finished_at=None)
    app = Flask(__name__)
    image_routes.register(app)
    train_routes.register(app)
    src = tmp_path / "photos" / "Trip"
    src.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (600, 600), (i, i, i)).save(str(src / f"p{i}.png"))
    Image.new("RGB", (32, 32), (9, 9, 9)).save(str(src / "thumb.png"))
    return app.test_client(), str(tmp_path / "photos")


def _wait(client):
    deadline = time.time() + WAIT_S
    while time.time() < deadline:
        body = client.get("/images/ingest/status").get_json()
        if body["status"] != image_routes.INGEST_RUNNING:
            return body
        time.sleep(0.05)
    raise AssertionError("ingest did not finish")


def test_a_bad_set_name_is_a_400_and_starts_nothing(client):
    c, src = client
    res = c.post("/images/ingest", json={"set": "../escape", "sources": [src]})
    assert res.status_code == 400
    assert c.get("/images/ingest/status").get_json()["status"] == image_routes.INGEST_IDLE


def test_a_missing_folder_is_a_400_naming_it(client):
    c, _src = client
    res = c.post("/images/ingest", json={"set": "mine", "sources": ["/nowhere/at/all"]})
    assert res.status_code == 400
    assert "/nowhere/at/all" in res.get_json()["error"]


def test_an_ingest_lands_pictures_that_discovery_then_lists(client):
    c, src = client
    res = c.post("/images/ingest", json={"set": "mine", "sources": [src], "caption_from_folder": True})
    assert res.status_code == 200 and res.get_json()["ok"] is True
    done = _wait(c)
    assert done["status"] == image_routes.INGEST_OK, done
    assert done["report"]["added"] == 3
    assert done["report"]["too_small"] == 1
    assert done["report"]["captions"] == 3
    sets = c.get("/images/sets").get_json()["sets"]
    assert sets == [{"name": "mine", "images": 3, "captions": 3}]
    disc = c.get("/train/discovery").get_json()
    assert disc["image_sets"] == sets
    assert disc["codecs"] == []


def test_a_second_ingest_while_one_runs_is_a_409(client):
    c, src = client
    with image_routes._INGEST_LOCK:
        image_routes._INGEST.update(status=image_routes.INGEST_RUNNING, set="busy")
    res = c.post("/images/ingest", json={"set": "mine", "sources": [src]})
    assert res.status_code == 409
    assert "busy" in res.get_json()["error"]


def test_pick_folder_returns_the_chosen_path(client, monkeypatch):
    """macOS: osascript prints the POSIX path; the route hands it back."""
    import subprocess
    import types
    monkeypatch.setattr(image_routes.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=0, stdout="/Users/me/Pictures/Trip/\n", stderr=""))
    c, _src = client
    body = c.post("/images/pick_folder").get_json()
    assert body == {"ok": True, "path": "/Users/me/Pictures/Trip"}


def test_pick_folder_reports_cancel_as_cancel_not_error(client, monkeypatch):
    import subprocess
    import types
    monkeypatch.setattr(image_routes.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=1, stdout="", stderr="execution error: User canceled. (-128)"))
    c, _src = client
    body = c.post("/images/pick_folder").get_json()
    assert body["ok"] is False and body.get("cancelled") is True


def test_pick_folder_says_when_no_dialog_exists(client, monkeypatch):
    """A headless box gets a clear signal so the form can offer the typed path."""
    monkeypatch.setattr(image_routes.platform, "system", lambda: "Linux")
    monkeypatch.setattr(image_routes.shutil, "which", lambda _cmd: None)
    c, _src = client
    body = c.post("/images/pick_folder").get_json()
    assert body["ok"] is False and body.get("unavailable") is True


class _FakeTeacher:
    model = "fake-vision"

    def __init__(self):
        self.provider = {"system_message_style": "inline"}

    def complete(self, messages, **kw):
        return "The image shows a grey square."


def _fake_teacher(*_a, **_k):
    return _FakeTeacher()


def _ingest_three(c, src):
    """Folder names are captions by default; these tests want none, so the teacher's
    captions are the only ones."""
    c.post("/images/ingest", json={"set": "mine", "sources": [src], "caption_from_folder": False})
    _wait(c)


def test_caption_options_carry_styles_providers_and_the_configured_teacher(client, monkeypatch):
    from runtime import settings as settings_mod
    monkeypatch.setattr(settings_mod, "get", lambda: {"teacher_provider": "ollama", "teacher_model": "llava"})
    c, _src = client
    body = c.get("/images/caption/options").get_json()
    assert {s["id"] for s in body["styles"]} == {"sentence", "tags", "detailed", "custom"}
    assert any(p["id"] == "ollama" for p in body["providers"])
    assert body["current"] == {"provider": "ollama", "model": "llava"}
    assert body["state"]["status"] == "idle"


def test_caption_preview_describes_one_picture_without_writing(client, monkeypatch):
    import veritate_core.plugin as plugin
    monkeypatch.setattr(plugin, "get_teacher_client", _fake_teacher)
    c, src = client
    _ingest_three(c, src)
    res = c.post("/images/caption/preview", json={"set": "mine", "style": "sentence"})
    assert res.status_code == 200, res.get_json()
    body = res.get_json()
    assert body["caption"] == "A grey square."
    assert body["thumbnail"].startswith("data:image/jpeg;base64,")
    assert c.get("/images/sets").get_json()["sets"][0]["captions"] == 0


def test_caption_all_writes_sidecars_and_reports_progress(client, monkeypatch):
    import veritate_core.plugin as plugin
    monkeypatch.setattr(plugin, "get_teacher_client", _fake_teacher)
    with image_routes._CAPTION_LOCK:
        image_routes._CAPTION.update(status=image_routes.CAPTION_IDLE, stop=False)
    c, src = client
    _ingest_three(c, src)
    res = c.post("/images/caption", json={"set": "mine", "style": "tags", "concurrency": 2})
    assert res.status_code == 200, res.get_json()
    deadline = time.time() + WAIT_S
    while time.time() < deadline:
        state = c.get("/images/caption/status").get_json()
        if state["status"] != image_routes.CAPTION_RUNNING:
            break
        time.sleep(0.05)
    assert state["status"] == image_routes.CAPTION_OK, state
    assert state["done"] == 3 and state["failed"] == 0
    assert state["samples"][-1]["caption"] == "A grey square."
    assert c.get("/images/sets").get_json()["sets"][0]["captions"] == 3


def test_caption_without_a_set_is_a_400(client):
    c, _src = client
    res = c.post("/images/caption", json={"set": "nope"})
    assert res.status_code == 400

