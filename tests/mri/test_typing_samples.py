# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - store + route tests for recorded typing sessions. SAMPLES_DIR is redirected into
#   tmp_path so the machine-local data/typing_samples/ is never read or written.
# tests/mri/test_typing_samples.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
from flask import Flask
from routes import settings_routes
from runtime import typing_samples

# ------------------------------------------------------------------------------------
# Constants

NAME = "20260727_101500"
KEYS = [
    {"t": 0,   "gap": 0,   "ch": "w", "ctx": "word",     "word": "",     "done": False},
    {"t": 140, "gap": 140, "ch": " ", "ctx": "boundary", "word": "what", "done": False},
    {"t": 900, "gap": 760, "ch": "?", "ctx": "sentence", "word": "",     "done": True},
]

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture
def store(monkeypatch, tmp_path):
    """typing_samples with its store redirected into tmp_path."""
    monkeypatch.setattr(typing_samples, "SAMPLES_DIR", str(tmp_path / "typing_samples"))
    return typing_samples


@pytest.fixture
def client(store):
    """Test client for the /typing/samples routes."""
    app = Flask(__name__)
    settings_routes.register(app)
    return app.test_client()


def test_save_then_load_round_trips_every_keystroke(store):
    """A stored session returns the same per-keystroke records it was given."""
    store.save({"name": NAME, "keys": KEYS})
    assert store.load(NAME)["keys"] == KEYS


def test_a_session_without_keystrokes_is_rejected(store):
    """An empty session carries no evidence, so it is not stored."""
    with pytest.raises(ValueError):
        store.save({"name": NAME, "keys": []})


def test_a_name_with_a_path_separator_is_rejected(store):
    """A session name is alphanumeric with - or _, so it can never escape the store."""
    with pytest.raises(ValueError):
        store.save({"name": "../escape", "keys": KEYS})


def test_listing_counts_the_labelled_keystrokes(store):
    """The listing reports how many keystrokes carry the done label."""
    store.save({"name": NAME, "keys": KEYS})
    assert store.listing()[0]["questions"] == 1


def test_listing_is_empty_before_anything_is_recorded(store):
    """A store with no sessions lists nothing rather than raising."""
    assert store.listing() == []


def test_post_stores_the_session(client, store):
    """POST /typing/samples writes the session and returns its name."""
    r = client.post("/typing/samples", json={"name": NAME, "keys": KEYS})
    assert r.get_json()["name"] == NAME


def test_get_lists_the_stored_sessions(client, store):
    """GET /typing/samples lists what has been recorded."""
    client.post("/typing/samples", json={"name": NAME, "keys": KEYS})
    assert [s["name"] for s in client.get("/typing/samples").get_json()["samples"]] == [NAME]


def test_post_without_keystrokes_returns_400(client):
    """A session with no keystrokes is a client error, not a stored empty file."""
    assert client.post("/typing/samples", json={"name": NAME, "keys": []}).status_code == 400


def test_get_one_sample_returns_its_raw_keystrokes(client, store):
    """GET /typing/samples/<name> returns the raw session, unsummarized."""
    client.post("/typing/samples", json={"name": NAME, "keys": KEYS})
    assert client.get(f"/typing/samples/{NAME}").get_json()["keys"] == KEYS


def test_get_a_missing_sample_returns_404(client):
    """An unrecorded name is a 404 rather than a 500."""
    assert client.get("/typing/samples/nope").status_code == 404
