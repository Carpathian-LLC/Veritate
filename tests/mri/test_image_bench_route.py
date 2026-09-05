# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - POST /images/decode_bench. Pins: a valid request returns the per-arm report F0
#   reads, and an out-of-range frame is a clean 400 rather than an allocation the
#   size of the mistake.
# tests/mri/test_image_bench_route.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
from flask import Flask
from routes import image_routes

# ------------------------------------------------------------------------------------
# Constants

FRAME = {"height": 120, "width": 160, "reps": 1, "warmup": 0}

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def client():
    app = Flask(__name__)
    image_routes.register(app)
    return app.test_client()


def test_bench_reports_peak_activation_bytes_per_arm(client):
    """The route returns the number F0 decides on, for every arm it ran."""
    body = client.post("/images/decode_bench", json=dict(FRAME, arms=["coord", "patch"])).get_json()
    assert body["ok"] is True
    assert set(body["report"]["arms"]) == {"coord", "patch"}
    for arm in body["report"]["arms"].values():
        assert arm["peak_activation_bytes"] > 0
        assert arm["gflop"] > 0


def test_a_frame_past_the_guard_is_a_400_not_an_allocation(client):
    """An edge beyond MAX_EDGE is refused before any tensor is built."""
    res = client.post("/images/decode_bench", json={"height": 120, "width": image_routes.MAX_EDGE + 1})
    assert res.status_code == 400
    assert res.get_json()["ok"] is False
