# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Contract tests for the /predict backend: baseline math, metric scoring, and the
#   eval endpoint. Synthetic instruments are injected so the test never depends on
#   the locally-built corpus bins.
# tests/mri/test_predict_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
MRI_DIR = os.path.join(REPO_ROOT, "veritate_mri")
for p in (REPO_ROOT, MRI_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from routes import predict_routes as pr

# ------------------------------------------------------------------------------------
# Fixtures

@pytest.fixture
def client():
    from veritate_mri.app import app
    app.config["TESTING"] = True
    return app.test_client()

# ------------------------------------------------------------------------------------
# Functions

def test_persistence_predicts_previous_bucket():
    """persistence baseline forecasts each bar's return as the prior bar's."""
    actual = [16, 20, 12, 18]
    assert pr._predict_baseline(actual, "persistence") == [16, 16, 20, 12]


def test_flat_predicts_center():
    """flat baseline always forecasts the center (no-move) bucket."""
    assert pr._predict_baseline([20, 12, 18], "flat") == [16, 16, 16]


def test_metrics_directional_accuracy_known():
    """directional accuracy counts sign hits over non-flat actual bars only."""
    actual = [16, 20, 12, 16, 22]
    pred = [16, 18, 18, 16, 25]
    m = pr._metrics(actual, pred)
    assert m["directional_accuracy"] == pytest.approx(2 / 3)
    assert m["n"] == 5


def test_eval_endpoint_scores_injected_series(client, monkeypatch):
    """POST /predict/eval returns aligned pred + metrics for a known series."""
    series = [16, 19, 13, 17, 21, 15]
    monkeypatch.setattr(pr, "_load_instruments", lambda src: [("x", series)])
    r = client.post("/predict/eval", json={"source": "stocks", "predictor": "persistence", "n_bars": 6})
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] and len(d["pred"]) == len(d["actual"]) == 6
    assert "directional_accuracy" in d["metrics"]


def test_eval_rejects_unknown_predictor(client, monkeypatch):
    """A predictor that is not a baseline is rejected until a model exists."""
    monkeypatch.setattr(pr, "_load_instruments", lambda src: [("x", [16, 18, 14])])
    r = client.post("/predict/eval", json={"source": "stocks", "predictor": "some_model"})
    assert r.status_code == 400
