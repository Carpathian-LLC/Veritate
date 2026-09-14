# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - /atlas/* validate the model name, parse the query into typed arguments for
#   training.atlas, read the brain off app.config for the circuit graph, and turn an
#   atlas failure into a JSON 500. training.atlas is stubbed: no hook artifacts needed.
# tests/mri/test_atlas_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
from flask import Flask
from routes import atlas_routes

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def rig(monkeypatch):
    calls = {}
    am = atlas_routes.atlas_mod

    def record(fn):
        def f(*a, **kw):
            calls[fn] = (a, kw)
            return {"fn": fn}
        return f
    for fn in ("concept_to_neuron", "neuron_to_concept", "neuron_lifetime", "circuit_graph", "concepts_inverted"):
        monkeypatch.setattr(am, fn, record(fn))
    app = Flask(__name__)
    app.config["BRAIN"] = "the-brain"
    atlas_routes.register(app)
    return app.test_client(), calls


@pytest.mark.parametrize("path", ["/atlas/concept?model=../x", "/atlas/neuron/1/2?model=/abs",
                                  "/atlas/lifetime/1/2?model=", "/atlas/concepts_inverted?model=a:b"])
def test_an_unsafe_or_missing_model_name_is_a_400(rig, path):
    client, calls = rig
    res = client.get(path)
    assert res.status_code == 400 and "invalid model name" in res.get_json()["error"]
    assert calls == {}


def test_concept_query_is_parsed_into_typed_arguments(rig):
    client, calls = rig
    assert client.get("/atlas/concept?model=m&step=40&substring=dog&top_k=3").get_json() == {"fn": "concept_to_neuron"}
    assert calls["concept_to_neuron"] == (("m", 40, "dog"), {"top_k": 3})


def test_neuron_and_lifetime_take_layer_and_neuron_from_the_path(rig):
    client, calls = rig
    client.get("/atlas/neuron/3/17?model=m&step=5")
    assert calls["neuron_to_concept"] == (("m", 5, 3, 17), {"top_k": atlas_routes.atlas_mod.ATLAS_DEFAULT_TOP_K})
    client.get("/atlas/lifetime/3/17?model=m")
    assert calls["neuron_lifetime"] == (("m", 3, 17), {})


def test_circuit_reads_the_brain_off_the_app_config(rig):
    client, calls = rig
    client.get("/atlas/circuit?layer=2")
    assert calls["circuit_graph"] == (("the-brain", 2), {"top_k": atlas_routes.atlas_mod.ATLAS_CIRCUIT_TOP_K})


def test_a_failing_inverted_view_is_a_json_500(rig, monkeypatch):
    client, _calls = rig

    def boom(name, step):
        raise RuntimeError("no hooks for " + name)
    monkeypatch.setattr(atlas_routes.atlas_mod, "concepts_inverted", boom)
    res = client.get("/atlas/concepts_inverted?model=m&step=1")
    assert res.status_code == 500 and "no hooks for m" in res.get_json()["error"]
