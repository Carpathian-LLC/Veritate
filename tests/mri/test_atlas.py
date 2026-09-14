# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the neuron atlas is read-only aggregation over hook artifacts: which neurons vote for
#   a byte set, what a neuron votes for and which concepts name it, its rank over
#   training, the static layer-to-layer transfer graph, and the concept inversion. The
#   hooks reader is stubbed with in-memory artifacts; the circuit graph uses a tiny model.
# tests/mri/test_atlas.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
from training import atlas

from veritate_core.model import VOCAB_BYTE_LEVEL, Veritate

# ------------------------------------------------------------------------------------
# Constants

FRAMES = [
    {"byte": ord("a"), "dla_picked": [{"layer": 0, "neuron": 5, "contrib": 0.5},
                                      {"layer": 1, "neuron": 2, "contrib": -0.2}]},
    {"byte": ord("a"), "dla_picked": [{"layer": 0, "neuron": 5, "contrib": 0.25}]},
    {"byte": ord("z"), "dla_picked": [{"layer": 0, "neuron": 7, "contrib": 9.0}]},
]
CONCEPTS = {"concepts": [
    {"concept": "vowels", "surprise": 1.5, "top_neurons": [{"layer": 0, "id": 5, "v": 0.9}]},
    {"concept": "edges", "surprise": 0.5, "top_neurons": [{"layer": 1, "neuron": 2, "magnitude": 0.4}]},
]}
PROBES = {10: {"layers": [{"layer": 0, "neurons": [{"id": 3, "v": 1.0}, {"id": 5, "v": 0.7}]}]},
          20: {"layers": [{"layer": 0, "neurons": [{"id": 5, "v": 0.95}]}]},
          30: {"layers": [{"layer": 0, "neurons": [{"id": 9, "v": 0.1}]}]}}

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def rig(monkeypatch):
    def load_artifact(name, step, artifact):
        if name != "m":
            return None
        if artifact == "generation":
            return {"frames": FRAMES} if step == 10 else None
        if artifact == "concepts":
            return CONCEPTS if step == 10 else None
        if artifact == "probe":
            return PROBES.get(step)
        return None
    monkeypatch.setattr(atlas.hooks, "load_artifact", load_artifact)
    monkeypatch.setattr(atlas.hooks, "list_steps", lambda name: [10, 20, 30])


def test_concept_to_neuron_sums_absolute_votes_over_frames_of_the_byte_set(rig):
    out = atlas.concept_to_neuron("m", 10, "a")
    assert out["n_frames"] == 3 and out["n_matched"] == 2
    assert out["neurons"] == [{"layer": 0, "neuron": 5, "score": 0.75}, {"layer": 1, "neuron": 2, "score": 0.2}]
    assert atlas.concept_to_neuron("m", 10, "az", top_k=1)["neurons"] == [{"layer": 0, "neuron": 7, "score": 9.0}]
    assert "error" in atlas.concept_to_neuron("m", 99, "a") and "error" in atlas.concept_to_neuron("m", 10, "")


def test_neuron_to_concept_lists_the_bytes_it_carries_and_the_concepts_naming_it(rig):
    out = atlas.neuron_to_concept("m", 10, 0, 5)
    assert out["bytes"] == [{"b": ord("a"), "score": 0.75}]
    assert out["concepts"] == [{"concept": "vowels", "surprise": 1.5, "magnitude": 0.9}]
    assert atlas.neuron_to_concept("m", 10, 1, 2)["concepts"][0]["magnitude"] == 0.4


def test_neuron_lifetime_reads_rank_and_magnitude_at_every_probed_step(rig):
    out = atlas.neuron_lifetime("m", 0, 5)
    assert [(s["step"], s["rank"], s["magnitude"], s["in_top_k"]) for s in out["series"]] == [
        (10, 1, 0.7, True), (20, 0, 0.95, True), (30, -1, 0.0, False)]


def test_concepts_inverted_keys_neurons_by_layer_and_id_in_both_dump_shapes(rig, monkeypatch):
    out = atlas.concepts_inverted("m", 10)
    assert {(n["layer"], n["neuron"]) for n in out["neurons"]} == {(0, 5), (1, 2)}
    assert "error" in atlas.concepts_inverted("m", 99)
    as_dict = {"concepts": {"vowels": {"surprise_bits": 2.0, "top_neurons": [{"layer": 0, "id": 5, "v": 0.9}]}}}
    monkeypatch.setattr(atlas.hooks, "load_artifact", lambda name, step, artifact: as_dict)
    out = atlas.concepts_inverted("m", 10)
    assert out["neurons"] == [{"layer": 0, "neuron": 5,
                               "concepts": [{"concept": "vowels", "magnitude": 0.9, "surprise": 2.0}]}]


def test_circuit_graph_needs_a_brain_and_lists_top_k_targets_per_source_neuron():
    assert "error" in atlas.circuit_graph(None, 0)

    class Brain:
        model = Veritate(vocab=VOCAB_BYTE_LEVEL, hidden=8, layers=3, ffn=6, heads=2, seq=4)
    out = atlas.circuit_graph(Brain(), 1, top_k=2)
    assert (out["src_layer"], out["dst_layer"], out["src_count"], out["dst_count"]) == (1, 2, 6, 6)
    assert len(out["edges"]) == 6 * 2 and all(e["src_layer"] == 1 and e["dst_layer"] == 2 for e in out["edges"])
    assert "error" in atlas.circuit_graph(Brain(), 2)
