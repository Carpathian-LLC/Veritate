# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the Core Plugins catalog the Training tab renders: a flow filter, the same-group
#   exclusion the form must refuse, and the flat trainer args a selection resolves to
#   (later plugins win a repeated key, unknown ids are ignored).
# tests/core/test_core_plugins.py
# ------------------------------------------------------------------------------------
# Imports:

from veritate_core import core_plugins as cp

# ------------------------------------------------------------------------------------
# Functions


def test_every_entry_has_the_documented_shape_and_ids_are_unique():
    ids = [p["id"] for p in cp.REGISTRY]
    assert len(ids) == len(set(ids))
    for p in cp.REGISTRY:
        assert {"id", "label", "description", "group", "default", "args", "applies_to"} <= set(p)
        assert isinstance(p["args"], dict) and isinstance(p["applies_to"], list)


def test_exactly_one_default_per_group():
    """A fresh form ticks one plugin in each mutually exclusive group, never two."""
    groups = {p["group"] for p in cp.REGISTRY}
    for g in groups:
        defaults = [p["id"] for p in cp.REGISTRY if p["group"] == g and p["default"]]
        assert len(defaults) <= 1, (g, defaults)


def test_the_flow_filter_keeps_unscoped_plugins_and_the_flow_s_own(monkeypatch):
    monkeypatch.setattr(cp, "REGISTRY", [
        {"id": "a", "group": "g1", "args": {}, "applies_to": []},
        {"id": "b", "group": "g2", "args": {}, "applies_to": ["rag"]},
        {"id": "c", "group": "g3", "args": {}, "applies_to": ["scratch"]},
    ])
    assert [p["id"] for p in cp.all_plugins()] == ["a", "b", "c"]
    assert [p["id"] for p in cp.all_plugins("rag")] == ["a", "b"]


def test_two_plugins_of_one_group_conflict_and_unknown_ids_are_ignored():
    same = [p["id"] for p in cp.REGISTRY if p["group"] == cp.GROUP_ACTIVATION][:2]
    assert cp.conflicts(same) == [(same[0], same[1])]
    assert cp.conflicts([same[0], "nope"]) == []


def test_selection_resolves_to_flat_args_with_later_plugins_winning(monkeypatch):
    monkeypatch.setattr(cp, "REGISTRY", [
        {"id": "a", "group": "g1", "args": {"activation": "gelu", "x": 1}, "applies_to": []},
        {"id": "b", "group": "g2", "args": {"activation": "relu"}, "applies_to": []},
    ])
    assert cp.args_for_selection(["a", "b", "ghost"]) == {"activation": "relu", "x": 1}
    assert cp.args_for_selection([]) == {}
