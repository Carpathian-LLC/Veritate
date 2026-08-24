# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - a sleep run writes consolidated weights to checkpoints/step_N.pt. Until finalize
#   re-exports them, the box keeps serving the pre-sleep bin, so no amount of sleeping
#   changes what the user talks to. Pins: only a model that already serves a bin gets
#   one, the exported dtype follows the bin in place, the engine is reloaded after,
#   and a failed export leaves the previous weights serving.
# tests/training/test_sleep_publish.py
# ------------------------------------------------------------------------------------
# Imports:

import json

import pytest
from training import sleep

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture(autouse=True)
def no_hook():
    sleep.set_publish_hook(None)
    yield
    sleep.set_publish_hook(None)


def _model(tmp_path, monkeypatch, bin_bytes=b"VRTE"):
    d = tmp_path / "models" / "toy"
    (d / "checkpoints").mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"step": 10, "training_args": {}}))
    (d / "checkpoints" / "step_12.pt").write_bytes(b"x")
    if bin_bytes is not None:
        (d / "veritate.bin").write_bytes(bin_bytes)
    monkeypatch.setattr(sleep, "MODELS_ROOT", str(tmp_path / "models"))
    return d


def _fake_export(monkeypatch, calls, result=None):
    import training.export as export_mod

    def fake(name, step, dtype=None, out_path=None):
        calls.append((name, step, dtype))
        return result or {"bytes": 41, "dtype": dtype or "fp16", "path": "x"}

    monkeypatch.setattr(export_mod, "export_checkpoint", fake)


def test_publish_exports_the_newest_checkpoint(tmp_path, monkeypatch):
    """The consolidated weights have to reach the artifact the box serves from."""
    _model(tmp_path, monkeypatch)
    monkeypatch.setattr(sleep.binr, "exists", lambda n: True)
    monkeypatch.setattr(sleep.binr, "weight_dtype", lambda n: "fp16")
    calls = []
    _fake_export(monkeypatch, calls)
    assert sleep.publish("toy")["bytes"] == 41
    assert calls == [("toy", 12, "fp16")]


def test_publish_keeps_the_dtype_the_box_already_serves(tmp_path, monkeypatch):
    """Re-exporting an int8 box at the fp16 default doubles its bin and changes
    its decode speed without anyone asking."""
    _model(tmp_path, monkeypatch)
    monkeypatch.setattr(sleep.binr, "exists", lambda n: True)
    monkeypatch.setattr(sleep.binr, "weight_dtype", lambda n: "int8")
    calls = []
    _fake_export(monkeypatch, calls)
    sleep.publish("toy")
    assert calls[0][2] == "int8"


def test_a_pytorch_only_model_does_not_grow_a_bin(tmp_path, monkeypatch):
    """Sleeping must not hand a model an engine artifact it never had."""
    _model(tmp_path, monkeypatch, bin_bytes=None)
    monkeypatch.setattr(sleep.binr, "exists", lambda n: False)
    calls = []
    _fake_export(monkeypatch, calls)
    assert sleep.publish("toy") is None
    assert calls == []


def test_publish_reloads_the_engine_after_the_swap(tmp_path, monkeypatch):
    """The engine reads a bin into memory and closes it, so a live subprocess
    serves the pre-sleep weights until it is respawned."""
    _model(tmp_path, monkeypatch)
    monkeypatch.setattr(sleep.binr, "exists", lambda n: True)
    monkeypatch.setattr(sleep.binr, "weight_dtype", lambda n: "fp16")
    _fake_export(monkeypatch, [])
    reloaded = []
    sleep.set_publish_hook(reloaded.append)
    sleep.publish("toy")
    assert reloaded == ["toy"]


def test_a_failed_reload_still_reports_the_export(tmp_path, monkeypatch):
    """The weights are on disk and serve from the next engine start; losing the
    reload must not look like a failed publish."""
    _model(tmp_path, monkeypatch)
    monkeypatch.setattr(sleep.binr, "exists", lambda n: True)
    monkeypatch.setattr(sleep.binr, "weight_dtype", lambda n: "fp16")
    _fake_export(monkeypatch, [])

    def boom(_name):
        raise RuntimeError("engine gone")

    sleep.set_publish_hook(boom)
    assert sleep.publish("toy") is not None


def test_a_failed_export_leaves_the_previous_weights_serving(tmp_path, monkeypatch):
    """A veritate.bin opened "wb" is truncated before the first tensor lands, so
    an export that dies halfway leaves the box with nothing to serve."""
    from training import export as export_mod

    served = tmp_path / "veritate.bin"
    served.write_bytes(b"VRTE-old-weights")
    with pytest.raises(ValueError), export_mod._atomic_bin(str(served)) as f:
        f.write(b"VRTE-half")
        raise ValueError("out of memory")
    assert served.read_bytes() == b"VRTE-old-weights"
    assert not (tmp_path / "veritate.bin.part").exists()


def test_a_completed_export_replaces_the_bin_in_one_step(tmp_path):
    """os.replace is atomic on POSIX and Windows: readers see the old bin or the
    whole new one, never a partial file."""
    from training import export as export_mod

    served = tmp_path / "veritate.bin"
    served.write_bytes(b"VRTE-old")
    with export_mod._atomic_bin(str(served)) as f:
        f.write(b"VRTE-new-weights")
    assert served.read_bytes() == b"VRTE-new-weights"
    assert not (tmp_path / "veritate.bin.part").exists()


def test_every_setting_the_controller_reads_exists_in_defaults():
    """A cfg key the controller reads with [] and DEFAULTS does not carry raises
    KeyError on a real box while every test that builds its own cfg dict passes.
    That shipped once (sleep_val_tolerance, 2026-08-24)."""
    import re

    from runtime import settings as settings_mod

    with open(sleep.__file__, encoding="utf-8") as f:
        src = f.read()
    read = set(re.findall(r"""cfg\[["'](sleep_\w+)["']\]""", src))
    read |= set(re.findall(r"""cfg\.get\(["'](sleep_\w+)["']\)""", src))
    assert read, "found no settings reads to check"
    assert read <= set(settings_mod.DEFAULTS), sorted(read - set(settings_mod.DEFAULTS))
