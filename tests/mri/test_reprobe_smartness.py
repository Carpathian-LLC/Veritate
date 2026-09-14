# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - reprobe_smartness re-runs the smartness axes over a model's historical checkpoints so
#   old hook dirs carry a newly added probe. Pinned here: the axis table's names match the
#   probe functions it calls, the step and axis filters, the skip for a checkpoint that was
#   never finalized (no hook dir), and the rename to the canonical filename the dashboard
#   reads. Checkpoints and probes are stubbed; no weights are loaded.
# tests/mri/test_reprobe_smartness.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import sys
import types

import pytest
from tools import reprobe_smartness as rp

# ------------------------------------------------------------------------------------
# Functions


def _install_fake_probe(monkeypatch, calls):
    """The tool does `from training import checkpoint_probe` inside main(), which resolves
    off the already-imported package, so the attribute has to move too."""
    import training
    fake = _fake_probe_module(calls)
    monkeypatch.setitem(sys.modules, "training.checkpoint_probe", fake)
    monkeypatch.setattr(training, "checkpoint_probe", fake, raising=False)


def _fake_probe_module(step_dir_writes):
    """A checkpoint_probe stand-in whose dumps record the calls and write their file."""
    mod = types.ModuleType("training.checkpoint_probe")
    mod._load_checkpoint = lambda path: ("model", None)

    def _make(axis, tmpl):
        def dump(model, step_dir, step):
            step_dir_writes.append((axis, step))
            with open(f"{step_dir}/{tmpl.format(step=step)}", "w", encoding="utf-8") as f:
                json.dump({"axis": axis, "step": step}, f)
        return dump

    for axis, (fn_name, tmpl, _canonical) in rp.AXIS_TO_FN.items():
        setattr(mod, fn_name, _make(axis, tmpl))
    return mod


@pytest.fixture
def model_on_disk(tmp_path, monkeypatch):
    """Two checkpoints, only the first finalized through save.py (it has a hook dir)."""
    calls = []
    _install_fake_probe(monkeypatch, calls)
    (tmp_path / "hooks" / "step_100").mkdir(parents=True)
    monkeypatch.setattr(rp.checkpoints, "list_steps", lambda name: [100, 200])
    monkeypatch.setattr(rp.paths, "checkpoint_path", lambda name, s: str(tmp_path / f"step_{s}.pt"))
    monkeypatch.setattr(rp.paths, "hook_step_dir", lambda name, s: str(tmp_path / "hooks" / f"step_{s}"))
    return tmp_path, calls


def test_the_axis_table_names_probe_functions_that_exist():
    """A renamed probe would fail per axis at run time, one line into a long job."""
    from training import checkpoint_probe as cp
    for fn_name, _src, _canonical in rp.AXIS_TO_FN.values():
        assert callable(getattr(cp, fn_name))


def test_every_axis_runs_and_is_renamed_to_its_canonical_file(model_on_disk, monkeypatch):
    """The dashboard reads grades.json, not grades_step_100.json."""
    tmp_path, calls = model_on_disk
    monkeypatch.setattr(sys, "argv", ["reprobe", "m"])
    assert rp.main() == 0
    assert sorted(a for a, _s in calls) == sorted(rp.AXIS_TO_FN)
    for _fn, src_tmpl, canonical in rp.AXIS_TO_FN.values():
        assert (tmp_path / "hooks" / "step_100" / canonical).is_file()
        assert not (tmp_path / "hooks" / "step_100" / src_tmpl.format(step=100)).exists()


def test_a_checkpoint_with_no_hook_dir_is_skipped(model_on_disk, monkeypatch):
    """Step 200 was never finalized through save.py, so there is nowhere to write."""
    _tmp_path, calls = model_on_disk
    monkeypatch.setattr(sys, "argv", ["reprobe", "m"])
    rp.main()
    assert {s for _a, s in calls} == {100}


def test_only_runs_the_named_axes(model_on_disk, monkeypatch):
    _tmp_path, calls = model_on_disk
    monkeypatch.setattr(sys, "argv", ["reprobe", "m", "--only", "math", "grammar"])
    assert rp.main() == 0
    assert sorted(a for a, _s in calls) == ["grammar", "math"]


def test_a_step_filter_that_matches_nothing_fails_loudly(model_on_disk, monkeypatch):
    """Silently reprobing zero checkpoints looks exactly like success."""
    _tmp_path, _calls = model_on_disk
    monkeypatch.setattr(sys, "argv", ["reprobe", "m", "--steps", "999"])
    assert rp.main() == 1


def test_a_model_with_no_checkpoints_fails_loudly(monkeypatch):
    _install_fake_probe(monkeypatch, [])
    monkeypatch.setattr(rp.checkpoints, "list_steps", lambda name: [])
    monkeypatch.setattr(sys, "argv", ["reprobe", "nothing"])
    assert rp.main() == 1
