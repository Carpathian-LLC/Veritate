# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the full hook suite costs ~137s per checkpoint on a 200m trunk, measured on
#   three consecutive wren_base checkpoints. Nearly all of it is the five heavy
#   dumps; the rest is ~9s. A run that wants dense checkpoints for crash recovery
#   should not have to buy 137s of probes with each one.
# - these pin the three modes, the every-Nth promotion, and the invariant that
#   "off" really means every dump save() knows about.
# tests/training/test_trainer_hook_plan.py
# ------------------------------------------------------------------------------------
# Imports:

import importlib.util
import os
import sys
import types

import pytest

# ------------------------------------------------------------------------------------
# Constants

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAINER = os.path.join(REPO, "veritate_mri", "training", "veritate_trainer.py")

# ------------------------------------------------------------------------------------
# Fixtures


@pytest.fixture(scope="module")
def trainer():
    if os.path.join(REPO, "veritate_mri") not in sys.path:
        sys.path.insert(0, os.path.join(REPO, "veritate_mri"))
    spec = importlib.util.spec_from_file_location("veritate_trainer_hooks_test", TRAINER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(hooks="full", every=0, ckpt_every=250):
    return types.SimpleNamespace(hooks=hooks, hooks_full_every=every, ckpt_every=ckpt_every)


# ------------------------------------------------------------------------------------
# Tests


def test_full_skips_nothing(trainer):
    skip, label = trainer.hook_plan(_args("full"), 500)
    assert skip == set()
    assert label == "full"


def test_default_is_full(trainer):
    """A run launched before this flag existed must behave exactly as it did."""
    skip, _ = trainer.hook_plan(types.SimpleNamespace(ckpt_every=250), 500)
    assert skip == set()


def test_light_skips_exactly_the_heavy_dumps(trainer):
    skip, label = trainer.hook_plan(_args("light"), 250)
    assert skip == set(trainer.save.HEAVY_DUMPS)
    assert label == "light"


def test_light_keeps_the_cheap_probes(trainer):
    """probe/lens/classroom/grades/surprise/quant_kl are the ~9s that stay."""
    skip, _ = trainer.hook_plan(_args("light"), 250)
    for cheap in ("probe", "classroom", "grades", "grammar", "concepts",
                  "surprise", "quant_kl"):
        assert cheap not in skip


def test_off_skips_every_dump_save_knows_about(trainer):
    skip, label = trainer.hook_plan(_args("off"), 500)
    assert skip == set(trainer.save.ALL_DUMPS)
    assert label == "off"


def test_every_nth_checkpoint_is_promoted_to_full(trainer):
    """ckpt_every 250, promote every 4th checkpoint -> steps 1000, 2000, ..."""
    args = _args("light", every=4, ckpt_every=250)
    assert trainer.hook_plan(args, 1000)[1] == "full"
    assert trainer.hook_plan(args, 2000)[1] == "full"
    assert trainer.hook_plan(args, 750)[1] == "light"
    assert trainer.hook_plan(args, 1250)[1] == "light"


def test_promotion_counts_checkpoints_not_steps(trainer):
    """Halving ckpt_every must not halve the promotion interval in steps."""
    a = _args("light", every=2, ckpt_every=500)
    b = _args("light", every=2, ckpt_every=250)
    assert trainer.hook_plan(a, 1000)[1] == "full"
    assert trainer.hook_plan(b, 500)[1] == "full"
    assert trainer.hook_plan(b, 750)[1] == "light"


def test_promotion_off_when_every_is_zero(trainer):
    args = _args("light", every=0, ckpt_every=250)
    assert all(trainer.hook_plan(args, s)[1] == "light"
               for s in (250, 500, 750, 1000, 2000))


def test_full_ignores_the_promotion_knob(trainer):
    skip, label = trainer.hook_plan(_args("full", every=4), 750)
    assert skip == set() and label == "full"


def test_off_ignores_the_promotion_knob(trainer):
    """`off` is a deliberate choice; a promotion must not resurrect the suite."""
    skip, label = trainer.hook_plan(_args("off", every=4, ckpt_every=250), 1000)
    assert skip == set(trainer.save.ALL_DUMPS) and label == "off"


def test_unknown_mode_refuses_to_start(trainer):
    """A typo must not silently fall through to some mode the user didn't pick."""
    with pytest.raises(SystemExit) as e:
        trainer.hook_plan(_args("ligt"), 250)
    assert "ligt" in str(e.value)


def test_mode_is_case_and_space_tolerant(trainer):
    assert trainer.hook_plan(_args(" Light "), 250)[1] == "light"


def test_heavy_dumps_are_a_subset_of_all_dumps(trainer):
    """Guards against renaming a dump in one set and not the other."""
    assert set(trainer.save.HEAVY_DUMPS) <= set(trainer.save.ALL_DUMPS)


def test_flags_parse_from_argv(trainer, monkeypatch):
    """The dashboard sends these as plain argv; they must not be unknown flags."""
    monkeypatch.setattr(sys, "argv", ["veritate_trainer.py",
                                      "--hooks", "light", "--hooks_full_every", "4"])
    args = trainer.parse_args({"description": "t", "defaults": {"ckpt_every": 250}})
    assert args.hooks == "light"
    assert args.hooks_full_every == 4
