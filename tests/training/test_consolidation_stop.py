# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers veritate_trainer.val_rose_past_best, the consolidation stop rule. A sleep run
#   has no useful fixed length: measured on wren1_8 (2026-08-25, lr 2e-4 over 48 study
#   chunks) held-out val improved through step 10 (0.9782) then fell off a cliff by step
#   20 (1.1122). A fixed step count either stops short of the gain or trains straight
#   through the damage, which is what made the 2026-08-24 run degrade the model for 512
#   steps with the signal sitting in train.csv the whole time.
# - the rule compares against the run's BEST, not the previous reading: a slow monotone
#   drift never trips a previous-reading comparison while the total walks steadily up.
# tests/training/test_consolidation_stop.py
# ------------------------------------------------------------------------------------
# Imports:

from training.veritate_trainer import val_rose_past_best

# ------------------------------------------------------------------------------------
# Functions


def test_zero_tolerance_disables_the_rule():
    """An ordinary pretrain run must not acquire an early stop because the flag exists."""
    assert val_rose_past_best(99.0, 0.5, 0.0) is False
    assert val_rose_past_best(99.0, 0.5, None) is False


def test_rise_beyond_tolerance_stops():
    """The wren1_8 cliff: best 0.9782 at step 10, 1.1122 at step 20 is +13.7%."""
    assert val_rose_past_best(1.1122, 0.9782, 0.02) is True


def test_rise_inside_tolerance_continues():
    """The rule is a cliff detector, not a quality optimizer; ordinary wobble continues."""
    assert val_rose_past_best(0.9880, 0.9782, 0.02) is False


def test_improvement_never_stops():
    """A run that is still getting better must never be cut short."""
    assert val_rose_past_best(0.9000, 0.9782, 0.02) is False


def test_no_best_yet_cannot_trip():
    """Before any reading there is nothing to compare against, so the first eval of a
    run can never stop it."""
    assert val_rose_past_best(1.5, float("inf"), 0.02) is False
    assert val_rose_past_best(1.5, None, 0.02) is False


def test_compares_against_best_not_previous():
    """A drift where each reading sits inside tolerance against the one before it still
    trips against the high-water mark. This is the failure that let wren1_3 walk +1.8%
    across five runs with no single run above 0.5%."""
    best = 1.0
    readings = [1.015, 1.030, 1.045]
    assert [val_rose_past_best(v, best, 0.02) for v in readings] == [False, True, True]
