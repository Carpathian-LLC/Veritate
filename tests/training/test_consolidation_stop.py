# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers veritate_trainer.val_rose_past_start, the consolidation stop rule. A sleep run
#   has no useful fixed length: measured on wren1_8 (2026-08-25, lr 2e-4 over 48 study
#   chunks) held-out val improved through step 10 (0.9782) then fell off a cliff by step
#   20 (1.1122). A fixed step count either stops short of the gain or trains straight
#   through the damage, which is what made the 2026-08-24 run degrade the model for 512
#   steps with the signal sitting in train.csv the whole time.
# - the rule compares against the run's FIRST reading, the weights it started from: a
#   slow monotone drift never trips a previous-reading comparison while the total walks
#   steadily up, and a best-reading comparison halts a run that is improving on its start
#   for wobbling above its own best (wren1_12, 2026-08-26: better than step 0 on both chat
#   corpora at every checkpoint, stopped at step 150 for +3.4% over its step-50 best).
# tests/training/test_consolidation_stop.py
# ------------------------------------------------------------------------------------
# Imports:

from training.veritate_trainer import val_rose_past_start

# ------------------------------------------------------------------------------------
# Functions


def test_zero_tolerance_disables_the_rule():
    """An ordinary pretrain run must not acquire an early stop because the flag exists."""
    assert val_rose_past_start([0.5, 99.0], 0.0) is False
    assert val_rose_past_start([0.5, 99.0], None) is False


def test_rise_beyond_tolerance_on_two_consecutive_readings_stops():
    """The wren1_8 cliff, start 0.9782 to 1.1122 (+13.7%), held for a second reading."""
    assert val_rose_past_start([0.9782, 1.1122, 1.1150], 0.02) is True


def test_one_reading_above_the_line_is_a_transient_not_a_stop():
    """exp_fastsleep_0902 (2026-09-02): +12.0% at step 20 was +1.9% at step 40 with the
    first facts bound. One reading above the line continues; the next one decides."""
    assert val_rose_past_start([1.006792, 1.127872], 0.02) is False
    assert val_rose_past_start([1.006792, 1.127872, 1.025540], 0.02) is False


def test_rise_inside_tolerance_continues():
    """The rule is a cliff detector, not a quality optimizer; ordinary wobble continues."""
    assert val_rose_past_start([0.9782, 0.9880], 0.02) is False


def test_improvement_never_stops():
    """A run that is still getting better must never be cut short."""
    assert val_rose_past_start([0.9782, 0.9000], 0.02) is False


def test_fewer_than_two_readings_cannot_trip():
    """The first reading is the reference, so there is nothing to compare it against."""
    assert val_rose_past_start([], 0.02) is False
    assert val_rose_past_start([1.5], 0.02) is False
    assert val_rose_past_start([1.5, 9.0], 0.02) is False


def test_compares_against_start_not_previous():
    """A drift where each reading sits inside tolerance against the one before it still
    trips against the start. This is the failure that let wren1_3 walk +1.8% across five
    runs with no single run above 0.5%."""
    readings = [1.0, 1.015, 1.030, 1.045]
    assert [val_rose_past_start(readings[:i], 0.02) for i in range(2, 5)] == [False, False, True]


def test_wobble_above_best_but_below_start_continues():
    """The wren1_12 shape: better than the starting weights at every reading, 3.4% above
    the run's own best at the last one. Worse than its best is not worse than its start,
    so the run continues."""
    assert val_rose_past_start([1.0000, 0.9464, 0.9500, 0.9782], 0.02) is False
