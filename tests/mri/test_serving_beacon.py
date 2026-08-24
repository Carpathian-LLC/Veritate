# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers runtime/serving.py, the live signal background work reads to get out
#   of a served request's way. Pins: concurrent streams are counted (not a bool),
#   idle_s stays None while anything streams, began() fires registered hooks, and
#   a raising hook never breaks serving.
# tests/mri/test_serving_beacon.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
from runtime import serving

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture(autouse=True)
def clean():
    serving.reset()
    yield
    serving.reset()


def test_idle_is_none_before_anything_is_served():
    """A box that has served nothing reports no idle time, not zero."""
    assert serving.idle_s() is None
    assert not serving.active()


def test_active_while_a_stream_runs():
    """active() holds between began and ended."""
    serving.began()
    assert serving.active()
    assert serving.idle_s() is None
    serving.ended()
    assert not serving.active()


def test_concurrent_streams_are_counted():
    """The first of two overlapping requests to finish must not declare quiet."""
    serving.began()
    serving.began()
    serving.ended()
    assert serving.active()
    assert serving.idle_s() is None
    serving.ended()
    assert not serving.active()


def test_idle_clock_starts_after_the_last_stream():
    """idle_s becomes a real number once the last stream ends."""
    serving.began()
    serving.ended()
    idle = serving.idle_s()
    assert idle is not None and idle >= 0.0


def test_began_fires_registered_hooks():
    """Background work registers a yield callback and it runs on stream start."""
    fired = []
    serving.on_began(lambda: fired.append(1))
    serving.began()
    assert fired == [1]


def test_a_raising_hook_never_breaks_serving():
    """A failing yield callback must not propagate into the serving path."""
    def boom():
        raise RuntimeError("no")

    serving.on_began(boom)
    serving.began()          # must not raise
    assert serving.active()
