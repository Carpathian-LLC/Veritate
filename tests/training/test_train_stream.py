# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the live training stream is in-memory pub/sub: every subscriber gets every payload,
#   a slow subscriber's full queue drops rather than blocks the trainer, a quiet stream
#   yields None on the keepalive tick, and a closed subscriber leaves the list.
# tests/training/test_train_stream.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
from training import train_stream

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture(autouse=True)
def _clean_subscribers(monkeypatch):
    monkeypatch.setattr(train_stream, "SUBSCRIBERS", [])
    monkeypatch.setattr(train_stream, "SUBSCRIBE_TICK_S", 0.01)


def test_every_subscriber_receives_every_payload():
    a, b = train_stream.subscribe(), train_stream.subscribe()
    next(a), next(b)                      # register both (the first tick yields None)
    train_stream.publish({"step": 1})
    assert next(a) == {"step": 1} and next(b) == {"step": 1}
    a.close(), b.close()


def test_a_quiet_stream_yields_none_on_the_tick_and_a_closed_one_unsubscribes():
    s = train_stream.subscribe()
    assert next(s) is None
    assert len(train_stream.SUBSCRIBERS) == 1
    s.close()
    assert train_stream.SUBSCRIBERS == []


def test_a_full_queue_drops_the_payload_instead_of_blocking_the_trainer(monkeypatch):
    monkeypatch.setattr(train_stream, "QUEUE_MAX", 2)
    s = train_stream.subscribe()
    next(s)
    for i in range(5):
        train_stream.publish({"step": i})     # must return at once
    assert [next(s), next(s), next(s)] == [{"step": 0}, {"step": 1}, None]
    s.close()
