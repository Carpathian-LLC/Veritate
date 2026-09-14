# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - /logs/snapshot forwards the cursor and limit to the log ring and reports the latest
#   sequence; /logs/stream is SSE over a subscriber queue, sends a keepalive comment when
#   the ring is quiet, and unsubscribes when the client goes away. The ring is stubbed.
# tests/mri/test_logs_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import queue

import pytest
from flask import Flask
from routes import logs_routes

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def rig(monkeypatch):
    seen = {"snapshot": None, "unsubscribed": []}
    q = queue.Queue()
    monkeypatch.setattr(logs_routes.logmod, "snapshot",
                        lambda after_seq, limit: (seen.__setitem__("snapshot", (after_seq, limit)), [{"seq": 4}])[1])
    monkeypatch.setattr(logs_routes.logmod, "latest_seq", lambda: 9)
    monkeypatch.setattr(logs_routes.logmod, "subscribe", lambda: q)
    monkeypatch.setattr(logs_routes.logmod, "unsubscribe", lambda x: seen["unsubscribed"].append(x))
    monkeypatch.setattr(logs_routes, "STREAM_KEEPALIVE_SECS", 0.01)
    app = Flask(__name__)
    logs_routes.register(app)
    return app.test_client(), q, seen


def test_snapshot_forwards_the_cursor_and_limit_and_reports_the_latest_seq(rig):
    client, _q, seen = rig
    body = client.get("/logs/snapshot?after=3&limit=2").get_json()
    assert seen["snapshot"] == (3, 2)
    assert body == {"latest_seq": 9, "entries": [{"seq": 4}]}


def test_snapshot_without_a_limit_asks_for_everything_after_the_cursor(rig):
    client, _q, seen = rig
    client.get("/logs/snapshot")
    assert seen["snapshot"] == (0, None)


def test_stream_sends_queued_entries_as_sse_and_unsubscribes_on_close(rig):
    client, q, seen = rig
    q.put({"seq": 5, "msg": "hello"})
    res = client.get("/logs/stream")
    assert res.mimetype == "text/event-stream"
    chunks = iter(res.response)
    assert next(chunks) == b'data: {"seq": 5, "msg": "hello"}\n\n'
    assert next(chunks) == b": keepalive\n\n"          # the ring went quiet
    res.response.close()
    assert seen["unsubscribed"] == [q]
