# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Covers what the Distillation tab's interview panel reads while a run is live:
#   the per-call feed (what was sent, what came back, how long the reply took) and
#   the gate's banned-phrase accounting.
# - interview.ask() is the only chokepoint every teacher call in this mode passes,
#   so the feed is wired there. If a call can reach the teacher without going
#   through it, the panel silently under-reports and these tests are the guard.
# - No network: the client is a stub whose complete() returns canned text.
# tests/mri/test_interview_call_feed.py
# ------------------------------------------------------------------------------------
# Imports:

import json

import pytest
from flask import Flask
from routes import teacher_routes
from teacher import authoring, interview
from teacher.client import TeacherCancelled, TeacherError
from teacher.interview_job import (
    FAILURE_ABORT_STREAK,
    OPENER_BATCH,
    OPENERS_FILE,
    SAMPLES_FILE,
    CallFeed,
    InterviewJob,
    clip,
    conversation_id,
    percentile,
)

# ------------------------------------------------------------------------------------
# Constants

CONVERSATION = "conversation_00007"

# ------------------------------------------------------------------------------------
# Functions

class StubClient:
    """Answers every call with fixed text and records what it was asked."""

    def __init__(self, reply="a real answer that is long enough to keep around"):
        self.reply = reply
        self.seen = []

    def complete(self, messages, **kwargs):
        self.seen.append(messages[-1]["content"])
        return self.reply


class StreamingClient(StubClient):
    """A teacher that streams: the first token lands before the full reply does."""

    def complete(self, messages, **kwargs):
        first = kwargs.get("on_first_token")
        if first is not None:
            first()
        return super().complete(messages, **kwargs)


class FlakyClient(StubClient):
    """Answers `ok` calls, then raises `error` on every call after that."""

    def __init__(self, ok, error=None):
        super().__init__()
        self.ok = ok
        self.error = error or TeacherError("upstream unavailable: 503")

    def complete(self, messages, **kwargs):
        if len(self.seen) >= self.ok:
            self.seen.append(messages[-1]["content"])
            raise self.error
        return super().complete(messages, **kwargs)


def test_a_call_is_reported_at_both_edges():
    """ask() tells the watcher what went out and what came back, with a latency."""
    events = []
    interview.ask(StubClient(), [{"role": "user", "content": "how do I proof dough?"}],
                  "system", 0.9, 100, None,
                  lambda *a: events.append(a), interview.CALL_ANSWER)
    assert [e[0] for e in events] == [interview.PHASE_START, interview.PHASE_DONE]
    assert events[0][2] == "how do I proof dough?"
    assert events[1][2].startswith("a real answer")
    assert events[1][3] >= 0


def test_a_finished_call_carries_its_request_reply_and_latency():
    """The feed pairs the reply with the request that was still open."""
    feed = CallFeed()
    watch = feed.watcher(CONVERSATION)
    watch(interview.PHASE_START, interview.CALL_ANSWER, "what is a starter?", 0.0)
    watch(interview.PHASE_DONE, interview.CALL_ANSWER, "flour and water", 1234.6)
    call = feed.snapshot()["calls"][0]
    assert call["id"] == CONVERSATION
    assert call["kind"] == interview.CALL_ANSWER
    assert call["sent"] == "what is a starter?"
    assert call["got"] == "flour and water"
    assert call["ms"] == 1235
    assert call["got_bytes"] == len("flour and water")


def test_an_open_call_is_in_flight_until_the_reply_lands():
    """A request with no reply yet is what the panel shows as in flight."""
    feed = CallFeed()
    watch = feed.watcher(CONVERSATION)
    watch(interview.PHASE_START, interview.CALL_ANSWER, "still thinking", 0.0)
    live = feed.snapshot()["inflight"]
    assert len(live) == 1
    assert live[0]["sent"] == "still thinking"
    assert live[0]["elapsed_ms"] >= 0
    watch(interview.PHASE_DONE, interview.CALL_ANSWER, "done", 5.0)
    assert feed.snapshot()["inflight"] == []


def test_a_call_that_raised_stops_being_in_flight():
    """A failed call never reports a reply; its row must not sit there forever."""
    feed = CallFeed()
    feed.watcher(CONVERSATION)(interview.PHASE_START, interview.CALL_ANSWER, "boom", 0.0)
    feed.drop()
    assert feed.snapshot()["inflight"] == []


def test_the_feed_reports_throughput_and_latency_spread():
    """One slow call must not hide behind an average of fast ones."""
    feed = CallFeed()
    for i, ms in enumerate([100.0, 200.0, 300.0, 9000.0]):
        watch = feed.watcher(conversation_id("conversation", i))
        watch(interview.PHASE_START, interview.CALL_ANSWER, "q", 0.0)
        watch(interview.PHASE_DONE, interview.CALL_ANSWER, "reply", ms)
    stats = feed.snapshot()["stats"]
    assert stats["calls"] == 4
    assert stats["p50_ms"] == 300
    assert stats["p95_ms"] == 9000
    assert stats["reply_bytes"] == 4 * len("reply")


def test_the_newest_call_is_first():
    """The panel reads top-down; the run's latest call belongs at the top."""
    feed = CallFeed()
    for i in range(3):
        watch = feed.watcher(conversation_id("conversation", i))
        watch(interview.PHASE_START, interview.CALL_ANSWER, f"q{i}", 0.0)
        watch(interview.PHASE_DONE, interview.CALL_ANSWER, f"a{i}", 1.0)
    assert [c["sent"] for c in feed.snapshot()["calls"]] == ["q2", "q1", "q0"]


def test_display_text_is_clipped_on_a_character_boundary():
    """A 40 KB reply must not be shipped to the browser every two seconds."""
    out = clip("é" * 500, 20)
    assert out.endswith("...")
    assert len(out.encode("utf-8")) <= 23


def test_percentiles_of_nothing_are_zero():
    assert percentile([], 0.5) == 0


def test_a_failure_part_way_through_keeps_the_turns_already_paid_for():
    """A conversation costs 2*depth-1 calls. Losing all of them because the last
    one 503'd is the expensive way to handle a blip."""
    client = FlakyClient(ok=3)
    turns = interview.build_conversation(client, "how do I keep a starter alive?", 4)
    assert turns is not None
    assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant"]


def test_stop_keeps_what_the_conversation_already_had():
    """Stop reaches the client as TeacherCancelled. The turns generated before it
    were paid for and are written like any others."""
    client = FlakyClient(ok=1, error=TeacherCancelled("cancelled"))
    turns = interview.build_conversation(client, "why is the sky blue?", 3)
    assert [t["role"] for t in turns] == ["user", "assistant"]


def test_a_conversation_with_no_reply_at_all_is_still_dropped():
    """Salvage keeps complete exchanges, not an unanswered question."""
    assert interview.build_conversation(FlakyClient(ok=0), "anyone there?", 3) is None


def test_a_salvaged_conversation_is_counted_and_the_failure_is_kept():
    """The panel has to say a run is limping, not silently produce short records."""
    feed = CallFeed()
    interview.build_conversation(FlakyClient(ok=3), "how do I proof dough?", 4,
                                 watch=feed.watcher(CONVERSATION))
    snap = feed.snapshot()
    assert snap["stats"]["salvaged"] == 1
    assert snap["stats"]["failed"] == 1
    assert snap["inflight"] == []
    assert "TeacherError" in snap["calls"][0]["error"]


def test_a_failed_call_stays_out_of_the_latency_percentiles():
    """A call that died on a 60 s socket timeout is not a reply time."""
    feed = CallFeed()
    watch = feed.watcher(CONVERSATION)
    watch(interview.PHASE_START, interview.CALL_ANSWER, "q", 0.0)
    watch(interview.PHASE_DONE, interview.CALL_ANSWER, "a", 250.0)
    watch(interview.PHASE_START, interview.CALL_ANSWER, "q", 0.0)
    watch(interview.PHASE_FAIL, interview.CALL_ANSWER, "TeacherError: timed out", 60000.0)
    stats = feed.snapshot()["stats"]
    assert stats["p95_ms"] == 250
    assert (stats["calls"], stats["failed"]) == (1, 1)


def test_the_gate_names_the_banned_phrase_that_blocked_a_record():
    """"12 rejected" does not say which entry in the list is costing the run."""
    # The shipped list is user-editable now, so the phrase under test is pinned
    # here rather than borrowed from whatever is on disk.
    spec = authoring.load_spec()
    spec["gates"]["banned_phrases"] = ["great question"]
    gate = authoring.RecordGate(spec)
    record = {authoring.GENRE_KEY: "conversation", authoring.VOICE_KEY: "interview",
              authoring.TURNS_KEY: [
                  {"role": "user", "text": "why is the sky blue?"},
                  {"role": "assistant", "text": "Great question, it is the way light scatters "
                                                "through the atmosphere over a long path."},
                  {"role": "user", "text": "so why is a sunset red then?"},
                  {"role": "assistant", "text": "The light travels further through the air at "
                                                "that angle and the blue is scattered out of it."}]}
    kept, why = gate({"genre": "conversation"}, json.dumps(record))
    assert not kept
    assert why == [authoring.REJECT_BANNED]
    assert gate.stats()["banned_hits"] == {"great question": 1}


def test_an_empty_ban_list_bans_nothing():
    """An empty alternation would match everywhere and reject every record."""
    assert authoring.compile_ban_re([]).search("anything at all") is None


@pytest.mark.parametrize("phrase", ["as an ai", "great question"])
def test_a_saved_phrase_is_matched_whole_word(phrase):
    ban = authoring.compile_ban_re([phrase])
    assert ban.search(f"well {phrase} here")
    assert ban.search(f"x{phrase}x") is None


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Test client with the jobs root and the authoring spec redirected to tmp_path."""
    root = tmp_path / "synth_jobs"
    root.mkdir()
    spec_path = tmp_path / "corpus_spec.json"
    spec_path.write_text(json.dumps(authoring.load_spec()), encoding="utf-8")
    monkeypatch.setattr(teacher_routes.paths_mod, "synth_jobs_root", lambda: str(root))
    monkeypatch.setattr(teacher_routes.paths_mod, "synth_job_dir", lambda jid: str(root / jid))
    monkeypatch.setattr(teacher_routes.paths_mod, "authoring_spec_path", lambda: str(spec_path))
    app = Flask(__name__)
    teacher_routes.register(app)
    return app.test_client()


def test_the_call_feed_answers_empty_for_a_job_that_is_not_running(client):
    """The feed is memory on the live job; a finished run must not 500 the panel."""
    body = client.get("/teacher/synth/calls?job_id=deadbeef").get_json()
    assert body["calls"] == [] and body["inflight"] == [] and body["stats"] == {}


def test_saving_the_ban_list_normalises_and_persists_it(client):
    """The gate matches lower case; blanks and repeats are the user's, not the gate's."""
    saved = client.post("/teacher/authoring/banned",
                        json={"phrases": ["As An AI", " ", "great question", "as an ai"]}).get_json()
    assert saved["phrases"] == ["as an ai", "great question"]
    spec_path = teacher_routes.paths_mod.authoring_spec_path()
    assert authoring.load_spec(spec_path)["gates"]["banned_phrases"] == saved["phrases"]


def test_the_ban_list_can_be_emptied(client):
    """Nothing banned is a legitimate setting, not an error."""
    assert client.post("/teacher/authoring/banned", json={"phrases": []}).get_json()["phrases"] == []


def test_a_ban_list_that_is_not_a_list_is_refused(client):
    assert client.post("/teacher/authoring/banned", json={"phrases": "as an ai"}).status_code == 400


class OpenerClient(StubClient):
    """Returns a fresh batch of opening lines on every call."""

    def __init__(self):
        super().__init__()
        self.batches = 0

    def complete(self, messages, **kwargs):
        self.seen.append(messages[-1]["content"])
        self.batches += 1
        return "\n".join(f"what should I know about topic {self.batches}-{i}?"
                          for i in range(OPENER_BATCH))


def _job(tmp_path, **kw):
    spec = authoring.load_spec()
    return InterviewJob("testjob", str(tmp_path), spec, ["conversation"],
                        kw.pop("conversations", 10), 3, "openai",
                        gate=authoring.RecordGate(spec), seeds=["baking bread"], **kw)


def test_openers_are_written_to_disk_as_they_arrive(tmp_path):
    """Pass 1 is real teacher calls. An interrupted run used to pay for them
    twice: once before it died, once on the resume."""
    job = _job(tmp_path)
    out = job._openers_for(OpenerClient(), {"id": "conversation"}, OPENER_BATCH, [])
    stored = (tmp_path / OPENERS_FILE).read_text(encoding="utf-8").strip().splitlines()
    assert len(out) == OPENER_BATCH
    assert [json.loads(line)["text"] for line in stored] == out


def test_a_resume_reuses_the_openers_it_already_paid_for(tmp_path):
    """The pool on disk is used before the teacher is asked for anything new."""
    job = _job(tmp_path)
    job._openers_for(OpenerClient(), {"id": "conversation"}, OPENER_BATCH, [])
    reloaded = _job(tmp_path)._stored_openers()
    assert len(reloaded["conversation"]) == OPENER_BATCH


def test_an_opener_already_answered_is_not_asked_again(tmp_path):
    """Re-asking one costs a whole conversation for a record the gate then
    rejects as a near duplicate of the one already on disk."""
    job = _job(tmp_path)
    openers = job._openers_for(OpenerClient(), {"id": "conversation"}, OPENER_BATCH, [])
    record = {authoring.GENRE_KEY: "conversation", authoring.VOICE_KEY: "interview",
              authoring.TURNS_KEY: [{"role": "user", "text": openers[0]},
                                    {"role": "assistant", "text": "an answer"}]}
    (tmp_path / SAMPLES_FILE).write_text(
        json.dumps({"id": "conversation_00000", authoring.RECORD_KEY: record}) + "\n",
        encoding="utf-8")
    assert openers[0] not in _job(tmp_path)._stored_openers()["conversation"]


def test_a_half_written_line_does_not_stop_a_resume(tmp_path):
    """A run appends and flushes as it goes, so a machine crash can tear the last
    line. Losing the whole pool over one bad line is the wrong trade."""
    job = _job(tmp_path)
    job._openers_for(OpenerClient(), {"id": "conversation"}, OPENER_BATCH, [])
    with open(tmp_path / OPENERS_FILE, "a", encoding="utf-8") as f:
        f.write('{"genre": "conversation", "text": "torn')
    assert len(_job(tmp_path)._stored_openers()["conversation"]) == OPENER_BATCH


def test_an_opener_already_in_hand_is_not_generated_twice(tmp_path):
    """`taken` is the pool loaded from disk; a batch that repeats it is waste."""
    job = _job(tmp_path)
    client = OpenerClient()
    first = job._openers_for(client, {"id": "conversation"}, OPENER_BATCH, [])
    again = job._openers_for(client, {"id": "conversation"}, OPENER_BATCH, first)
    assert not set(first) & set(again)


def test_a_dead_teacher_stops_the_opener_pass_instead_of_grinding(tmp_path):
    """Every batch failing used to run the full round budget, each round paying
    the client's own five retries. On a large run that is hours of nothing."""
    job = _job(tmp_path, conversations=2400)
    client = FlakyClient(ok=0)
    assert job._openers_for(client, {"id": "conversation"}, 200, []) == []
    assert len(client.seen) == FAILURE_ABORT_STREAK
    assert job._aborted


class StoppingClient(StubClient):
    """Answers normally, then stops the job mid-conversation the way the Stop
    button does: the client's own cancel check raises TeacherCancelled."""

    def __init__(self, job, stop_on):
        super().__init__()
        self.job = job
        self.stop_on = stop_on

    def complete(self, messages, **kwargs):
        self.seen.append(messages[-1]["content"])
        if len(self.seen) >= self.stop_on:
            self.job.stop()
            raise TeacherCancelled("cancelled")
        return (f"Answer number {len(self.seen)}: the dough needs a warm spot and "
                f"a few hours, and the timing shifts with the weather in the room.")


def test_stopping_keeps_the_conversation_that_was_in_flight(tmp_path):
    """Stop drops what is queued and keeps reading the rest. The run had already
    paid for the turns in flight; discarding them was the loss this guards."""
    job = _job(tmp_path, conversations=3)
    job.depth = 3
    job.max_concurrency = 1
    with open(tmp_path / OPENERS_FILE, "w", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps({"genre": "conversation",
                                "text": f"how do I get bread {i} to rise properly?"}) + "\n")
    client = StoppingClient(job, stop_on=5)
    job._client = lambda: client
    job.run()
    rows = [json.loads(line) for line in
            (tmp_path / SAMPLES_FILE).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert len(rows[0][authoring.RECORD_KEY][authoring.TURNS_KEY]) == 4


def test_a_pool_that_comes_up_short_is_reported_against_what_was_asked(tmp_path):
    """The shortfall counts openers the run never got, whether they came off disk
    or off the teacher. It is what tells the user asking for more cannot help."""
    job = _job(tmp_path, conversations=3)
    with open(tmp_path / OPENERS_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps({"genre": "conversation", "text": "why does bread go stale?"}) + "\n")
    job._client = lambda: FlakyClient(ok=0)
    job.run()
    assert job._opener_shortfall == {"conversation": 2}


def test_the_feed_splits_waiting_for_a_reply_from_reading_one():
    """A teacher that is busy and a teacher that is slow are the same number in a
    single total. The first-token mark is the only thing that tells them apart."""
    feed = CallFeed()
    interview.ask(StreamingClient(), [{"role": "user", "content": "why does dough rise?"}],
                  "system", 0.9, 100, None, feed.watcher(CONVERSATION), interview.CALL_ANSWER)
    call = feed.snapshot()["calls"][0]
    assert call["wait_ms"] is not None
    assert call["wait_ms"] <= call["ms"]


def test_a_provider_that_cannot_stream_never_claims_a_first_word():
    """No stream, no first-token signal. Reporting one would be invented data."""
    feed = CallFeed()
    interview.ask(StubClient(), [{"role": "user", "content": "q"}], "system", 0.9, 100,
                  None, feed.watcher(CONVERSATION), interview.CALL_ANSWER)
    assert feed.snapshot()["calls"][0]["wait_ms"] is None


def test_an_open_call_says_when_the_reply_started_arriving():
    """Sending and receiving are different states and the panel shows which."""
    feed = CallFeed()
    watch = feed.watcher(CONVERSATION)
    watch(interview.PHASE_START, interview.CALL_ANSWER, "q", 0.0)
    assert feed.snapshot()["inflight"][0]["wait_ms"] is None
    watch(interview.PHASE_FIRST, interview.CALL_ANSWER, "", 120.4)
    assert feed.snapshot()["inflight"][0]["wait_ms"] == 120


def test_a_retry_that_streams_twice_keeps_the_first_mark():
    """The mark is measured from the request that is still open, not from the
    last attempt, or a retried call would look faster than it was."""
    feed = CallFeed()
    watch = feed.watcher(CONVERSATION)
    watch(interview.PHASE_START, interview.CALL_ANSWER, "q", 0.0)
    watch(interview.PHASE_FIRST, interview.CALL_ANSWER, "", 100.0)
    watch(interview.PHASE_FIRST, interview.CALL_ANSWER, "", 8000.0)
    assert feed.snapshot()["inflight"][0]["wait_ms"] == 100


def test_the_first_word_median_is_reported_separately():
    """Time to the first word is the number that says whether to raise concurrency."""
    feed = CallFeed()
    for ms in (100.0, 300.0, 500.0):
        watch = feed.watcher(CONVERSATION)
        watch(interview.PHASE_START, interview.CALL_ANSWER, "q", 0.0)
        watch(interview.PHASE_FIRST, interview.CALL_ANSWER, "", ms)
        watch(interview.PHASE_DONE, interview.CALL_ANSWER, "reply", ms * 4)
    assert feed.snapshot()["stats"]["p50_first_ms"] == 300


def test_the_streaming_client_reports_the_first_token_once(monkeypatch):
    """The signal comes off the wire, so this is what the whole split rests on."""
    import io as _io
    import urllib.request as _req

    from teacher import client as client_mod

    lines = [b'data: {"choices":[{"delta":{"content":"first"}}]}\n',
             b'data: {"choices":[{"delta":{"content":" second"}}]}\n',
             b"data: [DONE]\n"]

    class FakeResponse(_io.IOBase):
        def __iter__(self):
            return iter(lines)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(_req, "urlopen", lambda *a, **kw: FakeResponse())
    marks = []
    c = client_mod.Client("openai", model="m", base_url="https://teacher.invalid", api_key="k")
    out = c.complete([{"role": "user", "content": "q"}], cancel_check=lambda: False,
                     on_first_token=lambda: marks.append(1))
    assert out == "first second"
    assert len(marks) == 1


def test_a_deep_run_is_not_rejected_by_the_genres_turn_bound(client, monkeypatch):
    """`conversation` caps at 8 turns and a conversation of D replies is 2D turns,
    so every complete conversation past depth 4 was thrown away as "turn count out
    of range" after paying for all 2D-1 calls. The requested depth is the shape."""
    started = {}
    monkeypatch.setattr(teacher_routes.settings_mod, "get",
                        lambda: {"teacher_provider": "openai", "teacher_model": "m"})
    monkeypatch.setattr(teacher_routes.interview_mod, "InterviewJob",
                        lambda *a, **kw: started.setdefault("job", _Recorded(*a, **kw)))
    monkeypatch.setattr(teacher_routes.threading, "Thread", _NoThread)
    r = client.post("/teacher/interview/start",
                    json={"genres": ["conversation"], "conversations": 4, "depth": 5,
                          "vertical": "conversation"})
    assert r.status_code == 200
    spec = started["job"].spec
    assert authoring.genre_by_id(spec, "conversation")["max_turns"] >= 10


class _Recorded:
    def __init__(self, job_id, output_dir, spec, *a, **kw):
        self.spec = spec
        self.job_id = job_id

    def run(self):
        return None


class _NoThread:
    def __init__(self, *a, **kw):
        pass

    def start(self):
        return None

    def is_alive(self):
        return False
