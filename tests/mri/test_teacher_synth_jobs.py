# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Tests for the job-management side of /teacher/synth/*: what the Distillation
#   tab's "corpora on disk" list reads, renames, and deletes.
# - A job id is a bare directory name under the jobs root, so the traversal guard
#   is the security-relevant assertion here: a rename or delete that resolves
#   outside that root must 404, never touch the path.
# - `_count_lines` is the hot path: the corpora list counts every job on disk and
#   a running job is polled every couple of seconds, so it resumes from a cached
#   byte offset. The append / truncate / partial-record cases below are what make
#   that resume safe; get them wrong and a record count silently drifts.
# - The jobs root is redirected to tmp_path; nothing reads or writes real data.
# tests/mri/test_teacher_synth_jobs.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os

import pytest
from flask import Flask
from routes import teacher_routes

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture
def client(tmp_path, monkeypatch):
    """Test client with the synth jobs root redirected into tmp_path."""
    root = tmp_path / "synth_jobs"
    root.mkdir()
    monkeypatch.setattr(teacher_routes.paths_mod, "synth_jobs_root", lambda: str(root))
    monkeypatch.setattr(teacher_routes.paths_mod, "synth_job_dir",
                        lambda jid: str(root / jid))
    app = Flask(__name__)
    teacher_routes.register(app)
    return app.test_client()


def _make_job(client, job_id, records=2):
    root = teacher_routes.paths_mod.synth_jobs_root()
    d = os.path.join(root, job_id)
    os.makedirs(d)
    with open(os.path.join(d, teacher_routes.SAMPLES_FILE), "w", encoding="utf-8") as f:
        for i in range(records):
            f.write(json.dumps({"response": f"r{i}"}) + "\n")
    return d


def _jobs(client):
    r = client.get("/teacher/synth/jobs")
    assert r.status_code == 200
    return r.get_json()["jobs"]


def test_a_job_reports_records_bytes_and_mtime(client):
    """Every field the corpora list renders comes from this one payload."""
    _make_job(client, "abc123", records=3)
    j = _jobs(client)[0]
    assert j["job_id"] == "abc123"
    assert j["completed"] == 3
    assert j["bytes"] > 0
    assert j["updated_at"] > 0
    assert j["label"] == ""


def test_rename_sets_a_label_the_pickers_can_show(client):
    """A hex job id is unrecognisable a week later; the label is the fix."""
    _make_job(client, "abc123")
    r = client.post("/teacher/synth/rename", json={"job_id": "abc123", "label": "medical v2"})
    assert r.status_code == 200
    assert _jobs(client)[0]["label"] == "medical v2"


def test_rename_keeps_the_categories_already_on_disk(client):
    """meta.json holds seeds and categories too; a rename must not blank them."""
    d = _make_job(client, "abc123")
    teacher_routes._write_job_meta(d, ["seed_a"], ["medical"])
    client.post("/teacher/synth/rename", json={"job_id": "abc123", "label": "kept"})
    j = _jobs(client)[0]
    assert j["categories"] == ["medical"]
    assert j["seeds"] == ["seed_a"]
    assert j["label"] == "kept"


def test_an_empty_label_clears_back_to_the_id(client):
    """The rename prompt offers blank as the way out of a bad name."""
    d = _make_job(client, "abc123")
    teacher_routes._save_job_meta(d, {"label": "old"})
    client.post("/teacher/synth/rename", json={"job_id": "abc123", "label": ""})
    assert _jobs(client)[0]["label"] == ""


def test_a_label_is_capped(client):
    """An unbounded label would wreck the picker's option width."""
    _make_job(client, "abc123")
    r = client.post("/teacher/synth/rename", json={"job_id": "abc123", "label": "x" * 500})
    assert len(r.get_json()["label"]) == teacher_routes.JOB_LABEL_MAX


def test_rename_of_an_unknown_job_is_a_404(client):
    """No directory, no meta file written anywhere."""
    assert client.post("/teacher/synth/rename", json={"job_id": "nope", "label": "x"}).status_code == 404


def test_rename_cannot_escape_the_jobs_root(client, tmp_path):
    """A job id is a directory name, so `..` must not resolve out of the root."""
    outside = tmp_path / "outside"
    outside.mkdir()
    r = client.post("/teacher/synth/rename", json={"job_id": "../outside", "label": "x"})
    assert r.status_code == 404
    assert not (outside / teacher_routes.JOB_META_FILE).exists()


def test_delete_removes_the_directory(client):
    """Deleting a job takes its files with it and drops it from the list."""
    d = _make_job(client, "abc123")
    assert client.post("/teacher/synth/delete", json={"job_id": "abc123"}).status_code == 200
    assert not os.path.isdir(d)
    assert _jobs(client) == []


def test_delete_cannot_escape_the_jobs_root(client, tmp_path):
    """The same traversal guard, on the destructive route."""
    outside = tmp_path / "outside"
    outside.mkdir()
    assert client.post("/teacher/synth/delete", json={"job_id": "../outside"}).status_code == 404
    assert outside.is_dir()


# ------------------------------------------------------------------------------------
# Incremental line counting

@pytest.fixture(autouse=True)
def _clear_count_cache():
    """The cache is module state; a stale entry would leak between tests."""
    teacher_routes._LINE_COUNTS.clear()
    yield
    teacher_routes._LINE_COUNTS.clear()


def _write(path, text, mode="w"):
    """Write and force a distinct mtime, so a same-second rewrite is still seen."""
    with open(path, mode, encoding="utf-8") as f:
        f.write(text)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


def _append(path, text):
    _write(path, text, mode="a")


def test_counting_an_absent_file_is_zero(tmp_path):
    """A job directory exists before its first record is written."""
    assert teacher_routes._count_lines(str(tmp_path / "nope.jsonl")) == 0


def test_blank_lines_are_not_records(tmp_path):
    """A trailing newline must not inflate the count by one."""
    f = tmp_path / "s.jsonl"
    f.write_text("a\nb\n\n\n", encoding="utf-8")
    assert teacher_routes._count_lines(str(f)) == 2


def test_an_unchanged_file_is_not_reread(tmp_path):
    """The whole point of the cache: a second call does no I/O on the contents."""
    f = tmp_path / "s.jsonl"
    f.write_text("a\nb\n", encoding="utf-8")
    assert teacher_routes._count_lines(str(f)) == 2
    f.unlink()
    (tmp_path / "s.jsonl").write_bytes(b"")   # same path, but stat now differs
    assert teacher_routes._count_lines(str(f)) == 0


def test_appending_resumes_from_the_cached_offset(tmp_path):
    """The case this exists for: a running job appends and the count still lands
    exactly, without re-reading what it already counted."""
    f = str(tmp_path / "s.jsonl")
    _write(f, "a\nb\n")
    assert teacher_routes._count_lines(f) == 2
    _append(f, "c\nd\ne\n")
    assert teacher_routes._count_lines(f) == 5


def test_a_partial_record_is_not_counted_twice(tmp_path):
    """A writer caught mid-record leaves the offset off a line boundary. Resuming
    there would count the head now and the tail again on the next pass."""
    f = str(tmp_path / "s.jsonl")
    _write(f, "a\nb\npart")
    assert teacher_routes._count_lines(f) == 3
    _append(f, "ial\n")
    assert teacher_routes._count_lines(f) == 3


def test_a_truncated_file_is_recounted(tmp_path):
    """A shrunk file invalidates every cached offset."""
    f = str(tmp_path / "s.jsonl")
    _write(f, "a\nb\nc\n")
    assert teacher_routes._count_lines(f) == 3
    _write(f, "a\n")
    assert teacher_routes._count_lines(f) == 1


def test_the_cache_is_bounded(tmp_path):
    """Job ids are random, so every deleted-and-recreated job is a new key."""
    for i in range(teacher_routes.LINE_COUNT_CACHE_MAX + 5):
        f = tmp_path / f"s{i}.jsonl"
        f.write_text("a\n", encoding="utf-8")
        teacher_routes._count_lines(str(f))
    assert len(teacher_routes._LINE_COUNTS) <= teacher_routes.LINE_COUNT_CACHE_MAX


def test_deleting_a_job_drops_its_cached_count(client):
    """A new job could otherwise land on a freed path and inherit a stale count."""
    d = _make_job(client, "abc123", records=4)
    assert _jobs(client)[0]["completed"] == 4
    samples = os.path.join(d, teacher_routes.SAMPLES_FILE)
    assert samples in teacher_routes._LINE_COUNTS
    client.post("/teacher/synth/delete", json={"job_id": "abc123"})
    assert samples not in teacher_routes._LINE_COUNTS


# ------------------------------------------------------------------------------------
# /teacher/synth/browse — reading a corpus, not just listing it.
# `/teacher/synth/samples` tails the live job and cannot page; browse opens any
# job on disk at an arbitrary offset, so the traversal guard applies here too.

def _make_convo_job(job_id, records=5):
    root = teacher_routes.paths_mod.synth_jobs_root()
    d = os.path.join(root, job_id)
    os.makedirs(d)
    with open(os.path.join(d, teacher_routes.SAMPLES_FILE), "w", encoding="utf-8") as f:
        for i in range(records):
            f.write(json.dumps({"id": f"rec_{i}", "record": {
                "genre": "carryover", "voice": "pm_deadline",
                "turns": [{"role": "user", "text": f"q{i}"},
                          {"role": "assistant", "text": f"a{i}"}]}}) + "\n")
    return d


def test_browse_returns_a_page_and_the_true_total(client):
    _make_convo_job("job1", records=40)
    r = client.get("/teacher/synth/browse?job_id=job1&offset=0&limit=10")
    assert r.status_code == 200
    d = r.get_json()
    assert d["total"] == 40 and d["offset"] == 0
    assert len(d["rows"]) == 10
    assert d["rows"][0]["id"] == "rec_0"


def test_browse_pages_from_an_offset(client):
    _make_convo_job("job1", records=40)
    d = client.get("/teacher/synth/browse?job_id=job1&offset=25&limit=5").get_json()
    assert [x["id"] for x in d["rows"]] == [f"rec_{i}" for i in range(25, 30)]


def test_browse_past_the_end_is_an_empty_page_not_an_error(client):
    _make_convo_job("job1", records=3)
    d = client.get("/teacher/synth/browse?job_id=job1&offset=900&limit=10").get_json()
    assert d["rows"] == [] and d["total"] == 3


def test_browse_keeps_turns_structured_for_the_viewer(client):
    _make_convo_job("job1", records=1)
    row = client.get("/teacher/synth/browse?job_id=job1").get_json()["rows"][0]
    assert row["genre"] == "carryover" and row["voice"] == "pm_deadline"
    assert row["turns"] == [{"role": "user", "text": "q0"},
                            {"role": "assistant", "text": "a0"}]


def test_browse_falls_back_to_flat_text_for_raw_synth_rows(client):
    _make_job(client, "job2", records=2)   # writes {"response": ...} rows
    row = client.get("/teacher/synth/browse?job_id=job2").get_json()["rows"][0]
    assert row["text"] == "r0" and "turns" not in row


def test_browse_truncates_a_runaway_record(client):
    root = teacher_routes.paths_mod.synth_jobs_root()
    d = os.path.join(root, "job3")
    os.makedirs(d)
    with open(os.path.join(d, teacher_routes.SAMPLES_FILE), "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "big", "response": "x" * 50000}) + "\n")
    row = client.get("/teacher/synth/browse?job_id=job3").get_json()["rows"][0]
    assert len(row["text"]) == teacher_routes.BROWSE_TEXT_MAX


def test_browse_survives_an_unreadable_line(client):
    root = teacher_routes.paths_mod.synth_jobs_root()
    d = os.path.join(root, "job4")
    os.makedirs(d)
    with open(os.path.join(d, teacher_routes.SAMPLES_FILE), "w", encoding="utf-8") as f:
        f.write('{"response": "ok"}\n')
        f.write("{not json\n")
        f.write('{"response": "also ok"}\n')
    rows = client.get("/teacher/synth/browse?job_id=job4").get_json()["rows"]
    assert [r["text"] for r in rows] == ["ok", "(unreadable record)", "also ok"]


def test_browse_caps_the_page_size(client):
    _make_convo_job("job1", records=500)
    d = client.get("/teacher/synth/browse?job_id=job1&limit=9999").get_json()
    assert d["limit"] == teacher_routes.BROWSE_PAGE_MAX
    assert len(d["rows"]) == teacher_routes.BROWSE_PAGE_MAX


def test_browse_ignores_a_junk_offset(client):
    _make_convo_job("job1", records=4)
    d = client.get("/teacher/synth/browse?job_id=job1&offset=-9&limit=zz").get_json()
    assert d["offset"] == 0 and d["limit"] == teacher_routes.BROWSE_PAGE_DEFAULT


def test_browse_of_an_unknown_job_is_a_404(client):
    assert client.get("/teacher/synth/browse?job_id=nope").status_code == 404
    assert client.get("/teacher/synth/browse").status_code == 400


def test_browse_cannot_escape_the_jobs_root(client, tmp_path):
    outside = tmp_path / "secret"
    outside.mkdir()
    (outside / teacher_routes.SAMPLES_FILE).write_text('{"response": "leak"}\n', encoding="utf-8")
    assert client.get("/teacher/synth/browse?job_id=../secret").status_code == 404


def test_browse_reports_the_label_so_the_viewer_can_title_itself(client):
    _make_convo_job("job1", records=1)
    client.post("/teacher/synth/rename", json={"job_id": "job1", "label": "bedside v2"})
    assert client.get("/teacher/synth/browse?job_id=job1").get_json()["label"] == "bedside v2"


# ------------------------------------------------------------------------------------
# Naming a corpus at creation: the same `label` field rename writes, set once so a
# corpus never has to carry a hex id.

def test_a_start_time_label_names_a_new_corpus(client):
    root = teacher_routes.paths_mod.synth_jobs_root()
    d = os.path.join(root, "job1")
    os.makedirs(d)
    teacher_routes._write_job_meta(d, [], ["instruct"], "  my corpus  ")
    assert teacher_routes._read_job_meta(d)["label"] == "my corpus"


def test_appending_without_a_name_keeps_the_name_already_there(client):
    root = teacher_routes.paths_mod.synth_jobs_root()
    d = os.path.join(root, "job1")
    os.makedirs(d)
    teacher_routes._write_job_meta(d, [], ["instruct"], "keep me")
    teacher_routes._write_job_meta(d, [], ["chat"], "")
    meta = teacher_routes._read_job_meta(d)
    assert meta["label"] == "keep me"
    assert meta["categories"] == ["chat", "instruct"]


def test_a_start_time_label_is_capped(client):
    root = teacher_routes.paths_mod.synth_jobs_root()
    d = os.path.join(root, "job1")
    os.makedirs(d)
    teacher_routes._write_job_meta(d, [], [], "z" * 500)
    assert len(teacher_routes._read_job_meta(d)["label"]) == teacher_routes.JOB_LABEL_MAX
