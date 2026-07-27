# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Guards the corpus authoring gates in veritate_mri/teacher/authoring.py: schema,
#   em dash, word-boundary ban list, near dup, repetition floor, determinism.
# - The audit that motivated these gates found chat_5gb at 1.4% distinct 5-grams and
#   10,196 em dashes in synthetic_v1; every gate here has a corresponding defect.
# tests/mri/test_corpus_authoring.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "veritate_mri")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from teacher import authoring

# ------------------------------------------------------------------------------------
# Constants

GENRE = "conversation"
PROMPT = {"id": "auth_conversation_000000", "genre": GENRE}
LONG_USER = "tell me about the ferry timetable on the north side of the lake please"
LONG_REPLY = ("the first boat leaves at six and the last one at half nine, "
              "but the winter timetable drops the midday run entirely")
DISTINCT_THREE = [
    ("where can i get a decent loaf of bread around here on a sunday",
     "the bakery by the tannery opens late, everything else shuts after the market",
     "gossipy neighbor"),
    ("my clutch is slipping on hills, is that expensive to sort out",
     "depends whether the plate is glazed or the cable has stretched, one is cheap and one is not",
     "sardonic bike mechanic"),
    ("i keep putting off writing to my sister and now it feels too late",
     "it is never too late, it is just heavier, write two lines and post them today",
     "gentle hospice volunteer"),
]

# ------------------------------------------------------------------------------------
# Functions


def _spec():
    """Shipped spec with this genre's depth floor pinned, so tuning min_turns for
    corpus quality cannot break tests that are about gate behaviour."""
    spec = authoring.load_spec()
    for g in spec["genres"]:
        if g["id"] == GENRE:
            g["min_turns"] = 2
    return spec


def _turns_record(user=LONG_USER, reply=LONG_REPLY, voice="dry retired machinist"):
    return {"genre": GENRE, "voice": voice,
            "turns": [{"role": "user", "text": user},
                      {"role": "assistant", "text": reply}]}


def _response(*records):
    return "\n".join(json.dumps(r) for r in records)


def test_valid_record_is_accepted():
    """A schema-clean conversation record passes every gate."""
    gate = authoring.RecordGate(_spec())
    kept, why = gate(PROMPT, _response(_turns_record()))
    assert len(kept) == 1 and why == []


def test_em_dash_is_rewritten_out_under_default_policy():
    """The shipped rewrite policy strips em dashes from an accepted record."""
    gate = authoring.RecordGate(_spec())
    kept, _ = gate(PROMPT, _response(_turns_record(reply="the boat leaves at six — maybe seven, nobody knows for sure")))
    assert "—" not in authoring.record_text(kept[0], authoring.SCHEMA_TURNS)


def test_em_dash_is_rejected_under_reject_policy():
    """Switching em_dash_policy to reject drops the record instead of editing it."""
    spec = _spec()
    spec["gates"]["em_dash_policy"] = authoring.EM_DASH_REJECT
    gate = authoring.RecordGate(spec)
    kept, why = gate(PROMPT, _response(_turns_record(reply="the boat leaves at six — maybe seven, nobody knows")))
    assert kept == [] and why == [authoring.REJECT_EM_DASH]


def test_banned_phrase_in_assistant_turn_is_rejected():
    """A whole banned phrase in the assistant turn drops the record."""
    gate = authoring.RecordGate(_spec())
    kept, why = gate(PROMPT, _response(_turns_record(
        reply="great question, the first boat leaves at six and the last at half nine")))
    assert kept == [] and why == [authoring.REJECT_BANNED]


def test_banned_phrase_does_not_false_positive_inside_a_longer_word():
    """'as an ai' must not fire inside 'as an airport', which merge_dedup's substring scan hit."""
    gate = authoring.RecordGate(_spec())
    kept, why = gate(PROMPT, _response(_turns_record(
        reply="it started as an airport shuttle and ended up as an airline of sorts")))
    assert len(kept) == 1 and why == []


def test_banned_phrase_in_user_turn_is_kept():
    """Banned phrases are assistant-side tells; a user saying one is natural speech."""
    gate = authoring.RecordGate(_spec())
    kept, why = gate(PROMPT, _response(_turns_record(
        user="great question, is that what you always say when you stall for time")))
    assert len(kept) == 1 and why == []


def test_exact_duplicate_is_rejected():
    """The same record twice in a batch keeps one copy."""
    gate = authoring.RecordGate(_spec())
    kept, why = gate(PROMPT, _response(_turns_record(), _turns_record()))
    assert len(kept) == 1 and why == [authoring.REJECT_EXACT_DUP]


def test_near_duplicate_is_rejected():
    """A record differing by one token is caught by the simhash gate, not just exact match."""
    gate = authoring.RecordGate(_spec())
    kept, why = gate(PROMPT, _response(_turns_record(),
                                       _turns_record(reply=LONG_REPLY + " today")))
    assert len(kept) == 1 and why == [authoring.REJECT_NEAR_DUP]


def test_schema_mismatch_on_extra_key_is_rejected():
    """An extra top-level key is a schema violation, matching merge_dedup's strict key set."""
    gate = authoring.RecordGate(_spec())
    rec = _turns_record()
    rec["extra"] = 1
    kept, why = gate(PROMPT, _response(rec))
    assert kept == [] and why == [authoring.REJECT_SCHEMA]


def test_schema_mismatch_on_wrong_role_order_is_rejected():
    """Roles must alternate starting with user."""
    gate = authoring.RecordGate(_spec())
    rec = _turns_record()
    rec["turns"][0]["role"] = "assistant"
    kept, why = gate(PROMPT, _response(rec))
    assert kept == [] and why == [authoring.REJECT_SCHEMA]


def test_grounded_read_requires_the_context_marker():
    """The grounded_read genre rejects a first user turn with no 'context:' block."""
    spec = _spec()
    genre = authoring.genre_by_id(spec, "grounded_read")
    gate = authoring.RecordGate(spec)
    rec = {"genre": "grounded_read", "voice": "field notebook",
           "turns": [{"role": "user", "text": "what colour was the boat in the passage above"},
                     {"role": "assistant", "text": "the passage does not say what colour the boat was"}]}
    kept, why = gate({"id": "x", "genre": "grounded_read"}, _response(rec))
    assert genre["require_in_first_user"] == "context:"
    assert kept == [] and why == [authoring.REJECT_MISSING_MARKER]


def test_carryover_genre_rejects_a_short_conversation():
    """The carry-over genre enforces its min_turns floor so two-turn filler cannot pass."""
    spec = _spec()
    gate = authoring.RecordGate(spec)
    rec = {"genre": "carryover", "voice": "project manager tracking a deadline",
           "turns": [{"role": "user", "text": LONG_USER},
                     {"role": "assistant", "text": LONG_REPLY}]}
    kept, why = gate({"id": "x", "genre": "carryover"}, _response(rec))
    assert kept == [] and why == [authoring.REJECT_TURN_COUNT]


def test_repetition_floor_warns_on_template_generated_text():
    """chat_5gb-style template expansion drives the distinct-5-gram ratio under the floor."""
    spec = _spec()
    spec["gates"]["ngram_recompute_every"] = 1
    spec["gates"]["near_dup_hamming"] = -1
    spec["gates"]["opening_cap"] = 10 ** 6
    gate = authoring.RecordGate(spec)
    for i in range(40):
        gate(PROMPT, _response(_turns_record(
            reply=f"the boat leaves at six and the last one at half nine on day {i}")))
    assert gate.stats()["ngram_below_floor"] is True


def test_repetition_floor_clear_on_varied_text():
    """Genuinely varied records stay above the floor, so the warning is not always on."""
    spec = _spec()
    spec["gates"]["ngram_recompute_every"] = 1
    words = open(os.path.join(REPO_ROOT, "veritate_mri", "teacher", "authoring.py"),
                 encoding="utf-8").read().split()
    gate = authoring.RecordGate(spec)
    for i in range(30):
        chunk = " ".join(words[i * 40:(i + 1) * 40])
        gate(PROMPT, _response(_turns_record(user=chunk, reply=chunk[::-1])))
    assert gate.stats()["ngram_below_floor"] is False


def test_prompt_build_is_deterministic_under_a_fixed_seed():
    """Same spec plus same seed gives byte-identical prompts."""
    spec = _spec()
    a = authoring.build_prompts(spec, {GENRE: 3}, 1234, "auth_")
    b = authoring.build_prompts(spec, {GENRE: 3}, 1234, "auth_")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_prompt_build_varies_with_the_seed():
    """A different seed rotates the voice pool, so repeated runs are not identical."""
    spec = _spec()
    a = authoring.build_prompts(spec, {GENRE: 3}, 1234, "auth_")
    b = authoring.build_prompts(spec, {GENRE: 3}, 9999, "auth_")
    assert json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True)


def test_simhash_is_stable_across_processes():
    """simhash64 must not depend on the per-process string hash salt."""
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); "
            "from teacher.quality import simhash64; print(simhash64('the boat leaves at six'))"
            % os.path.join(REPO_ROOT, "veritate_mri"))
    env = dict(os.environ, PYTHONHASHSEED="1")
    one = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    env["PYTHONHASHSEED"] = "2"
    two = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert one.stdout.strip() == two.stdout.strip() != ""


def test_plan_calls_splits_a_byte_target_across_genres():
    """A megabyte target turns into a per-genre call count weighted by the spec."""
    spec = _spec()
    calls = authoring.plan_calls(spec, [GENRE, "jokes"], 10 * 1024 * 1024)
    assert set(calls) == {GENRE, "jokes"} and calls[GENRE] > calls["jokes"]


def test_synth_job_expands_one_response_into_gated_records(tmp_path):
    """A batch reply becomes N records on disk, with gate stats in state.json."""
    from teacher import synth

    spec = _spec()
    prompts = authoring.build_prompts(spec, {GENRE: 1}, 7, "auth_")
    batch = _response(*[_turns_record(user=u, reply=r, voice=v) for u, r, v in DISTINCT_THREE])

    class _Stub:
        def complete(self, messages, **kw):
            return batch

    job = synth.SynthJob("t1", "test", "m", prompts, str(tmp_path),
                         client_factory=_Stub, record_gate=authoring.RecordGate(spec))
    out = job.run()
    lines = (tmp_path / "samples.jsonl").read_text(encoding="utf-8").strip().splitlines()
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert out["authoring"]["records"] == 3
    assert len(lines) == 3
    assert json.loads(lines[0])["id"] == prompts[0]["id"] + "#0"
    assert state["authoring"]["bytes"] > 0


def test_build_packs_zips_and_registers_a_coming_soon_entry(tmp_path, monkeypatch):
    """The handoff artefact is a top-level zip of both bins plus a placeholder catalog entry."""
    import zipfile

    from routes import teacher_routes as tr

    spec = _spec()
    samples = tmp_path / "samples.jsonl"
    with open(samples, "w", encoding="utf-8") as f:
        for i in range(8):
            f.write(json.dumps({"id": f"auth_conversation_000000#{i}",
                                "record": _turns_record(reply=f"{LONG_REPLY} number {i}")}) + "\n")
            f.write(json.dumps({"id": f"auth_jokes_000000#{i}",
                                "record": {"genre": "jokes", "voice": "deadpan",
                                           "text": f"my rowing machine finally admitted it was a chair, number {i}"}}) + "\n")
    fam_dir, counts = tr._write_families(str(tmp_path), spec)
    manifest = tr.build_sft_bins("authored_probe", fam_dir, str(tmp_path / "dist"),
                                 sorted(f"{g}.jsonl" for g in counts),
                                 tr.AUTHORING_PURPOSE, tr.AUTHORING_LICENSE, 1, 0.25)
    zip_path = tr._zip_bins(str(tmp_path / "dist"), "authored_probe")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"corpora": []}), encoding="utf-8")
    monkeypatch.setattr(tr, "LOCAL_CATALOG_PATH", str(catalog))
    entry = tr._register_catalog_entry("authored_probe", "Probe", "probe corpus", manifest, 0, None)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    assert sorted(names) == ["authored_probe_train.bin", "authored_probe_val.bin"]
    assert counts == {GENRE: 8, "jokes": 8}
    assert entry["coming_soon"] is True and "PLACEHOLDER" in entry["train_url"]
    assert entry["sha256_train"] == manifest["train_sha256"]
    assert json.loads(catalog.read_text(encoding="utf-8"))["corpora"][0]["stem"] == "authored_probe"


def test_resume_seeds_dedup_state_from_disk(tmp_path):
    """Restarting a job rejects records already written, so resume cannot duplicate."""
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps({"id": "auth_conversation_000000#0",
                                   "record": _turns_record()}) + "\n", encoding="utf-8")
    gate = authoring.RecordGate(_spec())
    gate.seed_from_file(str(samples))
    kept, why = gate(PROMPT, _response(_turns_record()))
    assert kept == [] and why == [authoring.REJECT_EXACT_DUP]


def test_genre_is_stamped_from_the_call_not_trusted_from_the_model():
    """A model-invented genre label is overwritten by the call's genre, not rejected."""
    gate = authoring.RecordGate(_spec())
    rec = _turns_record()
    rec["genre"] = "casual conversation"
    kept, why = gate(PROMPT, _response(rec))
    assert why == [] and kept[0]["genre"] == GENRE
