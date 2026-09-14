# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the in-pretrain "missing skills" slice. Its design constraints are the measured ones:
#   answers are literal spans of real source text, never a closed hand-written vocabulary
#   (a combinatorial generator scored val 0.092 while teaching nothing, failures.md
#   role-binding 2026-07-25); no single question template may dominate; the frame must
#   equal the serving prompt surface; and the build must be deterministic so a rebuilt
#   corpus has the same sha256. Those are what this pins, over a tiny synthetic source.
# tests/corpus/test_build_skills_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import random
from collections import Counter, deque

import build_skills_corpus as bsc

# ------------------------------------------------------------------------------------
# Constants

# real-looking prose: several sentences, each long enough to survive the sentence filter,
# carrying spans the extractor can find (a year, a quoted phrase, a capitalised name).
SENTENCES = [
    "The harbour office at Larkfell opened in 1897 and still keeps the tide ledgers.",
    "Keepers recorded every lamp lighting in a bound book kept behind the counter.",
    "A storm in November tore the roof from the eastern shed and flooded the yard.",
    "The council paid 240 pounds for repairs and reopened the office that spring.",
    "Visitors may read the ledgers on request between April and September each year.",
]
PASSAGE = " ".join(SENTENCES)

# ------------------------------------------------------------------------------------
# Functions


def _source(tmp_path, copies=400):
    p = tmp_path / "src_train.bin"
    p.write_bytes(bsc.EOT_BYTES.join(PASSAGE.encode() for _ in range(copies)))
    return str(p)


def _build(tmp_path, target_mb=0.05, val_ratio=0.0, seed=7, copies=400):
    train, val = tmp_path / "s_train.bin", tmp_path / "s_val.bin"
    stats = bsc.build(_source(tmp_path, copies), str(train), str(val),
                      target_mb=target_mb, val_ratio=val_ratio, seed=seed)
    return train, val, stats


def test_records_split_on_the_separator(tmp_path):
    p = tmp_path / "c.bin"
    p.write_bytes(bsc.EOT_BYTES.join([b"one", b"two"]))
    assert list(bsc._iter_records(str(p))) == ["one", "two"]


def test_a_record_carrying_chat_markup_is_dropped():
    """The source is prose. A record that already contains ChatML would frame a turn
    inside a turn."""
    assert bsc._is_clean(PASSAGE)
    assert not bsc._is_clean(PASSAGE + bsc.IM_START)


def test_passages_are_whole_sentences_within_the_size_band():
    rng = random.Random(1)
    out = list(bsc._passages_from_record(rng, PASSAGE))
    assert out
    for p in out:
        assert bsc.PASSAGE_MIN_CHARS <= len(p) <= bsc.PASSAGE_MAX_CHARS + len(SENTENCES[0])
        assert p[0].isupper()


def test_a_span_is_a_literal_substring_of_its_passage():
    """The anti-template-collapse rule: the answer is real text, so the answer set stays
    open."""
    for span, _kind, idx in bsc._unique_spans(PASSAGE):
        assert PASSAGE[idx:idx + len(span)] == span


def test_the_frame_is_the_serving_prompt_surface():
    """training surface == inference surface (hybrid_routes prompt template)."""
    framed = bsc._frame("ctx", "q", "a")
    assert framed.startswith(bsc.CONTEXT_PREFIX + "ctx\n")
    assert f"{bsc.IM_START}user\nq{bsc.IM_END}" in framed
    assert framed.endswith(f"{bsc.IM_START}assistant\na{bsc.IM_END}\n{bsc.EOT}\n")


def test_the_neediest_family_is_the_one_furthest_behind_its_share():
    """One attempt per passage, always the family behind quota, so a family with a high
    rejection rate does not quietly shrink the others."""
    families = Counter({k: int(100 * v) for k, v in bsc.FAMILY_SHARES.items()})
    families["honest_miss"] -= 10
    assert bsc._neediest_family(families, 100) == "honest_miss"


def test_no_question_template_runs_away_with_the_corpus():
    """A template over its allowance is passed over while another is eligible: that cap is
    what keeps question phrasing from becoming the thing the model learns."""
    over, under = "g_which", "g_direct"
    bank = {k: bsc.GROUNDED_QUESTIONS[k] for k in (over, under)}
    counts = Counter({over: 100, under: 0})
    rng = random.Random(0)
    picks = {bsc._pick_template(rng, bank, {"kind": "year"}, counts, 200) for _ in range(20)}
    assert picks == {under}


def test_a_template_needing_a_field_the_passage_lacks_is_not_picked():
    """g_which needs a `kind`; with none resolved there is nothing to ask."""
    bank = {"g_which": bsc.GROUNDED_QUESTIONS["g_which"]}
    assert bsc._pick_template(random.Random(0), bank, {"kind": ""}, Counter(), 10) is None


def test_the_novel_string_is_absent_from_the_passage_it_is_planted_in():
    """Copying is only tested if the string cannot be produced from the passage's own
    statistics."""
    rng = random.Random(3)
    made = bsc.novel_string_example(rng, PASSAGE, deque(), Counter(), 10)
    assert made is not None
    _user, assistant, context, _key, _pos = made
    surface = assistant.strip().rstrip(".").split()[-1]
    assert surface in context and surface not in PASSAGE


def test_an_honest_miss_answers_that_the_context_does_not_say():
    """The fourth skill: refuse rather than invent when the answer is not there."""
    rng = random.Random(5)
    made = bsc.honest_miss_example(rng, PASSAGE, deque(), Counter(), 10)
    assert made is not None
    _user, assistant, context, _key, _pos = made
    assert context == PASSAGE
    assert assistant


def test_the_build_writes_records_and_reports_what_it_made(tmp_path):
    train, _val, stats = _build(tmp_path)
    body = train.read_bytes()
    assert body.count(bsc.EOT.encode()) == sum(stats["families"].values())
    assert body.startswith(bsc.CONTEXT_PREFIX.encode())
    assert set(stats["families"]) <= set(bsc.FAMILY_SHARES)


def _sub(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return d


def test_the_same_seed_rebuilds_the_same_bytes(tmp_path):
    """Deterministic by design, so a corpus can be identified by its sha256."""
    a = _build(_sub(tmp_path, "a"), seed=11)
    b = _build(_sub(tmp_path, "b"), seed=11)
    assert a[0].read_bytes() == b[0].read_bytes()
    assert a[2]["sha256"][str(a[0])] == b[2]["sha256"][str(b[0])]


def test_a_different_seed_gives_a_different_corpus(tmp_path):
    a = _build(_sub(tmp_path, "a"), seed=11)
    b = _build(_sub(tmp_path, "b"), seed=12)
    assert a[0].read_bytes() != b[0].read_bytes()


def test_the_val_ratio_decides_where_a_record_lands(tmp_path):
    """Every record goes to exactly one file; the ratio is the only thing choosing."""
    train_only, val_of_train_only, _s = _build(_sub(tmp_path, "t"), val_ratio=0.0)
    assert train_only.stat().st_size > 0 and val_of_train_only.stat().st_size == 0
    train_of_val_only, val_only, _s2 = _build(_sub(tmp_path, "v"), val_ratio=1.0)
    assert val_only.stat().st_size > 0 and train_of_val_only.stat().st_size == 0


def test_an_exhausted_source_stops_short_and_says_so(tmp_path, capsys):
    """Silence would leave a short corpus looking like a finished one."""
    train, val, stats = _build(tmp_path, target_mb=50, copies=5)
    bsc._report(stats, str(train), str(val))
    assert "SOURCE EXHAUSTED" in capsys.readouterr().out
