# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers tools/extract_facts.py, the tell-it-once extraction pre-pass (IDEA 20
#   E4; failures.md 2026-08-21 m2). The acceptance test is the closed loop: the
#   m2 corpus generator renders the 50 e4 gold facts into 1,000 natural
#   conversations with distractor turns, those are replayed as experience-log
#   records, and extraction must recover the gold set at >=90% precision (hard)
#   and >=98% recall (measured 100% at authoring). Distractor small talk must
#   extract nothing. Units pin negation/question/hedge rejection, the copula
#   occupation guard, dedupe, revision (newest object wins, revised flag), and
#   the bin-filename model matching. Also covers build_experience_corpus
#   --facts mode (build_fact_bins).
# tests/mri/test_extract_facts.py
# ------------------------------------------------------------------------------------
# Imports:

import base64
import json
import os
import random
import re

import pytest
from conftest import MRI_ROOT
from tools import build_experience_corpus as bec
from tools import build_fact_chats as bfc
from tools import extract_facts as ef

# ------------------------------------------------------------------------------------
# Constants

E4_FACTS = os.path.join(MRI_ROOT, "data", "eval", "e4_facts.json")
PAIR_RX = re.compile(
    r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n(.*?)<\|im_end\|>\n", re.S)

# ------------------------------------------------------------------------------------
# Functions


def _b64(s):
    return base64.b64encode(s.encode()).decode()


def _rec(prompt, output, ts=0, model="veritate.bin"):
    return {"ts": ts, "model": model, "prompt_b64": _b64(prompt), "output_b64": _b64(output)}


def _conv_records(conv, ts0):
    """Replay one rendered conversation as the experience log would record it:
    one record per served turn, prompt carrying the growing ChatML history."""
    recs, hist, ts = [], "", ts0
    for u, a in PAIR_RX.findall(conv):
        ts += 1
        prompt = hist + f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
        recs.append(_rec(prompt, a + "<|im_end|>\n", ts=ts))
        hist = prompt + a + "<|im_end|>\n"
    return recs


def _key(fact):
    return (fact["kind"], fact["subj"].lower(), fact["obj"].lower())


def test_closed_loop_recovers_e4_facts(tmp_path, monkeypatch):
    """1,000 m2 conversations over the 50 gold facts, replayed through the log,
    extract at >=90% precision and >=98% recall."""
    with open(E4_FACTS) as f:
        gold_facts = json.load(f)
    rng = random.Random(0)
    records, ts = [], 0
    for fact in gold_facts:
        for i in range(20):
            conv = bfc.render_conversation(fact, rng, reverse=(i % 3 == 2))
            recs = _conv_records(conv, ts)
            records.extend(recs)
            ts += len(recs)
    log = tmp_path / "experience"
    log.mkdir()
    (log / "20260821.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    monkeypatch.setattr(ef, "EXPERIENCE_ROOT", str(log))
    facts, _rej = ef.extract(ef.load_records())
    gold = {_key(f) for f in gold_facts}
    got = {_key(f) for f in facts}
    precision = len(got & gold) / len(got)
    recall = len(got & gold) / len(gold)
    assert precision >= 0.90, f"precision {precision:.3f}, false facts: {sorted(got - gold)}"
    assert recall >= 0.98, f"recall {recall:.3f}, missed: {sorted(gold - got)}"
    assert not any(f.get("revised") for f in facts)          # no fact was restated with a new object


def test_distractor_turns_extract_nothing():
    """Conversations built only from the m2 distractor pool yield zero facts."""
    records, ts = [], 0
    for i in range(len(bfc.DISTRACTORS)):
        pairs = [bfc.DISTRACTORS[(i + j) % len(bfc.DISTRACTORS)] for j in range(3)]
        conv = "".join(f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
                       for u, a in pairs)
        records.extend(_conv_records(conv, ts))
        ts += 3
    facts, _rej = ef.extract(records)
    assert facts == []


def test_negation_and_pasttense_rejected():
    """Negated, no-longer, and used-to statements yield no facts."""
    for s in ("Talia doesn't live in Feldbrook.",
              "Talia no longer lives in Feldbrook.",
              "Bram never moved to Ostwick.",
              "Marek is not a beekeeper by trade.",
              "Marek used to work as a beekeeper."):
        cands, _rej = ef.scan_text(s)
        assert cands == [], s


def test_uncertainty_and_hypotheticals_rejected():
    """Hedged, hearsay, and hypothetical statements yield no facts."""
    for s in ("I think Odile lives in Marleth.",
              "Odile probably lives in Marleth.",
              "Suvi might be a glassblower.",
              "If Talia lives in Feldbrook, send a letter.",
              "Talia plans to move to Ostwick.",
              "Maybe Kellan works as a farrier."):
        cands, _rej = ef.scan_text(s)
        assert cands == [], s


def test_questions_rejected_but_rhetorical_tell_kept():
    """Questions never extract; the 'Did I mention X is a Y?' tell does."""
    for s in ("Where does Talia live?", "Who lives in Feldbrook?",
              "Does Marek work as a beekeeper?", "Who was it that works as a beekeeper?"):
        cands, _rej = ef.scan_text(s)
        assert cands == [], s
    cands, _rej = ef.scan_text("Did I mention Marek is a beekeeper these days?")
    assert [(c["kind"], c["subj"], c["obj"]) for c in cands] == [("job", "Marek", "beekeeper")]


def test_copula_guard():
    """Bare 'X is a Y' extracts only when Y passes the occupation lexicon."""
    cands, _rej = ef.scan_text("Vera is an archivist.")
    assert [(c["kind"], c["subj"], c["obj"]) for c in cands] == [("job", "Vera", "archivist")]
    for s in ("Talia is a wonder.", "Sam is a member now.",
              "Petra is a vegetarian these days.", "Sam is a woman."):
        cands, _rej = ef.scan_text(s)
        assert cands == [], s


def test_dedupe_repeated_statements():
    """The same fact stated three ways collapses to one unrevised fact."""
    records = [_rec("My friend Talia lives in Feldbrook these days.", "Noted, thanks.", ts=1),
               _rec("Talia has a place in Feldbrook now.", "I'll remember that: Talia lives in Feldbrook.", ts=2)]
    facts, _rej = ef.extract(records)
    assert len(facts) == 1
    assert _key(facts[0]) == ("lives", "talia", "feldbrook")
    assert "revised" not in facts[0] and facts[0]["seen"] >= 3
    assert set(facts[0]) >= {"stmt", "subj", "obj", "q_fwd", "a_fwd", "q_rev", "a_rev", "kind"}


def test_revision_keeps_newest_object():
    """Same subject+relation with a different object: newest ts wins, revised set."""
    records = [_rec("Talia moved to Ostwick.", "Got it.", ts=200),
               _rec("Talia lives in Feldbrook.", "Noted.", ts=100)]   # older, out of order
    facts, _rej = ef.extract(records)
    assert len(facts) == 1
    assert facts[0]["obj"] == "Ostwick" and facts[0]["revised"] is True


def test_assistant_restatement_extracts():
    """A fact present only in the assistant's reply is still captured."""
    records = [_rec("<|im_start|>user\nTell me about Marek.<|im_end|>\n<|im_start|>assistant\n",
                    "Marek works as a beekeeper.<|im_end|>\n", ts=1)]
    facts, _rej = ef.extract(records)
    assert [_key(f) for f in facts] == [("job", "marek", "beekeeper")]


def test_model_field_matching():
    """--model matches bin filenames and dir names either way round."""
    assert ef.model_match("veritate.bin", "veritate")
    assert ef.model_match("veritate", "veritate.bin")
    assert ef.model_match("wren2_0.bin", "wren2")
    assert not ef.model_match("veritate.bin", "wren2")
    records = [_rec("Talia lives in Feldbrook.", "Noted.", model="veritate.bin"),
               _rec("Marek works as a beekeeper.", "Noted.", model="other.bin")]
    facts, _rej = ef.extract(records, model="veritate")
    assert [_key(f) for f in facts] == [("lives", "talia", "feldbrook")]


def test_build_fact_bins_renders_extracted_facts(tmp_path, monkeypatch):
    """build_experience_corpus --facts mode: extraction feeds build_fact_sft and
    emits fact-SFT bins plus the audit facts JSON; an empty log emits nothing."""
    log = tmp_path / "experience"
    log.mkdir()
    out = tmp_path / "corpus"
    monkeypatch.setattr(ef, "EXPERIENCE_ROOT", str(log))
    nf, tb, vb = bec.build_fact_bins(out_dir=str(out))
    assert (nf, tb, vb) == (0, 0, 0) and not out.exists()
    recs = [_rec("Talia lives in Feldbrook.", "Noted: Talia lives in Feldbrook.", ts=1),
            _rec("Marek works as a beekeeper.", "So Marek works as a beekeeper now.", ts=2)]
    (log / "20260821.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    nf, tb, vb = bec.build_fact_bins(out_dir=str(out))
    assert nf == 2 and tb > 0 and vb > 0
    saved = json.loads((out / "experience_facts.json").read_text())
    assert {_key(f) for f in saved} == {("lives", "talia", "feldbrook"), ("job", "marek", "beekeeper")}
    data = (out / "experience_fact_sft_train.bin").read_bytes() + (out / "experience_fact_sft_val.bin").read_bytes()
    assert b"Talia lives in Feldbrook" in data and b"beekeeper" in data


@pytest.fixture(autouse=True)
def _no_real_log(monkeypatch):
    """Extraction in these tests must never read the install's real experience log."""
    monkeypatch.setattr(ef, "EXPERIENCE_ROOT", os.path.join(os.sep, "nonexistent", "experience"))


def test_self_facts_extract_from_user_turns():
    """First-person statements about the person become {kind: self} facts."""
    for sent, attr, obj in (("my sister's name is Wren.", "sister's name", "Wren"),
                            ("My name is Sam.", "name", "Sam"),
                            ("my timezone is Pacific.", "timezone", "Pacific"),
                            ("I live in Portland.", "home", "Portland"),
                            ("I work at Carpathian.", "workplace", "Carpathian"),
                            ("I prefer short answers.", "preference", "short answers")):
        cands, _rej = ef.scan_text(sent, role="user")
        assert [(c["kind"], c["subj"], c["obj"]) for c in cands] == [("self", attr, obj)], sent


def test_self_facts_only_mined_from_user_turns():
    """The assistant saying "my name is ..." is the model talking about itself."""
    assert ef.scan_text("My name is Sam.", role="assistant")[0] == []
    assert ef.scan_text("My name is Sam.")[0] == []


def test_self_intentions_are_not_facts():
    """Attributes naming a transient intention never become durable memory."""
    for s in ("my task is to fix the build.", "my goal is to ship tonight.",
              "my plan is to rewrite it.", "my question is about sleep."):
        cands, _rej = ef.scan_text(s, role="user")
        assert cands == [], s


def test_self_questions_hedges_and_negations_rejected():
    """A trailing question mark sits outside the match, so it is read from the source."""
    for s in ("my name is Sam?", "my name is not Sam.", "I might live in Portland.",
              "I think my timezone is Pacific."):
        cands, _rej = ef.scan_text(s, role="user")
        assert cands == [], s


def test_self_fact_renders_both_directions():
    """make_fact emits forward and reverse question forms for a self-fact."""
    fact = ef.make_fact("self", "sister's name", "Wren")
    assert fact["stmt"] == "Your sister's name is Wren."
    assert fact["q_fwd"] == "What is my sister's name?" and fact["a_fwd"] == "Wren"
    assert fact["a_rev"] == "your sister's name"
