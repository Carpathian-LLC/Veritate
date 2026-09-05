# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers tools/build_fact_chats.py --recall: conversations in which a fact is told,
#   small talk follows, and the user asks for it back with the assistant answering from
#   the conversation. The skill this trains is "use what was just said"; the probe in
#   lab 2026-09-03-working-memory-from-carried-state found the abstention SFT refusing
#   exactly these questions with the fact in the window.
# tests/corpus/test_build_fact_chats_recall.py
# ------------------------------------------------------------------------------------
# Imports:
# ------------------------------------------------------------------------------------
import json
import random

from build_fact_chats import IM_E, IM_S, build, render_recall_conversation

# ------------------------------------------------------------------------------------
# Constants:
# ------------------------------------------------------------------------------------
FACT_LIVES = {"subj": "Anselm", "obj": "Quillhaven", "kind": "lives"}
FACT_JOB = {"subj": "Petra", "obj": "engraver", "kind": "job"}


# ------------------------------------------------------------------------------------
# Functions:
# ------------------------------------------------------------------------------------
def _turns(text):
    parts = [p for p in text.split(IM_S) if p]
    return [(parts[i].split("\n", 1)[1].rstrip(IM_E + "\n"), parts[i + 1].split("\n", 1)[1].rstrip(IM_E + "\n"))
            for i in range(0, len(parts), 2)]


def test_recall_conversation_ends_with_the_fact_asked_back_and_answered_from_context():
    """The last turn is a question about the fact and the assistant's reply names the object."""
    text = render_recall_conversation(FACT_LIVES, random.Random(3), first_person=False)
    turns = _turns(text)
    assert "Quillhaven" in turns[1][0] or "Quillhaven" in turns[1][1]
    assert turns[-1][0].endswith("?")
    assert "Quillhaven" in turns[-1][1] or "Anselm" in turns[-1][1]
    assert "don't know" not in text


def test_small_talk_separates_the_telling_from_the_asking():
    """At least one distractor turn sits between the fact and the question, so the answer is
    recall from context and not an echo of the previous turn."""
    text = render_recall_conversation(FACT_JOB, random.Random(5), first_person=False)
    turns = _turns(text)
    assert len(turns) >= 4
    assert "engraver" not in turns[-2][0] and "engraver" not in turns[-2][1]


def test_first_person_form_speaks_as_the_user():
    """Half the corpus is first person: the user says 'I ...' and asks 'Where do I ...'."""
    text = render_recall_conversation(FACT_LIVES, random.Random(1), first_person=True)
    turns = _turns(text)
    assert turns[1][0].startswith("I")
    assert "I" in turns[-1][0] and turns[-1][0].endswith("?")
    assert "You" in turns[-1][1] and "Quillhaven" in turns[-1][1]
    assert "Anselm" not in text


def test_gap_bytes_pushes_the_question_past_the_window():
    """With gap_bytes the question sits at least that many bytes after the telling, so under a
    shorter training window only the carried state can connect them; without it the two are close."""
    far = render_recall_conversation(FACT_LIVES, random.Random(7), first_person=True, gap_bytes=2200)
    near = render_recall_conversation(FACT_LIVES, random.Random(7), first_person=True)
    tell_far, ask_far = far.encode().find(b"Quillhaven"), far.encode().rfind(b"?")
    tell_near, ask_near = near.encode().find(b"Quillhaven"), near.encode().rfind(b"?")
    assert ask_far - tell_far >= 2200
    assert ask_near - tell_near < 800
    assert "Quillhaven" in far.split("?")[-1]
    assert len(set(far.split(IM_S))) == len(far.split(IM_S))


def test_recall_flag_renders_recall_conversations_into_the_bins(tmp_path):
    """--recall swaps the renderer for the whole build; the default build is untouched."""
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps([FACT_LIVES, FACT_JOB]))
    nf, ne, _tb, _vb = build(str(facts), stem="rc", per_fact=4, seed=0, out_dir=str(tmp_path), recall=True)
    assert (nf, ne) == (2, 8)
    body = (tmp_path / "rc_train.bin").read_bytes().decode() + (tmp_path / "rc_val.bin").read_bytes().decode()
    assert "You told me" in body or "You said" in body or "you told me" in body
    build(str(facts), stem="plain", per_fact=4, seed=0, out_dir=str(tmp_path))
    plain = (tmp_path / "plain_train.bin").read_bytes().decode()
    assert "Remind me" not in plain
