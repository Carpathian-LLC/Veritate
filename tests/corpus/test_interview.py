# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Tests for teacher/interview.py, the two-pass generator that replaced
#   dialogue-scripting (failures.md 2026-08-20: scripting caps assistant turns at
#   ~120 B median; asking the same model the same question yields 2,380 B).
# - The load-bearing tests are the trim ones. Cutting mid-sentence would train the
#   model to stop mid-thought, which is precisely the defect the 2026-08-10 corpus
#   rebuild removed -- so trim_to_sentence must overshoot its ceiling rather than
#   ever emit a fragment.
# - The teacher is a stub; nothing hits a network (rule 48).
# tests/corpus/test_interview.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
from teacher import interview

# ------------------------------------------------------------------------------------
# Constants

THREE = "First sentence here. Second sentence here. Third sentence here."

# ------------------------------------------------------------------------------------
# Functions

class StubClient:
    """Teacher stand-in. `replies` is consumed in order; `asks` records every call
    so a test can assert on WHAT was asked, not just what came back."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.asks = []

    def complete(self, messages, temperature=None, max_tokens=None, system=None,
                 cancel_check=None):
        self.asks.append({"messages": messages, "system": system})
        return self.replies.pop(0) if self.replies else ""


def _long(sentences, word="word"):
    return " ".join(f"{word} " * 8 + "end." for _ in range(sentences))


def test_registers_are_a_probability_distribution():
    """The blend has to sum to 1 or the draw silently biases to the last one."""
    assert sum(r["weight"] for r in interview.REGISTERS) == pytest.approx(1.0)


def test_the_blend_produces_both_short_and_long_registers():
    """The user asked for a blend; a draw that only ever returns one is a bug."""
    import random
    rng = random.Random(0)
    drawn = {interview.pick_register(rng)["id"] for _ in range(400)}
    assert {"brief", "normal", "thorough"} <= drawn


def test_trim_keeps_whole_sentences():
    """Cut on a sentence boundary, never inside one. Two sentences is 41 bytes,
    three is 62, so a 50-byte ceiling must keep exactly two."""
    out = interview.trim_to_sentence(THREE, 50)
    assert out == "First sentence here. Second sentence here."
    assert interview.trim_to_sentence(THREE, 40) == "First sentence here."


def test_trim_never_emits_a_fragment_even_under_its_ceiling():
    """If the first sentence alone exceeds the ceiling, keep it whole and go over.
    A fragment is a worse training example than an over-long sentence."""
    text = "A single very long sentence that runs well past the ceiling given here."
    out = interview.trim_to_sentence(text, 10)
    assert out == text
    assert not out.endswith("that")


def test_trim_leaves_short_text_alone():
    """Nothing to do under the ceiling; no accidental reformatting."""
    assert interview.trim_to_sentence("Short.", 500) == "Short."


def test_trim_handles_text_with_no_sentence_end():
    """Unpunctuated output must pass through rather than vanish."""
    assert interview.trim_to_sentence("no punctuation here", 5) == "no punctuation here"


def test_disclaimers_are_removed_whole():
    """The gate counts these as artifacts, so they are cut before they are counted."""
    out = interview.strip_disclaimers(
        "As an AI language model, I cannot be sure. Gardens need sun and water.")
    assert "language model" not in out
    assert "Gardens need sun and water." in out


def test_list_scaffolding_becomes_prose():
    """Asked-for conversation should not arrive formatted like a manual."""
    out = interview.strip_structure("Here you go:\n- first thing\n- second thing")
    assert "-" not in out
    assert "first thing" in out and "second thing" in out


def test_headings_are_removed():
    """Markdown headings are documentation register, not conversation."""
    out = interview.strip_structure("## Getting Started\nDig the bed over first.")
    assert "#" not in out
    assert "Dig the bed over first." in out


def test_a_conversation_alternates_and_ends_on_the_assistant():
    """Ending on an unanswered user turn would teach the model to ignore questions."""
    c = StubClient([_long(4), "But what about clay soil?", _long(4)])
    turns = interview.build_conversation(c, "How do I start a garden?", depth=2, seed=1)
    assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant"]


def test_the_opener_is_asked_as_a_real_question_not_as_a_script_request():
    """This is the whole point of the module. The first call must send the user's
    question as a user message -- never a 'write a dialogue' instruction."""
    c = StubClient([_long(4)])
    interview.build_conversation(c, "How do I start a garden?", depth=1, seed=1)
    first = c.asks[0]
    assert first["messages"][0]["role"] == "user"
    assert first["messages"][0]["content"] == "How do I start a garden?"
    assert "dialogue" not in (first["system"] or "").lower()
    assert "turns" not in (first["system"] or "").lower()


def test_follow_up_asks_for_the_persons_next_line_only():
    """Pass 1 on a live conversation writes the USER side, not a reply."""
    c = StubClient([_long(4), "And in winter?", _long(4)])
    interview.build_conversation(c, "How do I start a garden?", depth=2, seed=1)
    assert "NEXT thing the PERSON says" in c.asks[1]["system"]


def test_follow_up_label_prefixes_are_stripped():
    """The teacher often echoes the PERSON: label; it must not enter the corpus."""
    c = StubClient([_long(4), 'PERSON: "And in winter?"', _long(4)])
    turns = interview.build_conversation(c, "Opener?", depth=2, seed=1)
    assert turns[2]["text"] == "And in winter?"


def test_an_empty_first_answer_yields_no_record():
    """A conversation with no reply is not a record."""
    assert interview.build_conversation(StubClient([""]), "Opener?", depth=2, seed=1) is None


def test_generation_stops_early_when_the_teacher_dries_up():
    """A blank mid-conversation reply truncates cleanly instead of padding."""
    c = StubClient([_long(4), "And then?", ""])
    turns = interview.build_conversation(c, "Opener?", depth=3, seed=1)
    assert [t["role"] for t in turns] == ["user", "assistant"]


def test_replies_are_capped_by_their_register():
    """A thorough answer is allowed to run; a brief one is not."""
    huge = _long(80)
    for reg in interview.REGISTERS:
        out = interview.clean_reply(huge, reg["max_bytes"])
        assert len(out.encode()) <= reg["max_bytes"] + 200
    brief = interview.clean_reply(huge, 180)
    thorough = interview.clean_reply(huge, 1400)
    assert len(brief.encode()) < len(thorough.encode())


@pytest.mark.parametrize("opener", [
    "Sure! To ensure everyone enjoys the trip, start by discussing it.",
    "Sure, I recommend checking out TripIt for that.",
    "Of course, you can prune them in autumn.",
    "Great question. The soil matters most here.",
    "Absolutely! It works well in shade.",
])
def test_filler_openers_are_stripped(opener):
    """Measured at 8.3% of replies on the first live run against a 0.22% baseline
    in mixed_chat -- a register tic, not noise, and it fails the artifact gate."""
    out = interview.strip_filler_opener(opener)
    assert not out.lower().startswith(("sure", "of course", "great question", "absolutely"))
    assert out[0].isupper()


def test_stripping_an_opener_does_not_lowercase_the_sentence():
    """'Of course, you can...' must not become 'you can...'."""
    assert interview.strip_filler_opener("Of course, you can prune them.") == "You can prune them."


def test_a_reply_that_does_not_open_with_filler_is_untouched():
    """The stripper must not eat legitimate openings."""
    text = "Surely you jest, but yes, that works."
    assert interview.strip_filler_opener(text) == text


def test_openers_are_deduplicated_and_a_dry_pool_stops_early():
    """The opener pool is bounded by the genre's `situations` list. Without dedup
    a large request re-generates the same questions, pays for the conversations,
    and RecordGate rejects them as near-duplicates afterwards."""
    from teacher import interview_job

    class Repeater:
        """A teacher stuck in a rut: always the same three openers."""
        def __init__(self): self.calls = 0
        def complete(self, messages, **kw):
            self.calls += 1
            return "How do I start a garden?\nWhy is my bread flat?\nWhich laptop should I buy?"

    job = interview_job.InterviewJob(
        "j", "/tmp/nonexistent-interview", {"genres": [], "gates": {}}, [], 100, 2,
        "ollama", gate=object())
    teacher = Repeater()
    genre = {"id": "conversation", "situations": ["gardening", "cooking"]}
    out = job._openers_for(teacher, genre, 100)

    assert len(out) == 3, "duplicates must not be kept"
    assert len(set(out)) == 3
    assert teacher.calls <= 6, "must give up once the pool is dry, not keep paying"
    assert job._opener_shortfall["conversation"] == 97
