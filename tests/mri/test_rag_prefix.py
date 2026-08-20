# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - build_rag_prefix has to hand the model the same framing the chat SFT trained:
#   ChatML ending at <|im_start|>assistant\n. loss_mask=assistant grades only the
#   bytes after that marker, so a prefix without it puts the model in completion
#   mode. These tests pin the framing, not the wording.
# - the client sends either a raw message or an already-framed ChatML prompt.
#   Both must come back framed exactly once, so both cases are covered here.
# - retriever hits carry the whole chunk. A preview-length cut used to drop the
#   answer whenever it sat past the cut, which is invisible end-to-end because the
#   model just answers wrong.
# tests/mri/test_rag_prefix.py
# ------------------------------------------------------------------------------------
# Imports:

import os

from conftest import REPO_ROOT
from inference.agent.rag import (
    ASSISTANT_OPEN,
    DEFAULT_MAX_PASSAGES_B,
    PASSAGE_SEPARATOR,
    REPLY_RESERVE_B,
    TRUNCATION_MARK,
    USER_OPEN,
    build_rag_prefix,
    injection_budget,
)
from inference.agent.tools import retriever

# ------------------------------------------------------------------------------------
# Constants

IM_END   = "<|im_end|>"
QUESTION = "Which folder has the contracts?"
PASSAGE  = "The blue folder holds the invoices. The red folder holds the contracts."

# ------------------------------------------------------------------------------------
# Functions


def test_raw_prompt_is_framed_as_a_chatml_turn():
    """A bare user message comes back as one ChatML turn ending at the assistant open."""
    out = build_rag_prefix(QUESTION, [PASSAGE])
    assert out.startswith(USER_OPEN)
    assert out.endswith(ASSISTANT_OPEN)
    assert out.count(IM_END) == 1


def test_passages_precede_the_question_inside_the_user_turn():
    """Context leads, question follows: the shape the grounded A/B measured."""
    out = build_rag_prefix(QUESTION, [PASSAGE])
    assert out.index(PASSAGE) < out.index(QUESTION) < out.index(IM_END)


def test_already_framed_prompt_is_not_double_framed():
    """A ChatML prompt keeps exactly one assistant open, not two."""
    framed = f"{USER_OPEN}{QUESTION}{IM_END}\n{ASSISTANT_OPEN}"
    out = build_rag_prefix(framed, [PASSAGE])
    assert out.count(ASSISTANT_OPEN) == 1
    assert out.count(USER_OPEN) == 1
    assert out.index(PASSAGE) < out.index(QUESTION)


def test_passages_land_in_the_last_user_turn_of_a_history():
    """Multi-turn history grounds the current question, not an earlier one."""
    framed = (f"{USER_OPEN}first{IM_END}\n{ASSISTANT_OPEN}reply{IM_END}\n"
              f"{USER_OPEN}{QUESTION}{IM_END}\n{ASSISTANT_OPEN}")
    out = build_rag_prefix(framed, [PASSAGE])
    assert out.index("first") < out.index(PASSAGE) < out.index(QUESTION)


def test_no_passages_returns_the_prompt_untouched():
    """Retrieving nothing must not fabricate an empty context block."""
    assert build_rag_prefix(QUESTION, []) == QUESTION


def test_passage_crossing_the_cap_is_truncated_not_dropped():
    """One oversized chunk still grounds the answer."""
    out = build_rag_prefix(QUESTION, ["x" * 200], max_bytes=100)
    assert TRUNCATION_MARK in out
    assert out.count("x") == 100


def test_passages_are_separated_by_a_blank_line():
    """Callers split hits on a blank line; the joiner has to match."""
    out = build_rag_prefix(QUESTION, ["alpha", "beta"])
    assert f"alpha{PASSAGE_SEPARATOR}beta" in out


def test_retriever_returns_the_whole_chunk(tmp_path):
    """A hit carries the full chunk, so an answer past the old 480-char preview
    still reaches the reader."""
    answer = "the vault code is 8891"
    body = ("filler " * 120) + answer
    src = tmp_path / "doc.txt"
    src.write_text(body, encoding="utf-8")

    tool = retriever.make_tool(str(src))
    out = tool.call({"query": "vault code", "k": 1})

    assert len(body) > 480
    assert answer in out


def test_retriever_hit_holds_no_blank_line(tmp_path):
    """Hits are split on a blank line, so no single hit may contain one."""
    src = tmp_path / "doc.txt"
    src.write_text("alpha beta\n\ngamma delta", encoding="utf-8")

    tool = retriever.make_tool(str(src))
    out = tool.call({"query": "gamma", "k": 1})

    assert "\n\n" not in out


def test_injection_budget_keeps_the_prefix_inside_the_window():
    """A budget-capped prefix leaves the reply reserve free inside a seq-byte window,
    so the user-turn opener is never evicted before the model answers."""
    seq = 1024
    budget = injection_budget(seq, len(QUESTION.encode()))
    out = build_rag_prefix(QUESTION, ["x" * 2048], max_bytes=budget)
    frame = len(USER_OPEN) + len(PASSAGE_SEPARATOR) + len(IM_END) + 1 + len(ASSISTANT_OPEN)
    assert len(out.encode()) <= seq - REPLY_RESERVE_B + frame + len(TRUNCATION_MARK)


def test_injection_budget_is_zero_when_the_prompt_fills_the_window():
    """A prompt already at the window means inject nothing, not a broken frame."""
    assert injection_budget(1024, 1024) == 0
    assert build_rag_prefix(QUESTION, ["x" * 100], max_bytes=0) == QUESTION


def test_injection_budget_is_capped_by_the_passage_default():
    """A long window still respects the global passage cap."""
    assert injection_budget(1 << 20, 0) == DEFAULT_MAX_PASSAGES_B


def test_rag_default_k_is_top_one():
    """Multi-candidate injection is falsified (failures.md); the shipped default
    must stay at one passage."""
    from routes import backends_routes
    assert backends_routes.RAG_K_DEFAULT == 1


def test_rag_prefix_builder_ships_in_the_platform():
    """The route imports this module at startup; it must exist in the repo."""
    assert os.path.exists(os.path.join(REPO_ROOT, "veritate_mri", "inference",
                                       "agent", "rag.py"))
