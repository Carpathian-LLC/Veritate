# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Regression: hybrid_routes prompt rendering uses the SAME ChatML format as the
#   Carpathian-hosted chat corpora (build_chat_corpus.py IM_START/IM_END) so the
#   trained model recognises the turn markers. Prior version emitted <|user|> /
#   <|assistant|> tags which the models never saw in training, producing garbled
#   completions (the tags leaking into the output on 80M chat80idk_80m).
# - Arch-agnostic: pure string templating, identical behaviour on win32/darwin/linux.
# tests/mri/test_chat_template_render.py
# ------------------------------------------------------------------------------------
# Imports

import os

import pytest
from conftest import REPO_ROOT
from routes import hybrid_routes

# ------------------------------------------------------------------------------------
# Constants

TEMPLATES       = ("PROMPT_TMPL", "PLAIN_TMPL")
CHATML_MARKERS  = ("<|im_start|>user", "<|im_end|>", "<|im_start|>assistant")
LEGACY_TAGS     = ("<|user|>", "<|assistant|>")
ASSISTANT_OPEN  = "<|im_start|>assistant"
WRAPCHAT_MARKER = "function wrapChat("

MULTI_TURN = [
    {"role": "user",      "content": "Hi!"},
    {"role": "assistant", "content": "Hey there."},
    {"role": "user",      "content": "What can you do?"},
]
SINGLE_TURN = [{"role": "user", "content": "How old am I?"}]

# ------------------------------------------------------------------------------------
# Functions

@pytest.mark.parametrize("tmpl", TEMPLATES)
@pytest.mark.parametrize("marker", CHATML_MARKERS)
def test_template_emits_chatml_marker(tmpl, marker):
    """Each chat template emits every ChatML turn marker the corpus uses."""
    assert marker in getattr(hybrid_routes, tmpl)


@pytest.mark.parametrize("tmpl", TEMPLATES)
@pytest.mark.parametrize("tag", LEGACY_TAGS)
def test_template_carries_no_legacy_tag(tmpl, tag):
    """No chat template carries a legacy turn tag the models never saw in training."""
    assert tag not in getattr(hybrid_routes, tmpl)


def test_render_local_closes_a_history_user_turn():
    """A multi-turn render closes each history user turn in ChatML."""
    assert "<|im_start|>user\nHi!<|im_end|>" in hybrid_routes._render_local(MULTI_TURN, system=None)


def test_render_local_closes_a_history_assistant_turn():
    """A multi-turn render closes each history assistant turn in ChatML."""
    assert "<|im_start|>assistant\nHey there.<|im_end|>" in \
        hybrid_routes._render_local(MULTI_TURN, system=None)


def test_render_local_multi_turn_ends_on_an_open_assistant_turn():
    """A multi-turn render ends on an open assistant marker so decode continues there."""
    assert hybrid_routes._render_local(MULTI_TURN, system=None).rstrip().endswith(ASSISTANT_OPEN)


def test_render_local_single_turn_closes_the_user_turn():
    """A single-turn render closes the user turn in ChatML."""
    assert "<|im_start|>user\nHow old am I?<|im_end|>" in \
        hybrid_routes._render_local(SINGLE_TURN, system=None)


def test_render_local_single_turn_ends_on_an_open_assistant_turn():
    """A single-turn render ends on an open assistant marker."""
    assert hybrid_routes._render_local(SINGLE_TURN, system=None).rstrip().endswith(ASSISTANT_OPEN)


def test_render_local_carries_system_as_a_context_block():
    """System text rides in as a context: block."""
    assert "context: You are a helper." in \
        hybrid_routes._render_local(SINGLE_TURN, system="You are a helper.")


@pytest.mark.parametrize("messages", (SINGLE_TURN, MULTI_TURN))
def test_render_local_open_is_a_strict_prefix_of_the_wire_prompt(messages):
    """The open form read-ahead sends is a strict prefix of the prompt submit renders."""
    wire = hybrid_routes._render_local(messages, system=None)
    open_form = hybrid_routes.render_local_open(messages, system=None)
    assert wire.startswith(open_form) and len(open_form) < len(wire)


@pytest.mark.parametrize("messages", (SINGLE_TURN, MULTI_TURN))
def test_render_local_open_ends_on_the_typed_text(messages):
    """The open form stops at the user's text, so the scaffold never moves as it grows."""
    assert hybrid_routes.render_local_open(messages, system=None).endswith(messages[-1]["content"])


def test_render_local_open_carries_system_as_a_context_block():
    """System text rides into the open form as the same context: block."""
    assert "context: You are a helper." in \
        hybrid_routes.render_local_open(SINGLE_TURN, system="You are a helper.")


def _wrapchat_body():
    js_path = os.path.normpath(os.path.join(REPO_ROOT, "veritate_mri", "web", "index.js"))
    with open(js_path, encoding="utf-8") as f:
        text = f.read()
    idx = text.find(WRAPCHAT_MARKER)
    assert idx >= 0, "wrapChat not found in index.js"
    depth, started = 0, False
    for i in range(idx, len(text)):
        if text[i] == "{":
            depth += 1; started = True
        elif text[i] == "}":
            depth -= 1
            if started and depth == 0:
                return text[idx:i + 1]
    raise AssertionError("wrapChat body never closes")


@pytest.mark.parametrize("name", ("CHATML_IM_START", "CHATML_IM_END"))
def test_frontend_wrapchat_uses_chatml_constant(name):
    """The browser wrapChat() renders turns with the ChatML marker constants."""
    assert name in _wrapchat_body()


@pytest.mark.parametrize("name", ("CHAT_USER_TAG", "CHAT_ASSISTANT_TAG"))
def test_frontend_wrapchat_drops_legacy_constant(name):
    """The browser wrapChat() no longer references the legacy turn-tag constants."""
    assert name not in _wrapchat_body()
