# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Regression: /hybrid/chat renders prompts in the SAME ChatML format as the
#   Carpathian-hosted chat corpora (build_chat_corpus.py IM_START/IM_END) so the
#   trained model recognises the turn markers. Prior version emitted <|user|> /
#   <|assistant|> tags which the models never saw in training, producing garbled
#   completions (the tags leaking into the output on 80M chat80idk_80m).
# - Arch-agnostic: pure string templating, identical behaviour on win32/darwin/linux.
# tests/mri/test_chat_template_render.py
# ------------------------------------------------------------------------------------
# Imports

import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "veritate_mri")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from routes import hybrid_routes


# ------------------------------------------------------------------------------------
# Functions

def test_prompt_tmpl_uses_chatml_markers():
    """PROMPT_TMPL and PLAIN_TMPL must emit ChatML markers matching the corpus."""
    assert "<|im_start|>user" in hybrid_routes.PROMPT_TMPL
    assert "<|im_end|>"        in hybrid_routes.PROMPT_TMPL
    assert "<|im_start|>assistant" in hybrid_routes.PROMPT_TMPL

    assert "<|im_start|>user" in hybrid_routes.PLAIN_TMPL
    assert "<|im_end|>"        in hybrid_routes.PLAIN_TMPL
    assert "<|im_start|>assistant" in hybrid_routes.PLAIN_TMPL


def test_stop_markers_are_chatml():
    """Stop markers must be the ChatML tags so streaming halts at the correct token."""
    assert "<|im_end|>"      in hybrid_routes.STOP_MARKERS
    assert "<|im_start|>"    in hybrid_routes.STOP_MARKERS
    assert "<|endoftext|>"   in hybrid_routes.STOP_MARKERS


def test_no_platform_tags_leak_into_templates():
    """The legacy <|user|> / <|assistant|> tags must not appear as literal user-turn
    markers. (They may still appear in AUTO-DETECT paths in backends_routes for
    backward compat with any legacy caller that hand-crafts a prompt.)"""
    assert "<|user|>"      not in hybrid_routes.PROMPT_TMPL
    assert "<|assistant|>" not in hybrid_routes.PROMPT_TMPL
    assert "<|user|>"      not in hybrid_routes.PLAIN_TMPL
    assert "<|assistant|>" not in hybrid_routes.PLAIN_TMPL


def test_render_local_multi_turn_chatml():
    """Multi-turn history + final turn rendered in ChatML end-to-end."""
    msgs = [
        {"role": "user",      "content": "Hi!"},
        {"role": "assistant", "content": "Hey there."},
        {"role": "user",      "content": "What can you do?"},
    ]
    rendered = hybrid_routes._render_local(msgs, system=None)
    assert "<|im_start|>user\nHi!<|im_end|>" in rendered
    assert "<|im_start|>assistant\nHey there.<|im_end|>" in rendered
    # Final turn ends with an OPEN assistant marker so decode continues from there.
    assert rendered.rstrip().endswith("<|im_start|>assistant")


def test_render_local_single_turn():
    """Single-turn prompt uses PLAIN_TMPL and ends with an open assistant marker."""
    msgs = [{"role": "user", "content": "How old am I?"}]
    rendered = hybrid_routes._render_local(msgs, system=None)
    assert "<|im_start|>user\nHow old am I?<|im_end|>" in rendered
    assert rendered.rstrip().endswith("<|im_start|>assistant")


def test_render_local_with_system_context():
    """System text rides in as a `context:` block via PROMPT_TMPL."""
    msgs = [{"role": "user", "content": "Hello."}]
    rendered = hybrid_routes._render_local(msgs, system="You are a helper.")
    assert "context: You are a helper." in rendered
    assert "<|im_start|>user\nHello.<|im_end|>" in rendered


def test_frontend_wrapchat_uses_chatml():
    """Regression: veritate_mri/web/index.js wrapChat() must emit ChatML so
    browser-side chat matches the corpus + server render. Grepping the JS is
    sufficient — the function is a one-liner template string."""
    js_path = os.path.normpath(os.path.join(REPO_ROOT, "veritate_mri", "web", "index.js"))
    with open(js_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Find the wrapChat function body.
    marker = "function wrapChat("
    idx = text.find(marker)
    assert idx >= 0, "wrapChat not found in index.js"
    # Skip past the first `{` (function opener) and count braces to find matching close.
    depth = 0
    body_end = idx
    started = False
    for i in range(idx, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
            started = True
        elif c == "}":
            depth -= 1
            if started and depth == 0:
                body_end = i + 1
                break
    body = text[idx:body_end]
    assert "CHATML_IM_START" in body and "CHATML_IM_END" in body, (
        "wrapChat must use ChatML tags (CHATML_IM_START / CHATML_IM_END) so "
        "browser chat matches the Carpathian chat corpus format.")
    assert "CHAT_USER_TAG"      not in body
    assert "CHAT_ASSISTANT_TAG" not in body
