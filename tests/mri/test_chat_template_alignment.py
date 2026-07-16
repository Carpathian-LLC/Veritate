# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Cross-boundary alignment: the Carpathian chat corpora (chat_50mb / chat_500mb /
#   chat_5gb), the corpus-BUILDER (build_chat_corpus.py), the SFT-BUILDER
#   (build_sft_idk_corpus.py), the backend chat render (hybrid_routes.PROMPT_TMPL /
#   PLAIN_TMPL / STOP_MARKERS), and the frontend chat render (web/index.js
#   wrapChat) MUST all use the same byte-level chat markers. Any drift here
#   silently poisons the model at serving time (the bug that shipped when the
#   backend used <|user|> but the corpus used <|im_start|>).
# - Arch-agnostic: pure string / bytes / file inspection, identical on all OS.
# tests/mri/test_chat_template_alignment.py
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
# Constants

CHATML_IM_START = "<|im_start|>"
CHATML_IM_END   = "<|im_end|>"
CHATML_EOT      = "<|endoftext|>"

CORPUS_DIR = os.path.join(REPO_ROOT, "trainers", "corpus")
BUILDERS_DIR = os.path.join(REPO_ROOT, "veritate_mri", "tools")
JS_INDEX = os.path.join(REPO_ROOT, "veritate_mri", "web", "index.js")


# ------------------------------------------------------------------------------------
# Functions

def _first_bytes(path, n=4096):
    with open(path, "rb") as f:
        return f.read(n)


def _read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def test_backend_chat_template_matches_chatml():
    """PROMPT_TMPL and PLAIN_TMPL and STOP_MARKERS on the backend use ChatML."""
    assert CHATML_IM_START in hybrid_routes.PROMPT_TMPL
    assert CHATML_IM_END   in hybrid_routes.PROMPT_TMPL
    assert CHATML_IM_START in hybrid_routes.PLAIN_TMPL
    assert CHATML_IM_END   in hybrid_routes.PLAIN_TMPL
    assert CHATML_IM_END   in hybrid_routes.STOP_MARKERS
    assert CHATML_EOT      in hybrid_routes.STOP_MARKERS
    assert CHATML_IM_START in hybrid_routes.STOP_MARKERS


def test_frontend_chat_template_matches_chatml():
    """Frontend wrapChat MUST render the same ChatML markers as the backend so
    a browser-round-trip prompt is byte-identical to a curl POST."""
    text = _read_file(JS_INDEX)
    idx = text.find("function wrapChat(")
    assert idx >= 0
    depth = 0
    end = idx
    started = False
    for i in range(idx, len(text)):
        c = text[i]
        if c == "{":
            depth += 1; started = True
        elif c == "}":
            depth -= 1
            if started and depth == 0:
                end = i + 1
                break
    body = text[idx:end]
    assert "CHATML_IM_START" in body, "frontend wrapChat must use ChatML"
    assert "CHATML_IM_END"   in body


def test_corpus_builder_declares_chatml():
    """build_chat_corpus.py (the OWNER of the shipped Carpathian chat corpora)
    must declare ChatML as its byte-level frame. If someone reintroduces a
    <|user|> builder, this catches it immediately."""
    src = _read_file(os.path.join(BUILDERS_DIR, "build_chat_corpus.py"))
    assert 'IM_START = "<|im_start|>"' in src
    assert 'IM_END   = "<|im_end|>"' in src


def test_sft_builder_declares_chatml():
    """build_sft_idk_corpus.py must emit the same ChatML frame as the shipped
    chat corpora so the sft_idk stem interleaves cleanly in the pretrain mix."""
    src = _read_file(os.path.join(BUILDERS_DIR, "build_sft_idk_corpus.py"))
    assert 'IM_START = "<|im_start|>"' in src
    assert 'IM_END   = "<|im_end|>"' in src


def test_shipped_chat_corpora_are_chatml():
    """Every shipped chat_*_train.bin under trainers/corpus/ MUST start with
    ChatML markers. Catches a corrupted or re-framed corpus swap silently
    breaking every downstream chat model."""
    for fn in os.listdir(CORPUS_DIR) if os.path.isdir(CORPUS_DIR) else []:
        if not (fn.startswith("chat_") and fn.endswith("_train.bin")):
            continue
        p = os.path.join(CORPUS_DIR, fn)
        head = _first_bytes(p, 256)
        assert CHATML_IM_START.encode("utf-8") in head, (
            f"{fn}: expected ChatML frame in first 256 bytes")


def test_shipped_sft_idk_corpus_is_chatml():
    """The generated SFT bin (matches sft_idk stem) uses ChatML if present."""
    for tail in ("sft_idk_train.bin",):
        p = os.path.join(CORPUS_DIR, tail)
        if not os.path.isfile(p):
            return  # optional artifact; skip if not built
        head = _first_bytes(p, 256)
        assert CHATML_IM_START.encode("utf-8") in head
        assert CHATML_IM_END.encode("utf-8")   in head


def test_render_local_output_matches_stop_markers():
    """End-to-end: a rendered prompt must terminate exactly where the STOP_MARKERS
    would begin the next turn. Guards against silent template drift between
    render and stop-detection."""
    msgs = [{"role": "user", "content": "hi"}]
    rendered = hybrid_routes._render_local(msgs, system=None)
    # The rendered prompt ends with an OPEN assistant marker. If the model outputs
    # <|im_end|> we should be able to detect it. Prove the marker literally appears
    # in STOP_MARKERS so streaming can halt.
    assert "<|im_start|>assistant" in rendered
    assert any(m in ("<|im_end|>",) for m in hybrid_routes.STOP_MARKERS)
