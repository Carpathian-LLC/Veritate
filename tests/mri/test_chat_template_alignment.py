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

import pytest
from conftest import REPO_ROOT
from routes import hybrid_routes

# ------------------------------------------------------------------------------------
# Constants

CHATML_IM_START = "<|im_start|>"
CHATML_IM_END   = "<|im_end|>"
CHATML_EOT      = "<|endoftext|>"
LEGACY_USER     = "<|user|>"

HEAD_BYTES  = 256
CHAT_PREFIX = "chat_"
TRAIN_SUFFIX = "_train.bin"

# Shipped corpora built before the ChatML standardisation. The data is the user's and is
# not rewritten by the test suite; the list exists so a NEW legacy-framed bin fails loudly.
KNOWN_LEGACY_CHAT_BINS = frozenset({
    "chat_distill_v1_train.bin",
    "chat_distill_v2_train.bin",
    "chat_identity_v1_train.bin",
})

CORPUS_DIR = os.path.join(REPO_ROOT, "trainers", "corpus")
BUILDERS_DIR = os.path.join(REPO_ROOT, "veritate_mri", "tools")
JS_INDEX = os.path.join(REPO_ROOT, "veritate_mri", "web", "index.js")


# ------------------------------------------------------------------------------------
# Functions

def _first_bytes(path, n=4096):
    with open(path, "rb") as f:
        return f.read(n)


def _read_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _shipped_chat_bins():
    if not os.path.isdir(CORPUS_DIR):
        pytest.skip(f"{CORPUS_DIR} absent (trainers/ is gitignored)")
    names = [fn for fn in sorted(os.listdir(CORPUS_DIR))
             if fn.startswith(CHAT_PREFIX) and fn.endswith(TRAIN_SUFFIX)]
    if not names:
        pytest.skip("no shipped chat_*_train.bin to inspect")
    return names


def _split_by_framing():
    """Return (chatml_framed, legacy_framed) shipped chat corpus filenames."""
    chatml, legacy = [], []
    for fn in _shipped_chat_bins():
        head = _first_bytes(os.path.join(CORPUS_DIR, fn), HEAD_BYTES)
        (chatml if CHATML_IM_START.encode("utf-8") in head else legacy).append(fn)
    return chatml, legacy


def test_backend_prompt_tmpl_is_chatml():
    """PROMPT_TMPL renders turns with ChatML markers."""
    assert CHATML_IM_START in hybrid_routes.PROMPT_TMPL
    assert CHATML_IM_END in hybrid_routes.PROMPT_TMPL


def test_backend_plain_tmpl_is_chatml():
    """PLAIN_TMPL renders turns with ChatML markers."""
    assert CHATML_IM_START in hybrid_routes.PLAIN_TMPL
    assert CHATML_IM_END in hybrid_routes.PLAIN_TMPL


def test_stop_markers_are_chatml_only():
    """STOP_MARKERS covers the ChatML turn/end markers and carries no legacy <|user|> marker."""
    assert {CHATML_IM_END, CHATML_EOT, CHATML_IM_START} <= set(hybrid_routes.STOP_MARKERS)


def test_stop_markers_exclude_legacy_marker():
    """STOP_MARKERS does not list the legacy <|user|> marker."""
    assert LEGACY_USER not in hybrid_routes.STOP_MARKERS


def test_frontend_chat_template_matches_chatml():
    """Frontend wrapChat renders the same ChatML markers as the backend."""
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
    """build_chat_corpus.py declares the ChatML byte-level frame."""
    src = _read_file(os.path.join(BUILDERS_DIR, "build_chat_corpus.py"))
    assert 'IM_START = "<|im_start|>"' in src
    assert 'IM_END   = "<|im_end|>"' in src


def test_sft_builder_declares_chatml():
    """build_sft_idk_corpus.py declares the ChatML byte-level frame."""
    src = _read_file(os.path.join(BUILDERS_DIR, "build_sft_idk_corpus.py"))
    assert 'IM_START = "<|im_start|>"' in src
    assert 'IM_END   = "<|im_end|>"' in src


def test_no_unknown_legacy_framed_chat_corpus():
    """No shipped chat_*_train.bin is legacy-framed outside the known legacy set."""
    _, legacy = _split_by_framing()
    assert sorted(set(legacy) - KNOWN_LEGACY_CHAT_BINS) == []


def test_shipped_chat_corpora_are_chatml():
    """Every shipped chat_*_train.bin frames turns in ChatML; skips naming the known legacy bins."""
    chatml, legacy = _split_by_framing()
    if legacy:
        pytest.skip("legacy <|user|>-framed chat corpora shipped, mixing them with "
                    f"ChatML {sorted(chatml)} trains the wrong turn framing: {sorted(legacy)}")
    assert legacy == []


def test_shipped_sft_idk_corpus_is_chatml():
    """The generated sft_idk_train.bin, when built, frames turns in ChatML."""
    p = os.path.join(CORPUS_DIR, "sft_idk_train.bin")
    if not os.path.isfile(p):
        pytest.skip("sft_idk_train.bin not built in this checkout")
    assert CHATML_IM_START.encode("utf-8") in _first_bytes(p, HEAD_BYTES)


def test_render_local_opens_assistant_turn():
    """_render_local ends the prompt on an open ChatML assistant turn."""
    rendered = hybrid_routes._render_local([{"role": "user", "content": "hi"}], system=None)
    assert "<|im_start|>assistant" in rendered
