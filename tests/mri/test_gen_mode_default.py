# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Generation-tab mode picker contract: chat is the default mode and is never
#   capability-gated. Only the other tiers grey out until trained, and when a
#   checked tier disables, the picker falls back to chat.
# - Source inspection of web/index.js and web/index.html, same pattern as
#   test_chat_template_alignment.py. Arch-agnostic.
# tests/mri/test_gen_mode_default.py
# ------------------------------------------------------------------------------------
# Imports

import os
import re

from conftest import REPO_ROOT

# ------------------------------------------------------------------------------------
# Constants

WEB_DIR    = os.path.join(REPO_ROOT, "veritate_mri", "web")
JS_INDEX   = os.path.join(WEB_DIR, "index.js")
HTML_INDEX = os.path.join(WEB_DIR, "index.html")


# ------------------------------------------------------------------------------------
# Functions

def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _js_fn(name):
    """Extract one function body from index.js by brace matching."""
    text = _read(JS_INDEX)
    idx = text.find(f"function {name}(")
    assert idx >= 0, f"{name} missing from index.js"
    depth = 0
    started = False
    for i in range(idx, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
            started = True
        elif c == "}":
            depth -= 1
            if started and depth == 0:
                return text[idx:i + 1]
    raise AssertionError(f"unbalanced braces in {name}")


def test_gen_mode_default_is_chat():
    """GEN_MODE_DEFAULT is chat and the chat radio ships checked."""
    assert 'const GEN_MODE_DEFAULT = "chat"' in _read(JS_INDEX)
    assert re.search(r'value="chat"\s+checked', _read(HTML_INDEX))


def test_chat_mode_is_never_capability_gated():
    """_applyModeAvailability keeps chat selectable regardless of trained status."""
    body = _js_fn("_applyModeAvailability")
    assert 'tier === GEN_MODE_DEFAULT || entry.status === "trained"' in body


def test_mode_fallback_lands_on_chat():
    """A disabled checked tier falls back to the chat default, not the first trained tier."""
    body = _js_fn("_applyModeAvailability")
    assert '[value="${GEN_MODE_DEFAULT}"]' in body
    assert "firstTrained" not in body
