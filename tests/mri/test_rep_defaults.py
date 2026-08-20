# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - chat-framed decode defaults the no-repeat hard ban ON: grounded-reply looping
#   measured 0.43 -> 0.03 at unchanged accuracy (successes.md 2026-08-17). Plain
#   completion keeps it OFF because ordinary text legitimately repeats. These tests
#   pin the default resolution; explicit caller params always win at the parse site.
# tests/mri/test_rep_defaults.py
# ------------------------------------------------------------------------------------
# Imports:

from inference.backends.pytorch import NO_REPEAT_NGRAM_OFF, REP_WINDOW_OFF
from inference.decode.repetition import NO_REPEAT_NGRAM_DEFAULT, REP_WINDOW_DEFAULT
from routes import backends_routes

# ------------------------------------------------------------------------------------
# Functions


def test_chat_framed_decode_defaults_the_ban_on():
    """A chat-framed prompt with no explicit params gets the no-repeat ban."""
    assert backends_routes._rep_defaults(True) == (REP_WINDOW_DEFAULT, NO_REPEAT_NGRAM_DEFAULT)


def test_plain_completion_defaults_the_ban_off():
    """A plain prompt keeps repetition control off; completion text repeats legitimately."""
    assert backends_routes._rep_defaults(False) == (REP_WINDOW_OFF, NO_REPEAT_NGRAM_OFF)


def test_chatml_prompt_is_detected_as_framed():
    """The framed check keys on the ChatML marker the SFT trained."""
    assert backends_routes._chat_stop_seq("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
    assert backends_routes._chat_stop_seq("Once upon a time") is None
