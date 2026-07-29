# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - _keep_assistant decides which bytes loss is computed on, so a change to it silently
#   changes what every SFT learns. The byte-walk reference below is the pre-optimization
#   implementation; the shipped scanner must agree with it on every input, including
#   windows that start or end mid-turn.
# tests/training/test_loss_mask_equivalence.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import random
import sys

import numpy as np

TRAINER_COMMON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "trainers", "common")
if TRAINER_COMMON not in sys.path:
    sys.path.insert(0, TRAINER_COMMON)

import vanilla_trainer as vt

# ------------------------------------------------------------------------------------
# Constants

TURN = (b"<|im_start|>user\nname two rivers<|im_end|>\n"
        b"<|im_start|>assistant\nthe Rhine and the Danube<|im_end|>\n<|endoftext|>\n")
PROSE = b"The culvert under the orchard road floods every spring and nobody has fixed it. "

# ------------------------------------------------------------------------------------
# Functions


def _reference(row_bytes):
    """The original byte-at-a-time walk, kept as the correctness oracle."""
    n = len(row_bytes)
    keep = np.zeros(n, dtype=bool)
    ao, ist, ie = vt.CHATML_ASSISTANT_OPEN, vt.CHATML_IM_START, vt.CHATML_IM_END
    la, li, le = len(ao), len(ist), len(ie)
    i = 0
    in_asst = False
    while i < n:
        if row_bytes[i:i + la] == ao:
            in_asst = True; i += la; continue
        if row_bytes[i:i + li] == ist:
            in_asst = False; i += li; continue
        if in_asst and row_bytes[i:i + le] == ie:
            keep[i:i + le] = True; i += le; in_asst = False; continue
        keep[i] = in_asst
        i += 1
    return keep


def _agree(row):
    assert np.array_equal(vt._keep_assistant(row), _reference(row)), row[:120]


def test_a_whole_turn_matches_the_reference():
    _agree(TURN)


def test_many_turns_match_the_reference():
    _agree(TURN * 12)


def test_pure_prose_matches_the_reference():
    _agree(PROSE * 6)


def test_a_window_starting_mid_assistant_reply_matches():
    """Training windows cut anywhere, so a reply with no visible opening must agree."""
    _agree(TURN[40:])


def test_a_window_ending_mid_assistant_reply_matches():
    _agree(TURN[:70])


def test_an_unterminated_assistant_turn_matches():
    _agree(b"<|im_start|>assistant\nthis reply never closes")


def test_a_user_turn_interrupting_an_assistant_turn_matches():
    """Malformed but reachable: the next turn opening must end the kept span."""
    _agree(b"<|im_start|>assistant\npartial<|im_start|>user\nnext<|im_end|>")


def test_the_closing_marker_is_kept_but_the_opening_is_not():
    keep = vt._keep_assistant(TURN)
    body = TURN.index(b"the Rhine")
    assert keep[body]
    assert not keep[TURN.index(b"<|im_start|>assistant")]


def test_the_user_turn_is_never_kept():
    keep = vt._keep_assistant(TURN)
    u = TURN.index(b"name two rivers")
    assert not keep[u:u + len(b"name two rivers")].any()


def test_random_byte_soup_matches_the_reference():
    """Marker fragments at arbitrary offsets are the cases a scanner gets wrong."""
    rng = random.Random(20260728)
    parts = [b"<|im_start|>", b"assistant\n", b"user\n", b"<|im_end|>", b"hello ", b"<|im_",
             b"|>", b"<|endoftext|>", PROSE[:20]]
    for _ in range(300):
        row = b"".join(rng.choice(parts) for _ in range(rng.randint(1, 24)))
        _agree(row)
