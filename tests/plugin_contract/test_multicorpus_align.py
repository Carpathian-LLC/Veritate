# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers multicorpus.make_mixed_loader(align=...): a drawn window slides forward to the
#   next turn marker so a chat window opens on a turn boundary. A window that opens on a
#   question whose answer was told before the window began trains a guess; on the recall
#   corpora that guess showed up as invented facts (lab 2026-09-05-working-memory-program).
# tests/plugin_contract/test_multicorpus_align.py
# ------------------------------------------------------------------------------------
# Imports:
# ------------------------------------------------------------------------------------
import numpy as np

from veritate_core.plugin.multicorpus import make_mixed_loader

# ------------------------------------------------------------------------------------
# Constants:
# ------------------------------------------------------------------------------------
MARK = b"<|im_start|>user\n"
SEQ = 64


# ------------------------------------------------------------------------------------
# Functions:
# ------------------------------------------------------------------------------------
def _chat_bin(tmp_path, n_turns=200):
    body = b"".join(MARK + b"q%03d<|im_end|>\n<|im_start|>assistant\na%03d<|im_end|>\n" % (i, i)
                    for i in range(n_turns))
    p = tmp_path / "chat_train.bin"
    p.write_bytes(body)
    return str(p)


def test_aligned_windows_open_on_the_marker(tmp_path):
    """With align set, a window either starts exactly on a marker or holds none at all; the
    targets stay the window shifted by one byte."""
    draw, _ = make_mixed_loader([(_chat_bin(tmp_path), None, 1.0)], 8, SEQ, seed=1, align=MARK)
    on_marker = 0
    for _ in range(20):
        toks, tgts = draw()
        for row, nxt in zip(toks.numpy(), tgts.numpy(), strict=True):
            raw = bytes(row.astype(np.uint8))
            assert raw.startswith(MARK) or MARK not in raw
            on_marker += raw.startswith(MARK)
            assert (nxt[:-1] == row[1:]).all()
    assert on_marker > 100


def test_without_align_windows_land_anywhere(tmp_path):
    """The default draw is unchanged: windows open mid-turn most of the time."""
    draw, _ = make_mixed_loader([(_chat_bin(tmp_path), None, 1.0)], 8, SEQ, seed=1)
    starts_on_marker = 0
    for _ in range(20):
        toks, _ = draw()
        for row in toks.numpy():
            starts_on_marker += bytes(row[:len(MARK)].astype(np.uint8)) == MARK
    assert starts_on_marker < 20


def test_an_int_align_opens_windows_on_the_stride(tmp_path):
    """With an int stride every window opens on a multiple of it, so records padded to the
    stride are read whole: the first byte of each window is a record's first byte."""
    rec = b"<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\na<|im_end|>\n"
    stride = 128
    p = tmp_path / "padded_train.bin"
    p.write_bytes(b"".join((b"%03d" % i + rec).ljust(stride, b"\n") for i in range(64)))
    draw, _ = make_mixed_loader([(str(p), None, 1.0)], 8, stride, seed=2, align=stride)
    for _ in range(10):
        toks, _ = draw()
        for row in toks.numpy():
            raw = bytes(row.astype(np.uint8))
            assert raw[3:3 + len(rec)] == rec and raw[:3].isdigit()


def test_a_corpus_without_markers_draws_as_before(tmp_path):
    """Prose bins carry no marker; align must not change their sampling."""
    p = tmp_path / "prose_train.bin"
    p.write_bytes(bytes(range(256)) * 40)
    plain, _ = make_mixed_loader([(str(p), None, 1.0)], 4, SEQ, seed=3)
    aligned, _ = make_mixed_loader([(str(p), None, 1.0)], 4, SEQ, seed=3, align=MARK)
    a, _ = plain()
    b, _ = aligned()
    assert (a == b).all()
