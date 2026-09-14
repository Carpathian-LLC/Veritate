# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers veritate_trainer.window_align and the --align_windows / --align_stride flags:
#   what the mixed loader is told to open windows on. A stride wins over the marker; both
#   off means the draw is unchanged (lab 2026-09-05-working-memory-program).
# tests/training/test_window_align.py
# ------------------------------------------------------------------------------------
# Imports:
# ------------------------------------------------------------------------------------
import sys
import types

from training import veritate_trainer as vt

# ------------------------------------------------------------------------------------
# Functions:
# ------------------------------------------------------------------------------------


def test_both_off_means_no_alignment():
    """The default draw is unchanged: nothing to align to."""
    assert vt.window_align(types.SimpleNamespace(align_windows=False, align_stride=0)) is None


def test_the_marker_when_align_windows_is_set():
    """--align_windows opens windows on the user-turn marker."""
    assert vt.window_align(types.SimpleNamespace(align_windows=True, align_stride=0)) == vt.CHAT_USER_OPEN


def test_a_stride_wins_over_the_marker():
    """--align_stride N opens windows on multiples of N even when align_windows is also set."""
    assert vt.window_align(types.SimpleNamespace(align_windows=True, align_stride=4096)) == 4096


def test_the_flags_parse(monkeypatch):
    """Both are reserved flags every launch may pass; they default off."""
    monkeypatch.setattr(sys, "argv", ["veritate_trainer.py", "--align_stride", "4096", "--align_windows", "1"])
    args = vt.parse_args({"description": "t", "defaults": {}})
    assert args.align_stride == 4096 and args.align_windows is True
    monkeypatch.setattr(sys, "argv", ["veritate_trainer.py"])
    args = vt.parse_args({"description": "t", "defaults": {}})
    assert args.align_stride == 0 and args.align_windows is False


def test_the_header_names_the_alignment():
    """A finished run's log is the only evidence of where its windows opened, so the header
    states it in words for each of the three cases."""
    ns = types.SimpleNamespace
    assert vt.window_align_label(ns(align_windows=False, align_stride=0)) == "none"
    assert vt.window_align_label(ns(align_windows=True, align_stride=0)) == "user turn"
    assert vt.window_align_label(ns(align_windows=False, align_stride=4096)) == "stride 4096B"
