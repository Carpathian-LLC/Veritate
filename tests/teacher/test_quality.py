# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - tests for quality filters: length, json, fence strip, simhash, dedup.
# tests/teacher/test_quality.py
# ------------------------------------------------------------------------------------
# Imports:

from veritate_mri.teacher.quality import (
    is_json_valid,
    is_length_ok,
    is_near_dup,
    simhash64,
    strip_code_fence,
)

# ------------------------------------------------------------------------------------
# Constants

# ------------------------------------------------------------------------------------
# Functions

def test_length_ok_inside():
    """is_length_ok true when length within bounds."""
    assert is_length_ok("abcde", 1, 10) is True


def test_length_ok_below():
    """is_length_ok false when length below min."""
    assert is_length_ok("ab", 5, 10) is False


def test_length_ok_above():
    """is_length_ok false when length above max."""
    assert is_length_ok("abcdefghijk", 1, 5) is False


def test_length_ok_boundary_min():
    """is_length_ok true at min boundary."""
    assert is_length_ok("abc", 3, 10) is True


def test_length_ok_boundary_max():
    """is_length_ok true at max boundary."""
    assert is_length_ok("abc", 1, 3) is True


def test_json_valid_true():
    """is_json_valid true on valid json object."""
    assert is_json_valid('{"a":1}') is True


def test_json_valid_false():
    """is_json_valid false on non json."""
    assert is_json_valid("not json") is False


def test_strip_fence_json():
    """strip_code_fence removes leading ```json and trailing ```."""
    assert strip_code_fence('```json\n{"a":1}\n```') == '{"a":1}'


def test_strip_fence_plain():
    """strip_code_fence removes leading ``` and trailing ```."""
    assert strip_code_fence('```\n{"a":1}\n```') == '{"a":1}'


def test_strip_fence_none():
    """strip_code_fence leaves text alone when no fence."""
    assert strip_code_fence('{"a":1}') == '{"a":1}'


def test_simhash_deterministic():
    """simhash64 returns same value for same input."""
    assert simhash64("hello world foo bar baz") == simhash64("hello world foo bar baz")


def test_near_dup_identical():
    """is_near_dup true for identical simhash."""
    h = simhash64("the quick brown fox jumps over the lazy dog")
    assert is_near_dup(h, {h}) is True


def test_near_dup_distant():
    """is_near_dup false for very different strings."""
    h1 = simhash64("the quick brown fox jumps over the lazy dog")
    h2 = simhash64("completely different content here with no overlap at all xyz")
    assert is_near_dup(h1, {h2}, hamming_threshold=2) is False
