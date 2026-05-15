# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Byte-level constraints used by ConstrainedDecoder. Each constraint exposes:
#     mask() -> bool[256]   True at b iff b is legal as the very next byte.
#     step(b)               update internal state after committing to byte b.
#     done() -> bool        True when generation should stop now.
#     reset()               forget everything emitted so far.
# - Pure-Python (no torch). The decoder turns masks into -inf logit additions.
# veritate_mri/inference/decode/constraints.py
# ------------------------------------------------------------------------------------
# Imports:

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np


# ------------------------------------------------------------------------------------
# Constants

_ALL_BYTES_TRUE  = np.ones(256, dtype=bool)
_ALL_BYTES_FALSE = np.zeros(256, dtype=bool)


# ------------------------------------------------------------------------------------
# Functions

def _empty_mask() -> np.ndarray:
    return np.zeros(256, dtype=bool)


# ------------------------------------------------------------------------------------
# Base class
# ------------------------------------------------------------------------------------

class Constraint:
    """Abstract base. Concrete subclasses must override `mask`, `step`."""

    def mask(self) -> np.ndarray:           # pragma: no cover (abstract)
        raise NotImplementedError

    def step(self, byte: int) -> None:      # pragma: no cover (abstract)
        raise NotImplementedError

    def done(self) -> bool:
        return False

    def reset(self) -> None:
        pass


# ------------------------------------------------------------------------------------
# VocabConstraint
# ------------------------------------------------------------------------------------

class VocabConstraint(Constraint):
    """Restrict output to a fixed set of byte values.

    Example: ASCII-printable + tab/LF/CR
        VocabConstraint(set(range(0x20, 0x7f)) | {0x09, 0x0a, 0x0d})

    Example: lowercase ASCII letters only
        VocabConstraint(set(range(0x61, 0x7b)))
    """

    def __init__(self, allowed_bytes: Iterable[int]):
        allowed = set(int(b) & 0xff for b in allowed_bytes)
        if not allowed:
            raise ValueError("VocabConstraint needs at least one allowed byte")
        m = _empty_mask()
        for b in allowed:
            m[b] = True
        self._mask = m
        self.allowed = allowed

    def mask(self) -> np.ndarray:
        return self._mask

    def step(self, byte: int) -> None:
        # Stateless w.r.t. history; just sanity check.
        if byte not in self.allowed:
            # Caller violated the contract. We don't raise (that would be
            # rude during decode); just no-op.
            pass

    def reset(self) -> None:
        pass


# ------------------------------------------------------------------------------------
# StopOnConstraint
# ------------------------------------------------------------------------------------

class StopOnConstraint(Constraint):
    """Halt decoding when the emitted output ends with `stop_bytes`.

    The constraint never masks anything (every byte is allowed). It only
    affects `done()`. The decoder is responsible for checking `done()` after
    each `step()` and breaking out of its loop.

    Why this is a "constraint": it composes through CombineConstraint -- you
    can ask for "lowercase only, until you see \\n\\n" without writing a
    special-cased decoder.
    """

    def __init__(self, stop_bytes: bytes):
        if not isinstance(stop_bytes, (bytes, bytearray)):
            raise TypeError(f"stop_bytes must be bytes, got {type(stop_bytes)}")
        if len(stop_bytes) == 0:
            raise ValueError("stop_bytes must be non-empty")
        self.stop = bytes(stop_bytes)
        self._tail: bytearray = bytearray()
        self._done = False

    def mask(self) -> np.ndarray:
        return _ALL_BYTES_TRUE

    def step(self, byte: int) -> None:
        self._tail.append(byte & 0xff)
        # Keep the tail at most len(stop) bytes long.
        if len(self._tail) > len(self.stop):
            del self._tail[:len(self._tail) - len(self.stop)]
        if len(self._tail) == len(self.stop) and bytes(self._tail) == self.stop:
            self._done = True

    def done(self) -> bool:
        return self._done

    def reset(self) -> None:
        self._tail.clear()
        self._done = False


# ------------------------------------------------------------------------------------
# CombineConstraint
# ------------------------------------------------------------------------------------

class CombineConstraint(Constraint):
    """AND together a list of constraints.

    mask = element-wise AND of children's masks.
    done = OR of children's done() (any child saying stop -> stop).
    step  propagates to every child.
    """

    def __init__(self, constraints: List[Constraint]):
        if not constraints:
            raise ValueError("CombineConstraint needs at least one child")
        self.children: List[Constraint] = list(constraints)

    def mask(self) -> np.ndarray:
        out = _ALL_BYTES_TRUE.copy()
        for c in self.children:
            out &= c.mask()
        return out

    def step(self, byte: int) -> None:
        for c in self.children:
            c.step(byte)

    def done(self) -> bool:
        return any(c.done() for c in self.children)

    def reset(self) -> None:
        for c in self.children:
            c.reset()


# ------------------------------------------------------------------------------------
# JSONConstraint
# ------------------------------------------------------------------------------------
#
# Streaming JSON grammar. The state of the constraint is:
#
#   value_state : what we expect to see at the current position
#                 (VAL_START | OBJ_AFTER_OPEN | OBJ_AFTER_KEY | ...)
#   stack       : list of container kinds we're inside ("obj" or "arr"),
#                 deepest last.
#   in_string   : True iff we're currently between the open and close quote
#                 of a string (key or value).
#   esc         : 0 if no pending escape; 1 if we just saw '\'; 2..5 if
#                 we're partway through a \uXXXX escape (counting hex digits
#                 still to consume).
#   number_state: nested state for the in-progress numeric literal:
#                 None if we're not in a number;
#                 'sign' (just consumed '-'),
#                 'int_zero' (an unambiguous 0; cannot accept more digits),
#                 'int_nonzero' (1..9 followed by 0+ digits),
#                 'frac_dot' (just saw '.'; need at least one digit),
#                 'frac_digits' (one or more digits after '.'),
#                 'exp_sign' (saw 'e'/'E', awaiting sign or digit),
#                 'exp_digits' (digits of the exponent).
#   literal     : if we're partway through 'true'/'false'/'null', this is
#                 the bytes we still need to match, e.g. b'rue'. None
#                 otherwise.
#   done_       : True iff we've just completed the outermost JSON value
#                 (or '}' / ']'); at that point the constraint allows
#                 trailing whitespace only AND reports done()=True so the
#                 decoder halts.
#
# The grammar implemented is RFC-8259 minus number leading zeros for fractions
# only (we accept "0.5" but disallow "01"). Strings allow any byte 0x20-0x10ffff
# except for the unescaped control set; we approximate by allowing 0x20..0xff
# except '"' and '\' unescaped (technically UTF-8 multi-byte sequences pass
# through transparently as raw bytes, which is what we want for a byte-level
# model).
#
# After we close the outermost value, we accept exactly one of: nothing more,
# or pure whitespace (space, tab, LF, CR). `done()` returns True the moment
# the outermost value closes, so a decoder that calls `done()` after each
# step will halt at the natural boundary. Whitespace acceptance after that
# is just a safety net for callers that ignore `done()`.
# ------------------------------------------------------------------------------------

_WS = {0x20, 0x09, 0x0a, 0x0d}
_HEX = set(b"0123456789abcdefABCDEF")
_DIGITS_NONZERO = set(b"123456789")
_DIGITS = set(b"0123456789")


# value_state codes
_VS_VALUE_START      = "value_start"
_VS_OBJ_AFTER_OPEN   = "obj_after_open"     # just saw '{', want '"' or '}'
_VS_OBJ_AFTER_KEY    = "obj_after_key"      # just closed a key, want ':'
_VS_OBJ_AFTER_COLON  = "obj_after_colon"    # want a value
_VS_OBJ_AFTER_VALUE  = "obj_after_value"    # want ',' or '}'
_VS_OBJ_AFTER_COMMA  = "obj_after_comma"    # want next '"' (a key)
_VS_ARR_AFTER_OPEN   = "arr_after_open"     # just saw '[', want value or ']'
_VS_ARR_AFTER_VALUE  = "arr_after_value"    # want ',' or ']'
_VS_ARR_AFTER_COMMA  = "arr_after_comma"    # want next value
_VS_DONE             = "done"


class JSONConstraint(Constraint):
    """Streaming JSON grammar mask.

    Tracks the JSON state machine byte-by-byte. `mask()` returns the set of
    bytes whose acceptance leaves us in a state from which valid JSON is
    still reachable. `done()` flips True after the outermost value closes.

    Call `prime(bytes)` to feed an existing prompt prefix into the state
    machine before generation starts. This lets a user prompt the model
    with `{"name": "` and have the constraint pick up mid-string.
    """

    def __init__(self):
        self.value_state: str = _VS_VALUE_START
        self.stack:       List[str] = []
        self.in_string:   bool = False
        # esc states (we don't overload one int):
        #   esc_pending = True iff we just saw '\' and are about to consume
        #                 one of "\\/bfnrtu (single-char) or 'u' (-> 4 hex).
        #   hex_remaining = 0..4. If > 0, we're inside a \uXXXX escape and
        #                   must consume that many more hex digits.
        self.esc_pending:   bool = False
        self.hex_remaining: int  = 0
        self.number_state: Optional[str] = None
        self.literal:     Optional[bytes] = None  # bytes still to match
        self._done:       bool = False

    # ------------------------------------------------------------------ priming

    def prime(self, prefix: bytes) -> None:
        """Feed a prompt prefix through the state machine without using
        mask() (we trust the prompt). Useful when the model is given a JSON
        prefix to complete."""
        for b in prefix:
            self._step_internal(b, validate=False)

    # ------------------------------------------------------------------ API

    def mask(self) -> np.ndarray:
        m = _empty_mask()

        if self._done:
            # Outermost value is closed. Allow trailing whitespace only.
            for w in _WS:
                m[w] = True
            return m

        if self.literal is not None:
            # We're partway through true/false/null; only the next char of
            # the literal is allowed.
            m[self.literal[0]] = True
            return m

        if self.in_string:
            return self._mask_string()

        if self.number_state is not None:
            return self._mask_number()

        # Otherwise we're in a structural state. Whitespace is always allowed
        # in these states (RFC-8259 ws between tokens).
        for w in _WS:
            m[w] = True

        vs = self.value_state
        if vs == _VS_VALUE_START or vs == _VS_OBJ_AFTER_COLON or vs == _VS_ARR_AFTER_COMMA:
            self._allow_value_start(m)
        elif vs == _VS_OBJ_AFTER_OPEN:
            m[ord('"')] = True
            m[ord('}')] = True
        elif vs == _VS_OBJ_AFTER_KEY:
            m[ord(':')] = True
        elif vs == _VS_OBJ_AFTER_VALUE:
            m[ord(',')] = True
            m[ord('}')] = True
        elif vs == _VS_OBJ_AFTER_COMMA:
            m[ord('"')] = True
        elif vs == _VS_ARR_AFTER_OPEN:
            self._allow_value_start(m)
            m[ord(']')] = True
        elif vs == _VS_ARR_AFTER_VALUE:
            m[ord(',')] = True
            m[ord(']')] = True
        else:
            raise AssertionError(f"unreachable value_state {vs!r}")

        return m

    def step(self, byte: int) -> None:
        self._step_internal(byte & 0xff, validate=True)

    def done(self) -> bool:
        # We report done() right when the outermost value closes (or when a
        # toplevel number is followed by something other than a digit / .eE).
        # For a streaming decoder, halting here is the right call.
        return self._done

    def reset(self) -> None:
        self.value_state = _VS_VALUE_START
        self.stack.clear()
        self.in_string = False
        self.esc_pending = False
        self.hex_remaining = 0
        self.number_state = None
        self.literal = None
        self._done = False

    # ------------------------------------------------------------------ helpers

    def _allow_value_start(self, m: np.ndarray) -> None:
        """Bytes that may start a fresh JSON value."""
        m[ord('"')] = True
        m[ord('{')] = True
        m[ord('[')] = True
        m[ord('-')] = True
        for d in _DIGITS:
            m[d] = True
        # Start of literals true/false/null.
        m[ord('t')] = True
        m[ord('f')] = True
        m[ord('n')] = True

    def _mask_string(self) -> np.ndarray:
        m = _empty_mask()
        if self.hex_remaining > 0:
            # Inside \uXXXX -- need a hex digit.
            for c in _HEX:
                m[c] = True
            return m
        if self.esc_pending:
            # Just saw '\'. Allowed escape chars.
            for c in b'"\\/bfnrtu':
                m[c] = True
            return m
        # Plain string body. Allow any byte >= 0x20 (incl. UTF-8 multi-byte).
        for b in range(256):
            if b < 0x20:
                continue
            m[b] = True
        return m

    def _mask_number(self) -> np.ndarray:
        m = _empty_mask()
        ns = self.number_state
        if ns == 'sign':
            # Just consumed '-'. Need 0 or 1-9.
            for d in _DIGITS:
                m[d] = True
            return m
        if ns == 'int_zero':
            # An unambiguous '0' (or "-0"). Can be followed by:
            # '.', 'e', 'E', or by anything that ENDS the number.
            m[ord('.')] = True
            m[ord('e')] = True
            m[ord('E')] = True
            self._allow_number_terminators(m)
            return m
        if ns == 'int_nonzero':
            for d in _DIGITS:
                m[d] = True
            m[ord('.')] = True
            m[ord('e')] = True
            m[ord('E')] = True
            self._allow_number_terminators(m)
            return m
        if ns == 'frac_dot':
            # Just consumed '.'. Need at least one digit.
            for d in _DIGITS:
                m[d] = True
            return m
        if ns == 'frac_digits':
            for d in _DIGITS:
                m[d] = True
            m[ord('e')] = True
            m[ord('E')] = True
            self._allow_number_terminators(m)
            return m
        if ns == 'exp_sign':
            # Just consumed 'e'/'E'. Allow '+', '-', or a digit.
            m[ord('+')] = True
            m[ord('-')] = True
            for d in _DIGITS:
                m[d] = True
            return m
        if ns == 'exp_digits':
            for d in _DIGITS:
                m[d] = True
            self._allow_number_terminators(m)
            return m
        raise AssertionError(f"unknown number_state {ns!r}")

    def _allow_number_terminators(self, m: np.ndarray) -> None:
        """Bytes that legally END the current number."""
        # Whitespace always terminates.
        for w in _WS:
            m[w] = True
        # If we're inside a container, ',' and the matching close bracket
        # also terminate. If we're at toplevel, EOF terminates -- there are
        # no extra bytes we can produce.
        if self.stack:
            top = self.stack[-1]
            m[ord(',')] = True
            m[ord('}') if top == "obj" else ord(']')] = True

    # ------------------------------------------------------------------ step impl

    def _step_internal(self, b: int, validate: bool) -> None:
        # Note: when validate=True we trust mask() has already enforced
        # legality and just update state. When validate=False (priming
        # from a prompt) we do the same updates without checking.

        if self._done:
            # Whitespace consumes harmlessly.
            if b in _WS:
                return
            # Anything else is technically invalid (no value is reachable),
            # but we just stay done.
            return

        if self.literal is not None:
            # Consuming a true/false/null character.
            expected = self.literal[0]
            self.literal = self.literal[1:] if len(self.literal) > 1 else None
            if self.literal is None:
                self._close_value()
            return

        if self.in_string:
            self._step_in_string(b)
            return

        if self.number_state is not None:
            consumed = self._step_in_number(b)
            if consumed:
                return
            # Number terminated by `b`; fall through to process b as a
            # structural byte.
            self._close_value()
            if self._done:
                # Toplevel number just closed. The byte `b` is whitespace
                # (only legal terminator at toplevel per our mask).
                return

        # Structural byte.
        if b in _WS:
            return

        vs = self.value_state
        if vs == _VS_VALUE_START:
            self._start_value(b, is_toplevel=True)
            return
        if vs == _VS_OBJ_AFTER_OPEN:
            if b == ord('"'):
                self.in_string = True
                self.value_state = "obj_in_key"  # transient; we'll route on string close
                return
            if b == ord('}'):
                self._pop_container()
                return
            return
        if vs == "obj_in_key":
            # Should not be reached -- in_string=True consumes here.
            return
        if vs == _VS_OBJ_AFTER_KEY:
            if b == ord(':'):
                self.value_state = _VS_OBJ_AFTER_COLON
            return
        if vs == _VS_OBJ_AFTER_COLON:
            self._start_value(b, is_toplevel=False, parent="obj")
            return
        if vs == _VS_OBJ_AFTER_VALUE:
            if b == ord(','):
                self.value_state = _VS_OBJ_AFTER_COMMA
                return
            if b == ord('}'):
                self._pop_container()
            return
        if vs == _VS_OBJ_AFTER_COMMA:
            if b == ord('"'):
                self.in_string = True
                self.value_state = "obj_in_key"
            return
        if vs == _VS_ARR_AFTER_OPEN:
            if b == ord(']'):
                self._pop_container()
                return
            self._start_value(b, is_toplevel=False, parent="arr")
            return
        if vs == _VS_ARR_AFTER_VALUE:
            if b == ord(','):
                self.value_state = _VS_ARR_AFTER_COMMA
                return
            if b == ord(']'):
                self._pop_container()
            return
        if vs == _VS_ARR_AFTER_COMMA:
            self._start_value(b, is_toplevel=False, parent="arr")
            return

    # ------------------------------------------------------------------ string substates

    def _step_in_string(self, b: int) -> None:
        if self.hex_remaining > 0:
            # Consuming a hex digit of \uXXXX.
            self.hex_remaining -= 1
            return
        if self.esc_pending:
            self.esc_pending = False
            if b == ord('u'):
                self.hex_remaining = 4
                return
            # Single-char escape consumed; back to string body.
            return
        if b == ord('"'):
            self.in_string = False
            if self.value_state == "obj_in_key":
                self.value_state = _VS_OBJ_AFTER_KEY
            else:
                self._close_value()
            return
        if b == ord('\\'):
            self.esc_pending = True
            return
        # Other byte -- absorbed into the string. No state change.
        return

    # ------------------------------------------------------------------ number substates

    def _step_in_number(self, b: int) -> bool:
        """Try to consume `b` as part of the current number. Return True
        if consumed (state updated), False if `b` terminates the number
        (state unchanged; caller must process `b` as a structural byte)."""
        ns = self.number_state
        if ns == 'sign':
            if b == ord('0'):
                self.number_state = 'int_zero'
                return True
            if b in _DIGITS_NONZERO:
                self.number_state = 'int_nonzero'
                return True
            return False
        if ns == 'int_zero':
            if b == ord('.'):
                self.number_state = 'frac_dot'
                return True
            if b in (ord('e'), ord('E')):
                self.number_state = 'exp_sign'
                return True
            return False
        if ns == 'int_nonzero':
            if b in _DIGITS:
                return True
            if b == ord('.'):
                self.number_state = 'frac_dot'
                return True
            if b in (ord('e'), ord('E')):
                self.number_state = 'exp_sign'
                return True
            return False
        if ns == 'frac_dot':
            if b in _DIGITS:
                self.number_state = 'frac_digits'
                return True
            return False
        if ns == 'frac_digits':
            if b in _DIGITS:
                return True
            if b in (ord('e'), ord('E')):
                self.number_state = 'exp_sign'
                return True
            return False
        if ns == 'exp_sign':
            if b in (ord('+'), ord('-')):
                # Stay in exp_sign? No -- after the sign we MUST see digits.
                # We re-use exp_sign but the mask logic in _mask_number for
                # 'exp_sign' currently allows + - digits. After consuming a
                # sign, only digits are legal next. Switch to a dedicated
                # state.
                self.number_state = 'exp_sign_after'
                return True
            if b in _DIGITS:
                self.number_state = 'exp_digits'
                return True
            return False
        if ns == 'exp_sign_after':
            if b in _DIGITS:
                self.number_state = 'exp_digits'
                return True
            return False
        if ns == 'exp_digits':
            if b in _DIGITS:
                return True
            return False
        raise AssertionError(f"unknown number_state {ns!r}")

    # ------------------------------------------------------------------ structural helpers

    def _start_value(self, b: int, *, is_toplevel: bool, parent: Optional[str] = None) -> None:
        if b == ord('"'):
            self.in_string = True
            return
        if b == ord('{'):
            self.stack.append("obj")
            self.value_state = _VS_OBJ_AFTER_OPEN
            return
        if b == ord('['):
            self.stack.append("arr")
            self.value_state = _VS_ARR_AFTER_OPEN
            return
        if b == ord('-'):
            self.number_state = 'sign'
            return
        if b == ord('0'):
            self.number_state = 'int_zero'
            return
        if b in _DIGITS_NONZERO:
            self.number_state = 'int_nonzero'
            return
        if b == ord('t'):
            self.literal = b"rue"
            return
        if b == ord('f'):
            self.literal = b"alse"
            return
        if b == ord('n'):
            self.literal = b"ull"
            return
        # Should be unreachable when mask() is respected.

    def _close_value(self) -> None:
        """A value just finished. Update value_state per parent container."""
        self.number_state = None
        self.literal = None
        if not self.stack:
            self._done = True
            self.value_state = _VS_DONE
            return
        top = self.stack[-1]
        if top == "obj":
            self.value_state = _VS_OBJ_AFTER_VALUE
        else:
            self.value_state = _VS_ARR_AFTER_VALUE

    def _pop_container(self) -> None:
        if not self.stack:
            return
        self.stack.pop()
        self._close_value()


# Override _mask_number for the exp_sign_after sub-state (added in step impl).
# We patch the method dispatch by extending the if-chain:
_orig_mask_number = JSONConstraint._mask_number
def _mask_number_extended(self):
    if self.number_state == 'exp_sign_after':
        m = _empty_mask()
        for d in _DIGITS:
            m[d] = True
        return m
    return _orig_mask_number(self)
JSONConstraint._mask_number = _mask_number_extended
