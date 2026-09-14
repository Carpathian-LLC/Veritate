# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the agent's calculator evaluates model-written text, so what it refuses matters as
#   much as what it computes: no names, attributes, calls or syntax outside the whitelist,
#   no exponent that would stall the process, errors as strings and never raises.
# tests/mri/test_agent_calculator.py
# ------------------------------------------------------------------------------------
# Imports:

import time

import pytest
from inference.agent.tools import calculator

# ------------------------------------------------------------------------------------
# Functions


@pytest.mark.parametrize("expr, out", [
    ("2 + 3 * 4", "14"), ("(2 + 3) * 4", "20"), ("7 // 2", "3"), ("7 % 3", "1"), ("2 ** 10", "1024"),
    ("-3 + +2", "-1"), ("1 / 4", "0.25"), ("10 / 5", "2"), ("sqrt(16)", "4"), ("round(2.675, 2)", "2.67"),
    ("max(1, 9, 3)", "9"), ("sum([1, 2, 3])", "6"), ("floor(pi)", "3"), ("log(e)", "1"), ("min((4, 2))", "2"),
])
def test_arithmetic_and_the_whitelisted_functions(expr, out):
    assert calculator.evaluate(expr) == out


@pytest.mark.parametrize("expr, fragment", [
    ("__import__('os')", "unknown function"), ("(1).real", "disallowed expression node"),
    ("x + 1", "unknown name"), ("lambda: 1", "disallowed expression node"),
    ("open('f')", "unknown function"), ("round(1.5, ndigits=0)", "keyword arguments"),
    ("1 << 2", "unsupported binop"), ("'a' + 'b'", "unsupported constant"),
    ("1 / 0", "ZeroDivisionError"), ("2 +", "syntax"), ("", "empty expression"),
    ("[1][0]", "disallowed expression node"),
])
def test_everything_off_the_whitelist_is_an_error_string(expr, fragment):
    out = calculator.evaluate(expr)
    assert out.startswith("error: ") and fragment in out


def test_a_huge_exponent_is_refused_quickly():
    t0 = time.monotonic()
    out = calculator.evaluate("9 ** 9 ** 9")
    assert out.startswith("error: ") and "exponent" in out
    assert time.monotonic() - t0 < 1.0
    assert calculator.evaluate("2 ** 1000").startswith("10715086")


def test_length_and_type_guards():
    assert "too long" in calculator.evaluate("1+" * 600 + "1")
    assert "must be string" in calculator.evaluate(42)


def test_the_tool_contract_reports_a_missing_argument_and_never_raises():
    assert calculator.TOOL.call({}) == "error: missing required arg 'expression'"
    assert calculator.TOOL.call({"expression": "6 * 7"}) == "42"
    assert calculator.TOOL.name == "calculator"
