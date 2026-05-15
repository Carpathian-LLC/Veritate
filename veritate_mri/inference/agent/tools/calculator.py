# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Calculator tool. Evaluates arithmetic + a small whitelisted math function set.
#   NEVER uses eval(); walks Python AST and rejects anything not on the
#   whitelist. Safe to expose to model-generated input.
# - Supported ops: + - * / // % ** unary-minus, parentheses.
# - Supported names: pi, e, inf, nan.
# - Supported calls: abs, round, min, max, sum, sqrt, log, log2, log10, exp,
#   sin, cos, tan, asin, acos, atan, floor, ceil.
# - Numbers: int + float literals only. No imaginary, no underscores.
# veritate_mri/agent/tools/calculator.py
# ------------------------------------------------------------------------------------
# Imports:

import ast
import math
import operator as op
from typing import Any, Dict

from . import Tool

# ------------------------------------------------------------------------------------
# Constants

_BINOPS = {
    ast.Add:      op.add,
    ast.Sub:      op.sub,
    ast.Mult:     op.mul,
    ast.Div:      op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod:      op.mod,
    ast.Pow:      op.pow,
}
_UNARYOPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}
_NAMES = {
    "pi":  math.pi,
    "e":   math.e,
    "inf": math.inf,
    "nan": math.nan,
}
_FUNCS = {
    "abs":   abs,
    "round": round,
    "min":   min,
    "max":   max,
    "sum":   sum,
    "sqrt":  math.sqrt,
    "log":   math.log,
    "log2":  math.log2,
    "log10": math.log10,
    "exp":   math.exp,
    "sin":   math.sin,
    "cos":   math.cos,
    "tan":   math.tan,
    "asin":  math.asin,
    "acos":  math.acos,
    "atan":  math.atan,
    "floor": math.floor,
    "ceil":  math.ceil,
}

_MAX_EXPRESSION_LEN = 1024

# ------------------------------------------------------------------------------------
# Functions


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported constant: {type(node.value).__name__}")
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.Name):
        v = _NAMES.get(node.id)
        if v is None:
            raise ValueError(f"unknown name: {node.id!r}")
        return v
    if isinstance(node, ast.UnaryOp):
        fn = _UNARYOPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"unsupported unary op: {type(node.op).__name__}")
        return fn(_eval(node.operand))
    if isinstance(node, ast.BinOp):
        fn = _BINOPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"unsupported binop: {type(node.op).__name__}")
        return fn(_eval(node.left), _eval(node.right))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("only direct function calls allowed")
        fn = _FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"unknown function: {node.func.id!r}")
        if node.keywords:
            raise ValueError("keyword arguments not allowed")
        args = [_eval(a) for a in node.args]
        return fn(*args)
    if isinstance(node, ast.Tuple):
        return tuple(_eval(e) for e in node.elts)
    if isinstance(node, ast.List):
        return [_eval(e) for e in node.elts]
    raise ValueError(f"disallowed expression node: {type(node).__name__}")


def evaluate(expression: str) -> str:
    """Evaluate a single arithmetic expression and return a string result.
    Returns 'error: ...' on failure; never raises."""
    if not isinstance(expression, str):
        return f"error: expression must be string, got {type(expression).__name__}"
    expression = expression.strip()
    if not expression:
        return "error: empty expression"
    if len(expression) > _MAX_EXPRESSION_LEN:
        return f"error: expression too long (>{_MAX_EXPRESSION_LEN} chars)"
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        return f"error: syntax: {e.msg}"
    try:
        result = _eval(tree)
    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
        return f"error: {type(e).__name__}: {e}"
    # Pretty-print floats compactly (no trailing zeros).
    if isinstance(result, float):
        if math.isfinite(result) and result == int(result):
            return str(int(result))
        return repr(result)
    return str(result)


def _execute(args: Dict[str, Any]) -> str:
    expression = args.get("expression")
    if expression is None:
        return "error: missing required arg 'expression'"
    return evaluate(expression)


TOOL = Tool(
    name="calculator",
    description="Evaluate a single arithmetic expression. Use for any numeric work.",
    args_schema={
        "expression": {
            "type": "string", "required": True,
            "doc": "Python arithmetic, e.g. '2 + 3 * 4', 'sqrt(2) * pi'. Functions: abs, round, min, max, sum, sqrt, log, log2, log10, exp, sin, cos, tan, asin, acos, atan, floor, ceil.",
        },
    },
    execute=_execute,
)
