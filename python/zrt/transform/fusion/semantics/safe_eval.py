"""Sandboxed AST evaluator for shape / flops / memory formulas.

Allowed:
  - Int / float literals
  - Names from the provided namespace
  - One-level attribute access: ``x.shape``, ``x.dtype``, ``x.bytes``,
    ``x.numel``, ``x.itemsize``
  - Subscript on attribute: ``x.shape[0]``, ``x.shape[-1]``
  - Binary ops: ``+ - * / // % **``
  - Unary ops: ``+ -``
  - Compare ops: ``< <= > >= == !=`` (returns 0/1)
  - Boolean ops: ``and or not`` (used as guards in min/max)
  - Calls to a small whitelist: ``min, max, abs, ceil, floor, log, log2, sqrt, int, float``
  - Conditional expression: ``a if cond else b``
  - Tuple / list literals (used by ``min(a, b)`` style)

Disallowed (raises ``FormulaError``):
  - Imports, lambdas, comprehensions, function/class defs
  - Multi-level attribute (``x.shape.foo``)
  - Calls to non-whitelisted functions
  - Augmented assignment, walrus, await, yield, starred, fstring with calls
"""
from __future__ import annotations

import ast
import math
from typing import Any


class FormulaError(ValueError):
    """Raised when a formula is malformed or uses disallowed constructs."""


_ALLOWED_ATTRS: frozenset[str] = frozenset({
    "shape", "dtype", "bytes", "numel", "itemsize",
})

_ALLOWED_FUNCS: dict[str, Any] = {
    "min": min, "max": max, "abs": abs,
    "ceil": math.ceil, "floor": math.floor,
    "log": math.log, "log2": math.log2,
    "sqrt": math.sqrt,
    "int": int, "float": float,
}

_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_CMP_OPS = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


def safe_eval(expr: str, namespace: dict[str, Any]) -> Any:
    """Evaluate *expr* against *namespace* under the AST sandbox.

    Returns whatever the expression evaluates to (typically ``int`` or
    ``float``).  Raises :class:`FormulaError` for any malformed input or
    disallowed AST construct.
    """
    if not isinstance(expr, str):
        raise FormulaError(f"expr must be str, got {type(expr).__name__}")
    expr = expr.strip()
    if not expr:
        raise FormulaError("empty expression")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"syntax error in {expr!r}: {e}") from e

    return _eval(tree.body, namespace)


def _eval(node: ast.AST, ns: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise FormulaError(f"unsupported constant: {node.value!r}")

    if isinstance(node, ast.Name):
        if node.id in ns:
            return ns[node.id]
        if node.id in _ALLOWED_FUNCS:
            return _ALLOWED_FUNCS[node.id]
        raise FormulaError(f"unknown name: {node.id}")

    if isinstance(node, ast.Attribute):
        # Only one level: x.attr where x is a Name.
        if not isinstance(node.value, ast.Name):
            raise FormulaError("only one-level attribute access allowed")
        if node.attr not in _ALLOWED_ATTRS:
            raise FormulaError(f"attribute {node.attr!r} not allowed")
        obj = _eval(node.value, ns)
        return getattr(obj, node.attr)

    if isinstance(node, ast.Subscript):
        # Only x.attr[idx] or x[idx] where x is Name/Attribute.
        target = _eval(node.value, ns)
        if isinstance(node.slice, ast.Slice):
            lo = _eval(node.slice.lower, ns) if node.slice.lower else None
            hi = _eval(node.slice.upper, ns) if node.slice.upper else None
            st = _eval(node.slice.step, ns) if node.slice.step else None
            return target[lo:hi:st]
        idx = _eval(node.slice, ns)
        return target[idx]

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise FormulaError(f"binary op {op_type.__name__} not allowed")
        return _BIN_OPS[op_type](_eval(node.left, ns), _eval(node.right, ns))

    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, ns)
        if isinstance(node.op, ast.UAdd):
            return +v
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.Not):
            return not v
        raise FormulaError(f"unary op {type(node.op).__name__} not allowed")

    if isinstance(node, ast.Compare):
        # Chain: a < b < c — evaluate left-to-right.
        left = _eval(node.left, ns)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval(comparator, ns)
            op_type = type(op)
            if op_type not in _CMP_OPS:
                raise FormulaError(f"compare op {op_type.__name__} not allowed")
            if not _CMP_OPS[op_type](left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            v: Any = True
            for sub in node.values:
                v = _eval(sub, ns)
                if not v:
                    return v
            return v
        if isinstance(node.op, ast.Or):
            v = False
            for sub in node.values:
                v = _eval(sub, ns)
                if v:
                    return v
            return v
        raise FormulaError(f"bool op {type(node.op).__name__} not allowed")

    if isinstance(node, ast.IfExp):
        cond = _eval(node.test, ns)
        return _eval(node.body, ns) if cond else _eval(node.orelse, ns)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise FormulaError("only direct calls to whitelisted functions allowed")
        fn_name = node.func.id
        fn = _ALLOWED_FUNCS.get(fn_name)
        if fn is None:
            raise FormulaError(f"function {fn_name!r} not allowed")
        if node.keywords:
            raise FormulaError(f"keyword args not allowed in call to {fn_name}")
        args = [_eval(a, ns) for a in node.args]
        return fn(*args)

    if isinstance(node, ast.Tuple):
        return tuple(_eval(e, ns) for e in node.elts)

    if isinstance(node, ast.List):
        return [_eval(e, ns) for e in node.elts]

    raise FormulaError(f"AST node {type(node).__name__} not allowed")
