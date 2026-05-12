"""Static AST scanner for ``model.py`` files.

Extracts a structural summary of every ``nn.Module`` subclass and
top-level helper function in a target file:

  * class / function name and base classes,
  * ``__init__`` parameters with default-value source,
  * ``self.X = expr`` assignments collected from ``__init__``,
  * the high-level callables invoked inside ``forward`` (e.g. ``F.linear``,
    ``torch.matmul``, ``apply_rotary_emb``),
  * the ``self.<attr>`` chains called inside ``forward`` (which submodule
    is wired up to which),
  * whether ``forward`` issues any ``dist.*`` collective.

The scanner does **not** import the file — it relies on ``ast.parse`` so
it works on partially-broken sources (missing kernel modules etc.) and
keeps the discover skill safe to run in CI without a full PyTorch /
HuggingFace environment.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class AstClassInfo:
    """Summary of one ``nn.Module`` subclass (or top-level function).

    For top-level functions, ``bases`` is empty, ``is_nn_module`` is False
    and ``init_*`` are also empty; only ``forward_calls`` /
    ``forward_self_calls`` / ``has_dist_call`` are populated from the
    function body.
    """

    name: str
    bases: list[str] = field(default_factory=list)
    is_nn_module: bool = False
    init_params: dict[str, Optional[str]] = field(default_factory=dict)
    init_attrs: dict[str, str] = field(default_factory=dict)
    forward_calls: list[str] = field(default_factory=list)
    forward_self_calls: list[str] = field(default_factory=list)
    has_dist_call: bool = False
    docstring: str = ""


@dataclass
class AstScanResult:
    classes: list[AstClassInfo] = field(default_factory=list)
    top_level_funcs: list[AstClassInfo] = field(default_factory=list)
    file_path: str = ""


# ─── Public API ───────────────────────────────────────────────────────────────

def scan_model_file(path: str) -> AstScanResult:
    """Parse *path* and return the static summary."""
    p = Path(path)
    src = p.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(p))

    result = AstScanResult(file_path=str(p))

    # First pass: collect class names so subclasses can chain
    # `is_nn_module` through user-defined intermediates.
    local_class_names: set[str] = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }
    nn_module_classes: set[str] = set()  # filled progressively below

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            info = _scan_classdef(node, src, nn_module_classes,
                                  local_class_names)
            if info.is_nn_module:
                nn_module_classes.add(info.name)
            result.classes.append(info)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info = _scan_top_func(node, src)
            result.top_level_funcs.append(info)

    return result


# ─── Class-level scan ─────────────────────────────────────────────────────────

def _scan_classdef(
    node: ast.ClassDef,
    src: str,
    nn_module_classes: set[str],
    local_class_names: set[str],
) -> AstClassInfo:
    bases = [_format_base(b) for b in node.bases]
    is_nn = _is_nn_module(bases, nn_module_classes)
    docstring = (ast.get_docstring(node) or "")[:200]

    info = AstClassInfo(
        name=node.name,
        bases=bases,
        is_nn_module=is_nn,
        docstring=docstring,
    )

    for child in node.body:
        if isinstance(child, ast.FunctionDef):
            if child.name == "__init__":
                info.init_params = _extract_init_params(child, src)
                info.init_attrs = _extract_self_assigns(child, src)
            elif child.name == "forward":
                calls, self_calls, has_dist = _scan_forward(child)
                info.forward_calls = calls
                info.forward_self_calls = self_calls
                info.has_dist_call = has_dist

    return info


def _scan_top_func(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    src: str,
) -> AstClassInfo:
    """Treat top-level functions like ``sparse_attn`` as fusion candidates."""
    docstring = (ast.get_docstring(node) or "")[:200]
    calls, self_calls, has_dist = _scan_forward(node)
    return AstClassInfo(
        name=node.name,
        bases=[],
        is_nn_module=False,
        init_params={},
        init_attrs={},
        forward_calls=calls,
        forward_self_calls=self_calls,
        has_dist_call=has_dist,
        docstring=docstring,
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _format_base(node: ast.expr) -> str:
    """Stringify a base-class AST node (Name / Attribute / Subscript)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_format_base(node.value)}.{node.attr}"
    # ast.unparse exists since 3.9; fall back to a textual approximation
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


_NN_MODULE_BASES = {"nn.Module", "Module", "torch.nn.Module", "nn.modules.Module"}


def _is_nn_module(bases: list[str], known_nn_modules: set[str]) -> bool:
    for b in bases:
        if b in _NN_MODULE_BASES:
            return True
        # Inheritance from another already-scanned nn.Module subclass
        # (e.g. ``ColumnParallelLinear(Linear)``).
        if b in known_nn_modules:
            return True
        # Tail name often suffices for dotted bases
        tail = b.rsplit(".", 1)[-1]
        if tail == "Module" or tail in known_nn_modules:
            return True
    return False


def _extract_init_params(
    fn: ast.FunctionDef, src: str
) -> dict[str, Optional[str]]:
    """Map ``__init__`` parameter names → default-value source (or None).

    Skips ``self``.  For ``*args`` / ``**kwargs`` we record the bare name
    without a default.
    """
    args = fn.args
    out: dict[str, Optional[str]] = {}

    pos_args = list(args.posonlyargs) + list(args.args)
    pos_defaults = list(args.defaults)
    pad = len(pos_args) - len(pos_defaults)
    for i, a in enumerate(pos_args):
        if a.arg == "self":
            continue
        default_node = (
            pos_defaults[i - pad] if i >= pad and (i - pad) < len(pos_defaults) else None
        )
        out[a.arg] = _src_segment(default_node, src) if default_node is not None else None

    if args.vararg is not None:
        out[f"*{args.vararg.arg}"] = None

    for kwarg, default_node in zip(args.kwonlyargs, args.kw_defaults):
        out[kwarg.arg] = (
            _src_segment(default_node, src) if default_node is not None else None
        )

    if args.kwarg is not None:
        out[f"**{args.kwarg.arg}"] = None

    return out


def _src_segment(node: Optional[ast.AST], src: str) -> Optional[str]:
    if node is None:
        return None
    seg = ast.get_source_segment(src, node)
    if seg is not None:
        return seg
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _extract_self_assigns(fn: ast.FunctionDef, src: str) -> dict[str, str]:
    """Collect ``self.X = expr`` (and tuple targets) from the function body."""
    out: dict[str, str] = {}
    for stmt in ast.walk(fn):
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                _record_self_target(tgt, stmt.value, src, out)
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            _record_self_target(stmt.target, stmt.value, src, out)
    return out


def _record_self_target(
    tgt: ast.AST, value: ast.AST, src: str, out: dict[str, str]
) -> None:
    if (
        isinstance(tgt, ast.Attribute)
        and isinstance(tgt.value, ast.Name)
        and tgt.value.id == "self"
    ):
        expr_src = _src_segment(value, src)
        if expr_src is not None:
            out.setdefault(tgt.attr, expr_src)
    elif isinstance(tgt, (ast.Tuple, ast.List)):
        # ``self.a, self.b = foo()`` → record both with the same value src
        for elt in tgt.elts:
            _record_self_target(elt, value, src, out)


def _scan_forward(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[list[str], list[str], bool]:
    """Extract call summary from a forward-like function body."""
    calls: list[str] = []
    self_calls: list[str] = []
    has_dist = False
    seen_calls: set[str] = set()
    seen_self: set[str] = set()

    for stmt in ast.walk(fn):
        if not isinstance(stmt, ast.Call):
            continue
        func = stmt.func

        # ``self.X(...)`` or ``self.X.Y(...)``
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            tag = f"self.{func.attr}"
            if tag not in seen_self:
                seen_self.add(tag)
                self_calls.append(tag)
            continue

        # ``self.X.Y(...)``  → record both ``self.X`` (submodule access)
        # and ``self.X.Y`` (method call on submodule).
        if isinstance(func, ast.Attribute) and isinstance(
            func.value, ast.Attribute
        ):
            base_chain = _chain(func.value)
            if base_chain.startswith("self."):
                if base_chain not in seen_self:
                    seen_self.add(base_chain)
                    self_calls.append(base_chain)
                # skip recording the full chain into ``calls``
                continue

        full = _format_callable(func)
        if not full:
            continue

        # ``dist.<anything>`` → collective communication
        if full.startswith("dist."):
            has_dist = True

        if full not in seen_calls:
            seen_calls.add(full)
            calls.append(full)

    return calls, self_calls, has_dist


def _chain(node: ast.expr) -> str:
    """Build a dotted chain ``a.b.c`` from a nested Attribute / Name."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_chain(node.value)}.{node.attr}"
    return ""


def _format_callable(func: ast.expr) -> str:
    """Format the ``func`` of an ast.Call into a short textual key.

    Examples:
        ``F.linear``       → "F.linear"
        ``torch.matmul``   → "torch.matmul"
        ``linear``         → "linear"
        ``x.softmax``      → "softmax"        (method on local var)
    """
    if isinstance(func, ast.Name):
        return func.id

    if isinstance(func, ast.Attribute):
        # Accept up to a 2-level dotted base (matches F.linear / torch.matmul).
        base = func.value
        if isinstance(base, ast.Name):
            return f"{base.id}.{func.attr}"
        if isinstance(base, ast.Attribute):
            chain = _chain(base)
            if chain.count(".") <= 2:
                return f"{chain}.{func.attr}"
        # Method call on a local variable / expression — bare attr name
        return func.attr

    # Anonymous / complex callables — best-effort unparse
    try:
        return ast.unparse(func)
    except Exception:
        return ""
