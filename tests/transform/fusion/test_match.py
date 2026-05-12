"""Tests for the three-tier fusion-rule matcher.

Contract: see ``docs/fusion_v2_rich_rules_zh.md`` §2.1.
"""
from __future__ import annotations

import pytest

from python.zrt.transform.fusion.core.io_role import IORole
from python.zrt.transform.fusion.core.pattern import DEFAULT_SKIP_OPS, MatchPattern
from python.zrt.transform.fusion.core.rule import ModuleFusionRule
from python.zrt.transform.fusion.matching.matcher import best_rule, match_group


# ─────────────────────────────────────────────────────────────────────────────
# class_only
# ─────────────────────────────────────────────────────────────────────────────

class _Compressor:  # standalone class for type-based target_class
    pass


def test_class_only_basic_equality():
    """Plain string target_class matches identical module_class."""
    pat = MatchPattern(kind="class_only")
    assert match_group(
        ("aten.mm.default", "aten.add.Tensor"),
        "Compressor",
        pattern=pat,
        target_class="Compressor",
    ) is True
    # Mismatch must fail.
    assert match_group(
        ("aten.mm.default",),
        "OtherModule",
        pattern=pat,
        target_class="Compressor",
    ) is False


def test_class_only_tuple_alternatives():
    """Tuple target_class matches if any candidate matches."""
    pat = MatchPattern(kind="class_only")
    target = ("RMSNorm", "LlamaRMSNorm", "Qwen2RMSNorm")
    for cls in target:
        assert match_group(
            ("aten.pow.Tensor_Scalar",),
            cls,
            pattern=pat,
            target_class=target,
        ) is True
    # Outside the tuple → no match.
    assert match_group(
        ("aten.pow.Tensor_Scalar",),
        "MistralRMSNorm",
        pattern=pat,
        target_class=target,
    ) is False


def test_class_only_string_regex():
    """A regex string target matches multiple concrete class names.

    ``re.fullmatch`` is used, so the regex must cover the whole name —
    ``Llama.*RMSNorm.*`` matches both ``LlamaRMSNorm`` and ``LlamaRMSNorm2``,
    while ``Qwen2RMSNorm`` (different prefix) is rejected.
    """
    pat = MatchPattern(kind="class_only")
    regex = r"Llama.*RMSNorm.*"
    assert match_group(
        ("aten.mul.Tensor",),
        "LlamaRMSNorm",
        pattern=pat,
        target_class=regex,
    ) is True
    assert match_group(
        ("aten.mul.Tensor",),
        "LlamaRMSNorm2",
        pattern=pat,
        target_class=regex,
    ) is True
    # Non-matching prefix.
    assert match_group(
        ("aten.mul.Tensor",),
        "Qwen2RMSNorm",
        pattern=pat,
        target_class=regex,
    ) is False


def test_class_only_size_bounds():
    """min_ops / max_ops still apply for class_only."""
    pat = MatchPattern(kind="class_only", min_ops=2, max_ops=4)
    assert match_group(
        ("aten.mm.default",),
        "Foo",
        pattern=pat,
        target_class="Foo",
    ) is False  # too few
    assert match_group(
        ("aten.mm.default",) * 3,
        "Foo",
        pattern=pat,
        target_class="Foo",
    ) is True
    assert match_group(
        ("aten.mm.default",) * 5,
        "Foo",
        pattern=pat,
        target_class="Foo",
    ) is False  # too many


# ─────────────────────────────────────────────────────────────────────────────
# ordered_regex
# ─────────────────────────────────────────────────────────────────────────────

def _rms_pattern(min_ops: int = 6) -> MatchPattern:
    """Realistic RMSNorm op-regex sequence used by several tests below."""
    return MatchPattern(
        kind="ordered_regex",
        op_regexes=(
            r"aten\.pow\.Tensor_Scalar",
            r"aten\.mean\.dim",
            r"aten\.add\.(Tensor|Scalar)",
            r"aten\.rsqrt\.default",
            r"aten\.mul\.Tensor",
            r"aten\.mul\.Tensor",
        ),
        min_ops=min_ops,
    )


def test_ordered_regex_perfect_sequence():
    """Exact in-order matching with no extra ops succeeds."""
    pat = _rms_pattern()
    ops = (
        "aten.pow.Tensor_Scalar",
        "aten.mean.dim",
        "aten.add.Tensor",
        "aten.rsqrt.default",
        "aten.mul.Tensor",
        "aten.mul.Tensor",
    )
    assert match_group(ops, "LlamaRMSNorm",
                       pattern=pat, target_class="LlamaRMSNorm") is True


def test_ordered_regex_skip_ops_views_permutes():
    """view / permute / dtype-cast interleaved ops should be skipped."""
    pat = _rms_pattern()
    ops = (
        "aten.pow.Tensor_Scalar",
        "aten.view.default",          # skipped
        "aten.mean.dim",
        "aten._to_copy.default",      # skipped
        "aten.add.Tensor",
        "aten.rsqrt.default",
        "aten.permute.default",       # skipped
        "aten.mul.Tensor",
        "aten.mul.Tensor",
    )
    assert match_group(ops, "LlamaRMSNorm",
                       pattern=pat, target_class="LlamaRMSNorm") is True


def test_ordered_regex_min_ops_not_satisfied():
    """If actual op count < min_ops, match fails even if regexes line up."""
    pat = MatchPattern(
        kind="ordered_regex",
        op_regexes=(r"aten\.mm\.default",),
        min_ops=3,
    )
    ops = ("aten.mm.default",)
    assert match_group(ops, "Linear",
                       pattern=pat, target_class="Linear") is False


def test_ordered_regex_trailing_extra_op_fails():
    """A trailing non-skip op after last regex must cause failure."""
    pat = MatchPattern(
        kind="ordered_regex",
        op_regexes=(r"aten\.mm\.default", r"aten\.add\.Tensor"),
        min_ops=2,
    )
    # Trailing aten.relu.default is NOT in skip_ops → fail.
    ops = (
        "aten.mm.default",
        "aten.add.Tensor",
        "aten.relu.default",
    )
    assert match_group(ops, "Linear",
                       pattern=pat, target_class="Linear") is False
    # Same trailing op but in skip_ops → ok.
    ops_with_skip = (
        "aten.mm.default",
        "aten.add.Tensor",
        "aten.view.default",  # in DEFAULT_SKIP_OPS
    )
    assert match_group(ops_with_skip, "Linear",
                       pattern=pat, target_class="Linear") is True


# ─────────────────────────────────────────────────────────────────────────────
# dag_signature
# ─────────────────────────────────────────────────────────────────────────────

def test_dag_signature_perfect_match():
    """Exact multiset match (in any order) succeeds."""
    pat = MatchPattern(
        kind="dag_signature",
        op_multiset=(
            (r"aten\.mm\.default|aten\.matmul\..*", 1),
            (r"fp8_gemm|fp4_gemm|act_quant", 1),
        ),
        min_ops=2,
    )
    ops = ("act_quant", "aten.mm.default")
    assert match_group(ops, "Linear",
                       pattern=pat, target_class="Linear") is True


def test_dag_signature_min_count_not_satisfied():
    """If a regex's count < min_count, match fails."""
    pat = MatchPattern(
        kind="dag_signature",
        op_multiset=(
            (r"aten\.mul\.Tensor", 3),
        ),
        min_ops=1,
    )
    ops = ("aten.mul.Tensor", "aten.add.Tensor", "aten.mul.Tensor")
    # Only 2 mul.Tensor → less than 3 → fail.
    assert match_group(ops, "Foo",
                       pattern=pat, target_class="Foo") is False
    # 3 mul.Tensor → ok.
    ops_ok = ("aten.mul.Tensor", "aten.mul.Tensor", "aten.mul.Tensor")
    assert match_group(ops_ok, "Foo",
                       pattern=pat, target_class="Foo") is True


def test_dag_signature_unordered():
    """Multiset match is order-independent."""
    pat = MatchPattern(
        kind="dag_signature",
        op_multiset=(
            (r"aten\.mm\.default", 1),
            (r"aten\.add\.Tensor", 1),
            (r"aten\.relu\.default", 1),
        ),
        min_ops=3,
    )
    # All three permutations should match.
    perms = [
        ("aten.mm.default", "aten.add.Tensor", "aten.relu.default"),
        ("aten.relu.default", "aten.mm.default", "aten.add.Tensor"),
        ("aten.add.Tensor", "aten.relu.default", "aten.mm.default"),
    ]
    for ops in perms:
        assert match_group(ops, "Linear",
                           pattern=pat, target_class="Linear") is True


# ─────────────────────────────────────────────────────────────────────────────
# best_rule
# ─────────────────────────────────────────────────────────────────────────────

def test_best_rule_higher_priority_wins():
    """Among matching rules the highest priority is returned."""
    low = ModuleFusionRule(
        target_class="RMSNorm",
        op_type="rms_norm_low",
        pattern=MatchPattern(kind="class_only"),
        priority=10,
    )
    high = ModuleFusionRule(
        target_class="RMSNorm",
        op_type="rms_norm_high",
        pattern=MatchPattern(kind="class_only"),
        priority=30,
    )
    mid = ModuleFusionRule(
        target_class="RMSNorm",
        op_type="rms_norm_mid",
        pattern=MatchPattern(kind="class_only"),
        priority=20,
    )
    chosen = best_rule(("aten.mul.Tensor",), "RMSNorm", [low, high, mid])
    assert chosen is high
    assert chosen.op_type == "rms_norm_high"


def test_best_rule_tie_breaks_by_list_order():
    """When priorities tie, the earlier-registered rule wins."""
    first = ModuleFusionRule(
        target_class="Linear",
        op_type="linear_first",
        pattern=MatchPattern(kind="class_only"),
        priority=15,
    )
    second = ModuleFusionRule(
        target_class="Linear",
        op_type="linear_second",
        pattern=MatchPattern(kind="class_only"),
        priority=15,
    )
    chosen = best_rule(("aten.mm.default",), "Linear", [first, second])
    assert chosen is first

    # Reverse the list to confirm "list order" is the tiebreaker, not name.
    chosen_rev = best_rule(("aten.mm.default",), "Linear", [second, first])
    assert chosen_rev is second


def test_best_rule_returns_none_when_no_match():
    """Sanity check: no match → None."""
    rule = ModuleFusionRule(
        target_class="RMSNorm",
        pattern=MatchPattern(kind="class_only"),
        priority=10,
    )
    assert best_rule(("aten.mm.default",), "Linear", [rule]) is None
