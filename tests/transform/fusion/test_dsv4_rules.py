"""Validation tests for the DeepSeek-V4 fusion rule YAML.

Contract: ``docs/fusion_v2_rich_rules_zh.md`` §3 + §6.

These tests do not capture a real graph — they only assert that the
YAML file at ``python/zrt/transform/fusion/rules/deepseek_v4.yaml`` is
well-formed, that every rule exposes the required fields, and that
each rule's ``shape_derivation`` / ``flops_formula`` / ``memory_formula``
can be safely evaluated against mock ``TensorView`` objects.

If the test fixture model file changes line numbers, none of these
tests will fail unless a structural rule field becomes invalid.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from python.zrt.transform.fusion.core.io_role import IORole, ShapeDerivation
from python.zrt.transform.fusion.core.pattern import MatchPattern
from python.zrt.transform.fusion.core.rule import ModuleFusionRule
from python.zrt.transform.fusion.loading.yaml_rule_loader import (
    load_model_yaml_rules,
    load_yaml_rules,
)
from python.zrt.transform.fusion.registry import all_rules, clear_rules
from python.zrt.transform.fusion.semantics import TensorView
from python.zrt.transform.fusion.semantics.safe_eval import safe_eval


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

YAML_PATH = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "zrt"
    / "transform"
    / "fusion"
    / "rules"
    / "deepseek_v4.yaml"
)

EXPECTED_RULES = 16   # see task contract


def _mock_tv(shape, dtype="bf16", itemsize=2.0) -> TensorView:
    n = 1
    for d in shape:
        n *= d
    return TensorView(
        shape=tuple(shape),
        dtype=dtype,
        bytes=int(n * itemsize),
        numel=n,
        itemsize=itemsize,
    )


def _default_role_namespace() -> dict[str, TensorView]:
    """Provide a TensorView for every role name used across DSv4 rules."""
    return {
        "activation":     _mock_tv([2, 128, 4096]),
        "weight":         _mock_tv([2048, 4096]),
        "output":         _mock_tv([2, 128, 2048]),
        "input_ids":      _mock_tv([2, 128], dtype="int64", itemsize=8.0),
        "freqs_cis":      _mock_tv([128, 32], dtype="fp32", itemsize=4.0),
        "router_weight":  _mock_tv([8, 4096]),
    }


@pytest.fixture(scope="module")
def loaded_rules() -> list[ModuleFusionRule]:
    assert YAML_PATH.exists(), f"YAML file not found: {YAML_PATH}"
    return load_yaml_rules(YAML_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# 1. YAML loads without error
# ─────────────────────────────────────────────────────────────────────────────

def test_yaml_loads_without_error(loaded_rules):
    assert isinstance(loaded_rules, list)
    assert len(loaded_rules) >= EXPECTED_RULES, (
        f"expected at least {EXPECTED_RULES} rules, got {len(loaded_rules)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Every rule has the required fields
# ─────────────────────────────────────────────────────────────────────────────

def test_each_rule_has_required_fields(loaded_rules):
    for rule in loaded_rules:
        assert rule.target_class, f"missing target_class: {rule}"
        assert rule.op_type, f"missing op_type for {rule.target_class}"
        assert rule.pattern is not None, (
            f"missing pattern for {rule.target_class}/{rule.op_type}"
        )
        assert rule.io_roles, (
            f"missing io_roles for {rule.target_class}/{rule.op_type}"
        )
        # output role must be present
        assert any(io.role == "output" for io in rule.io_roles), (
            f"no output role declared for {rule.target_class}/{rule.op_type}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Priorities are sane (parallel linear > generic linear; class_only ≥ regex)
# ─────────────────────────────────────────────────────────────────────────────

def test_rules_have_priorities(loaded_rules):
    by_op = {r.op_type: r for r in loaded_rules}

    assert by_op["column_parallel_linear"].priority > by_op["linear"].priority
    assert by_op["row_parallel_linear"].priority > by_op["linear"].priority

    # Specialized class_only rules should outrank the generic Linear pattern.
    for op_type in ("rms_norm", "parallel_embedding", "kv_compressor",
                    "sparse_indexer", "mla_sparse_attn", "moe_dispatch",
                    "moe_gate", "hc_head"):
        assert by_op[op_type].priority >= by_op["linear"].priority, (
            f"{op_type} priority should be ≥ generic linear ({by_op['linear'].priority})"
        )

    # Every priority should be a positive int
    for r in loaded_rules:
        assert isinstance(r.priority, int) and r.priority > 0, (
            f"bad priority on {r.target_class}/{r.op_type}: {r.priority}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Each rule declares at least batch_size or seq_len in shape_derivation
#    (parallel_embedding is allowed to skip)
# ─────────────────────────────────────────────────────────────────────────────

def test_shape_derivation_well_formed(loaded_rules):
    embedding_exempt = {"parallel_embedding"}
    for rule in loaded_rules:
        sd = rule.shape_derivation
        assert sd is not None, (
            f"{rule.target_class}/{rule.op_type} missing shape_derivation"
        )
        if rule.op_type in embedding_exempt:
            continue
        assert sd.batch_size is not None or sd.seq_len is not None, (
            f"{rule.target_class}/{rule.op_type} must declare batch_size or seq_len"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. flops_formula evaluates safely → numeric ≥ 0
# ─────────────────────────────────────────────────────────────────────────────

def _eval_rule_formulas(rule: ModuleFusionRule) -> tuple[float, float]:
    """Run shape_derivation + flops/memory through safe_eval with mocks."""
    ns = _default_role_namespace()
    # Provide every declared role at least a default mock
    for io in rule.io_roles:
        ns.setdefault(io.role, _mock_tv([2, 128, 4096]))

    if rule.shape_derivation:
        for name, expr in rule.shape_derivation.items():
            ns[name] = safe_eval(expr, ns)

    flops = safe_eval(rule.flops_formula, ns) if rule.flops_formula else 0
    mem = safe_eval(rule.memory_formula, ns) if rule.memory_formula else 0
    return float(flops), float(mem)


def test_flops_formula_safe_eval_able(loaded_rules):
    for rule in loaded_rules:
        assert rule.flops_formula, (
            f"{rule.target_class}/{rule.op_type} missing flops_formula"
        )
        flops, _ = _eval_rule_formulas(rule)
        assert isinstance(flops, float)
        assert flops >= 0, (
            f"{rule.target_class}/{rule.op_type} flops < 0: {flops}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. memory_formula evaluates safely → numeric ≥ 0
# ─────────────────────────────────────────────────────────────────────────────

def test_memory_formula_safe_eval_able(loaded_rules):
    for rule in loaded_rules:
        assert rule.memory_formula, (
            f"{rule.target_class}/{rule.op_type} missing memory_formula"
        )
        _, mem = _eval_rule_formulas(rule)
        assert isinstance(mem, float)
        assert mem >= 0, (
            f"{rule.target_class}/{rule.op_type} memory < 0: {mem}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. class_only rules don't carry op_regexes / op_multiset
# ─────────────────────────────────────────────────────────────────────────────

def test_class_only_rules_dont_have_op_regexes(loaded_rules):
    for rule in loaded_rules:
        if rule.pattern.kind == "class_only":
            assert not rule.pattern.op_regexes, (
                f"class_only rule {rule.op_type} unexpectedly has op_regexes"
            )
            assert not rule.pattern.op_multiset, (
                f"class_only rule {rule.op_type} unexpectedly has op_multiset"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 8. No duplicate (target_class, op_type) signatures
#    (hc_pre / hc_post share Block but have distinct op_type)
# ─────────────────────────────────────────────────────────────────────────────

def test_no_duplicate_rule_signatures(loaded_rules):
    sigs = [(str(r.target_class), r.op_type) for r in loaded_rules]
    counts = Counter(sigs)
    # ``hc_pre`` legitimately has two rules: a raw 11-op form and a
    # post-rms_coef 8-op form (for multi-pass fusion).  The rules have
    # distinct names; only the (target_class, op_type) pair collides.
    allowed = {("*", "hc_pre")}
    duplicates = [k for k, v in counts.items() if v > 1 and k not in allowed]
    assert not duplicates, f"duplicate rule signatures: {duplicates}"


# ─────────────────────────────────────────────────────────────────────────────
# 9. load_model_yaml_rules registers every rule for the model id
# ─────────────────────────────────────────────────────────────────────────────

def test_register_via_yaml_loader():
    clear_rules()
    try:
        load_model_yaml_rules("deepseek-ai/DeepSeek-V4")
        registered = all_rules()
        assert len(registered) >= EXPECTED_RULES, (
            f"expected ≥ {EXPECTED_RULES} rules registered, got {len(registered)}"
        )
        op_types = {r.op_type for r in registered}
        for required in (
            "rms_norm", "parallel_embedding", "linear",
            "column_parallel_linear", "row_parallel_linear",
            "rotary_emb", "kv_compressor", "sparse_indexer",
            "mla_sparse_attn", "moe_gate", "moe_expert_swiglu",
            "moe_dispatch", "hc_pre", "hc_post", "hc_head",
            "sparse_attention_kernel",
        ):
            assert required in op_types, f"missing rule op_type={required!r}"
    finally:
        clear_rules()
