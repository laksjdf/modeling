"""FusionPass OpGraph IR integration — graph-contract tests only.

These tests verify that ``FusionPass`` preserves OpGraph invariants on
inputs that no rule should match (empty / single-op / no-scope / comm /
cross-layer).  Rich-rule matching, semantic annotations, and DSv4 rule
behavior are covered separately under ``tests/transform/fusion/``.

(Rule-specific cases formerly in this file relied on a deprecated
"semantic label auto-fusion" code path and were removed during the
v2 rich-rules refactor — see ``docs/fusion_v2_rich_rules_zh.md``.)
"""
from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.ir.types import DType, TensorMeta
from python.zrt.transform.context import (
    ParallelConfig,
    StreamConfig,
    TransformContext,
)
from python.zrt.transform.fusion.api import FusionPass
import python.zrt.hardware.registry as hw_registry


# ── helpers ───────────────────────────────────────────────────────────────────

def _t(tid, shape=(1, 128), dtype=DType.BF16):
    return TensorMeta.from_shape_dtype(tid, shape, dtype)


def _node(nid, op_type, scope="", layer="0", module_class="", category="compute"):
    return OpNode(
        id=nid, op_type=op_type,
        inputs=[_t(f"{nid}_in")], outputs=[_t(f"{nid}_out")],
        scope=scope, layer=layer, module_class=module_class, category=category,
    )


def _edge(src, dst):
    return Edge(src=src, src_idx=0, dst=dst, dst_idx=0, tensor=_t("e"))


def _graph(nodes, edges, name="test"):
    return OpGraph(name=name, phase="prefill",
                   nodes={n.id: n for n in nodes},
                   edges=edges)


def _ctx(hw_name="nvidia_h100_sxm"):
    hw = hw_registry.load(hw_name)
    return TransformContext(hw_spec=hw, parallel=ParallelConfig(tp=1),
                            stream_config=StreamConfig())


# ── pass-through cases ────────────────────────────────────────────────────────

def test_empty_graph_returns_empty():
    g = OpGraph(name="t", phase="prefill")
    out = FusionPass().run(g, _ctx())
    assert out.num_nodes() == 0


def test_single_node_no_scope_unchanged():
    n = _node("a", "aten.mm.default", scope="")
    g = _graph([n], [])
    out = FusionPass().run(g, _ctx())
    assert out.num_nodes() == 1
    assert out.nodes["a"].op_type == "aten.mm.default"


def test_does_not_mutate_input():
    n1 = _node("a", "aten.mm.default", scope="model.layers.0.mlp")
    n2 = _node("b", "aten.add.Tensor", scope="model.layers.0.mlp")
    g = _graph([n1, n2], [_edge("a", "b")])
    original_count = g.num_nodes()
    FusionPass().run(g, _ctx())
    assert g.num_nodes() == original_count


def test_different_layer_nodes_not_fused():
    """Same scope but different layers must not be merged."""
    scope = "model.layers.X.mlp"
    a = _node("a", "aten.mm.default", scope=scope, layer="0", module_class="MLP")
    b = _node("b", "aten.mm.default", scope=scope, layer="1", module_class="MLP")
    g = _graph([a, b], [])
    out = FusionPass().run(g, _ctx())
    assert out.num_nodes() == 2


def test_comm_node_breaks_group():
    """A communication node between same-scope nodes splits them."""
    scope = "model.layers.0.mlp"
    a    = _node("a", "aten.mm.default", scope=scope, module_class="MLP")
    comm = _node("c", "comm.all_reduce", scope="", category="communication")
    b    = _node("b", "aten.mm.default", scope=scope, module_class="MLP")
    g = _graph([a, comm, b], [_edge("a", "c"), _edge("c", "b")])

    out = FusionPass().run(g, _ctx())
    comm_nodes = [n for n in out.nodes.values() if n.category == "communication"]
    assert len(comm_nodes) == 1
    assert comm_nodes[0].op_type == "comm.all_reduce"


def test_different_scope_nodes_not_fused():
    """Different scopes never merge."""
    a = _node("a", "aten.mm.default", scope="model.layers.0.q_proj", module_class="Linear")
    b = _node("b", "aten.mm.default", scope="model.layers.0.k_proj", module_class="Linear")
    g = _graph([a, b], [])
    out = FusionPass().run(g, _ctx())
    # No rule fuses single-op groups across scopes; both survive.
    assert out.num_nodes() >= 2


# ── FusionConfig allowlist / denylist gating ──────────────────────────────────

def _rms_norm_group(scope="model.layers.0.norm", module_class="RMSNorm",
                    layer="0", call_id=42):
    """Build 6 OpNodes that match the built-in rms_norm rule sequence."""
    op_types = [
        "aten.pow.Tensor_Scalar",
        "aten.mean.dim",
        "aten.add.Tensor",
        "aten.rsqrt.default",
        "aten.mul.Tensor",
        "aten.mul.Tensor",
    ]
    nodes = []
    for i, op_t in enumerate(op_types):
        n = _node(f"r{i}", op_t, scope=scope, layer=layer, module_class=module_class)
        n.call_id = call_id
        nodes.append(n)
    edges = [_edge(nodes[i].id, nodes[i + 1].id) for i in range(len(nodes) - 1)]
    return nodes, edges


def _ctx_with_fusion(fusion_cfg, training=False, model_id="hf_models/deepseek_v4"):
    from python.zrt.transform.context import FusionConfig, TrainingConfig
    hw = hw_registry.load("nvidia_h100_sxm")
    kwargs = dict(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1),
        stream_config=StreamConfig(),
        fusion=fusion_cfg or FusionConfig(),
        model_id=model_id,
    )
    if training:
        kwargs["training"] = TrainingConfig()
    return TransformContext(**kwargs)


def test_enabled_rules_allowlist_only_fires_listed_rules():
    """When FusionConfig.enabled_rules is set, only those rules can fire."""
    from python.zrt.transform.context import FusionConfig

    nodes, edges = _rms_norm_group(call_id=11)
    g = _graph(nodes, edges)
    cfg = FusionConfig(enabled_rules={"rms_norm"})
    out = FusionPass().run(g, _ctx_with_fusion(cfg))

    fused = [n for n in out.nodes.values() if n.is_fused or n.fused_from]
    assert len(fused) == 1
    assert fused[0].op_type == "rms_norm"
    assert fused[0].annotations.get("fused_by_rule") == "rms_norm"


def test_enabled_rules_excludes_others():
    """RMSNorm group should NOT fuse when only an unrelated rule is enabled."""
    from python.zrt.transform.context import FusionConfig

    nodes, edges = _rms_norm_group(call_id=12)
    g = _graph(nodes, edges)
    cfg = FusionConfig(enabled_rules={"linear"})  # rms_norm explicitly excluded
    out = FusionPass().run(g, _ctx_with_fusion(cfg))

    # No fusion happened — all 6 raw aten nodes survive.
    assert out.num_nodes() == 6
    assert all(not n.is_fused for n in out.nodes.values())


def test_disabled_rules_subtracts_from_default_set():
    """disabled_rules removes a rule from the default-phase set."""
    from python.zrt.transform.context import FusionConfig

    nodes, edges = _rms_norm_group(call_id=13)
    g = _graph(nodes, edges)
    cfg = FusionConfig(disabled_rules={"rms_norm", "rms_norm_nn", "dsv4_rms_norm",
                                       "rms_norm_inline", "rms_coef"})
    out = FusionPass().run(g, _ctx_with_fusion(cfg))

    assert out.num_nodes() == 6  # raw, untouched


def _attn_fragment_graph(call_id_attn=99, call_id_child=100):
    """Attention forward whose body is split by a child module call.

    Layout:  attn0 → attn1 → attn2 → child → attn3 → attn4.

    The attention call_id appears in two buckets (length 3 and length 2),
    so neither bucket equals the full call_id total → is_full_forward=False
    and class_only rules must not fire on either bucket.
    """
    op_types = ["aten.pow.Tensor_Scalar", "aten.mean.dim", "aten.add.Tensor",
                "aten.rsqrt.default", "aten.mul.Tensor"]
    nodes = []
    for i, op_t in enumerate(op_types):
        n = _node(f"f{i}", op_t,
                  scope="model.layers.0.attn",
                  module_class="Attention", layer="0")
        n.call_id = call_id_attn
        nodes.append(n)
    child = _node("child", "aten.mm.default",
                  scope="model.layers.0.attn.q_proj",
                  module_class="Linear", layer="0")
    child.call_id = call_id_child
    # Interleave: attn[0..2] → child → attn[3..4]
    ordered = nodes[:3] + [child] + nodes[3:]
    edges = [_edge(ordered[i].id, ordered[i + 1].id) for i in range(len(ordered) - 1)]
    return ordered, edges


def test_unmatched_ops_left_raw_no_collapse_fallback():
    """No active rule matches a fragment bucket → ops survive as raw aten nodes."""
    from python.zrt.transform.context import FusionConfig

    nodes, edges = _attn_fragment_graph()
    g = _graph(nodes, edges)
    cfg = FusionConfig()  # defaults: structural collapse OFF
    out = FusionPass().run(g, _ctx_with_fusion(cfg))

    attn_nodes = [n for n in out.nodes.values()
                  if n.scope == "model.layers.0.attn"]
    # No bucket should fuse to "Attention" or any class_only result; raw aten survives.
    bad = [n.op_type for n in attn_nodes
           if n.op_type == "Attention" or n.op_type == "mla_sparse_attn"]
    assert not bad, f"class_only rule fired on a fragment bucket: {bad}"
    # Each of the 5 attn ops still appears as its raw aten op_type.
    raw_op_types = sorted(n.op_type for n in attn_nodes)
    assert raw_op_types == sorted([
        "aten.pow.Tensor_Scalar", "aten.mean.dim", "aten.add.Tensor",
        "aten.rsqrt.default", "aten.mul.Tensor",
    ])


def test_allow_structural_collapse_revives_legacy_behavior():
    """Escape hatch: legacy v2 collapse fires when explicitly enabled."""
    from python.zrt.transform.context import FusionConfig

    nodes, edges = _attn_fragment_graph(call_id_attn=200, call_id_child=201)
    g = _graph(nodes, edges)
    cfg = FusionConfig(allow_structural_collapse=True)
    out = FusionPass().run(g, _ctx_with_fusion(cfg))

    collapsed = [n for n in out.nodes.values()
                 if n.annotations.get("fused_by_rule") == "_collapsed"]
    assert collapsed, "structural collapse should have fired with the escape hatch"


def test_unknown_rule_names_raise_for_enabled_rules():
    """Typos in enabled_rules (allowlist) are surfaced loudly."""
    import pytest
    from python.zrt.transform.context import FusionConfig

    nodes, edges = _rms_norm_group(call_id=14)
    g = _graph(nodes, edges)
    cfg = FusionConfig(enabled_rules={"nonexistent_rule"})
    with pytest.raises(ValueError, match="Unknown fusion rule names"):
        FusionPass().run(g, _ctx_with_fusion(cfg))


def test_unknown_disabled_rule_names_are_noop():
    """disabled_rules tolerates names that aren't registered (model-specific)."""
    from python.zrt.transform.context import FusionConfig

    nodes, edges = _rms_norm_group(call_id=15)
    g = _graph(nodes, edges)
    # Disable a non-existent name AND a real one — the real disable should still work.
    cfg = FusionConfig(disabled_rules={"not_a_real_rule",
                                       "rms_norm", "rms_norm_nn", "dsv4_rms_norm",
                                       "rms_norm_inline", "rms_coef"})
    out = FusionPass().run(g, _ctx_with_fusion(cfg))
    # rms_norm rule was disabled → bucket stays raw.
    assert out.num_nodes() == 6


def test_training_default_skips_compressor_rule():
    """Training-default fusion config disables stateful kv_compressor."""
    from python.zrt.transform.fusion.loading.fusion_config import (
        resolve_fusion_config,
    )
    from python.zrt.transform.fusion.loading.rule_set_initializer import (
        initialize_rules,
    )
    from python.zrt.transform.fusion.registry import iter_active_rules

    initialize_rules("hf_models/deepseek_v4")
    cfg = resolve_fusion_config("hf_models/deepseek_v4", "training")
    active = iter_active_rules(cfg, "training")
    names = {r.name for r in active}

    assert "kv_compressor" not in names
    assert "sparse_indexer" not in names
    # Things we DO want fused in training are still active.
    assert "rms_norm" in names
    assert "hc_pre_attn" in names
    assert "linear" in names


# ── Step-2: target_class no longer gates pattern rules ────────────────────────

def _ctx_no_autoload(fusion_cfg=None):
    """Build a TransformContext that won't trigger ``load_platform_rules``.

    ``fuse()`` (the module-level entry point) does NOT call
    ``load_platform_rules``, so the rules currently in the registry are
    used as-is.  This lets a test isolate a single hand-registered rule
    via ``clear_rules()`` + ``register_rule()``.
    """
    from python.zrt.transform.context import FusionConfig
    hw = hw_registry.load("nvidia_h100_sxm")
    return TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1),
        stream_config=StreamConfig(),
        fusion=fusion_cfg or FusionConfig(),
    )


def test_target_class_no_longer_gates_ordered_regex():
    """An ordered_regex rule with target_class=Foo must still match a bucket
    whose module_class=Bar.  Pattern rules are matched on op-sequence shape
    alone after the v3 Step-2 change.
    """
    from python.zrt.transform.context import FusionConfig
    from python.zrt.transform.fusion import (
        MatchPattern,
        ModuleFusionRule,
        register_rule,
    )
    from python.zrt.transform.fusion.pipeline.fuser import fuse
    from python.zrt.transform.fusion.registry import clear_rules

    clear_rules()
    register_rule(ModuleFusionRule(
        target_class="ClassFoo",
        op_type="fused_foo",
        name="fused_foo",
        pattern=MatchPattern(
            kind="ordered_regex",
            op_regexes=(r"aten\.add\.Tensor", r"aten\.mul\.Tensor"),
            min_ops=2,
            max_ops=2,
        ),
        priority=50,
    ))

    # Bucket carries module_class="ClassBar" — does NOT match the rule's
    # target_class, but the regex sequence does.
    a = _node("a", "aten.add.Tensor",
              scope="model.layers.0.foo", module_class="ClassBar", layer="0")
    b = _node("b", "aten.mul.Tensor",
              scope="model.layers.0.foo", module_class="ClassBar", layer="0")
    a.call_id = 7777
    b.call_id = 7777
    g = _graph([a, b], [_edge("a", "b")])

    cfg = FusionConfig(enabled_rules={"fused_foo"})
    try:
        out = fuse(g, _ctx_no_autoload(cfg))
    finally:
        clear_rules()

    fused = [n for n in out.nodes.values() if n.is_fused or n.fused_from]
    assert len(fused) == 1, (
        "ordered_regex rule must match despite class mismatch: "
        f"got nodes={[(n.id, n.op_type) for n in out.nodes.values()]}"
    )
    assert fused[0].op_type == "fused_foo"
    assert fused[0].annotations.get("fused_by_rule") == "fused_foo"


def test_class_only_still_gates_target_class():
    """class_only rules have no other signal, so target_class must still gate.
    Bucket whose module_class matches → fuses; bucket with wrong class → not.
    """
    from python.zrt.transform.context import FusionConfig
    from python.zrt.transform.fusion import (
        MatchPattern,
        ModuleFusionRule,
        register_rule,
    )
    from python.zrt.transform.fusion.pipeline.fuser import fuse
    from python.zrt.transform.fusion.registry import clear_rules

    def _build_bucket(module_class: str, call_id: int):
        # 2 ops, both tagged with the same call_id so the bucket spans the
        # full forward call (is_full_forward=True) — required for class_only.
        a = _node("ca", "aten.mm.default",
                  scope=f"model.layers.0.{module_class.lower()}",
                  module_class=module_class, layer="0")
        b = _node("cb", "aten.add.Tensor",
                  scope=f"model.layers.0.{module_class.lower()}",
                  module_class=module_class, layer="0")
        a.call_id = call_id
        b.call_id = call_id
        return _graph([a, b], [_edge("ca", "cb")])

    def _run_with_isolated_rule(graph):
        clear_rules()
        register_rule(ModuleFusionRule(
            target_class="ClassOnlyTarget",
            op_type="class_only_fused",
            name="class_only_fused",
            pattern=MatchPattern(kind="class_only", min_ops=1, max_ops=10),
            priority=50,
        ))
        cfg = FusionConfig(enabled_rules={"class_only_fused"})
        return fuse(graph, _ctx_no_autoload(cfg))

    try:
        # Graph A: module_class matches → should fuse.
        out_a = _run_with_isolated_rule(
            _build_bucket("ClassOnlyTarget", call_id=8001),
        )
        fused_a = [n for n in out_a.nodes.values() if n.is_fused or n.fused_from]
        assert len(fused_a) == 1, (
            f"class_only rule should fire when module_class matches; got "
            f"{[(n.id, n.op_type) for n in out_a.nodes.values()]}"
        )
        assert fused_a[0].op_type == "class_only_fused"

        # Graph B: module_class does NOT match → must not fuse.
        out_b = _run_with_isolated_rule(
            _build_bucket("WrongClass", call_id=8002),
        )
        fused_b = [n for n in out_b.nodes.values() if n.is_fused or n.fused_from]
        assert not fused_b, (
            f"class_only rule must still gate on target_class; got fused="
            f"{[(n.id, n.op_type) for n in fused_b]}"
        )
        # The two raw aten ops survive untouched.
        assert out_b.num_nodes() == 2
    finally:
        clear_rules()


# ── Step-3: sliding-window partial match + multi-pass fixed-point ─────────────

def _seq_graph(op_types, *, scope, module_class, call_id, layer="0"):
    """Build a chained graph from a list of op_types, all sharing the same
    scope/module_class/call_id.  Returns (nodes, edges) ready for ``_graph``.
    """
    nodes = []
    for i, op_t in enumerate(op_types):
        n = _node(f"s{i}", op_t, scope=scope, layer=layer, module_class=module_class)
        n.call_id = call_id
        nodes.append(n)
    edges = [_edge(nodes[i].id, nodes[i + 1].id) for i in range(len(nodes) - 1)]
    return nodes, edges


def test_partial_match_inline_rmsnorm_in_attention():
    """A 5-op inline RMSNorm in the middle of a 7-op Attention bucket must
    be fused via the sliding-window scanner; the surrounding linears stay raw.
    """
    from python.zrt.transform.context import FusionConfig
    from python.zrt.transform.fusion import (
        MatchPattern,
        ModuleFusionRule,
        register_rule,
    )
    from python.zrt.transform.fusion.pipeline.fuser import fuse
    from python.zrt.transform.fusion.registry import clear_rules

    clear_rules()
    try:
        register_rule(ModuleFusionRule(
            target_class="*",
            op_type="rms_norm_inline",
            name="rms_norm_inline",
            pattern=MatchPattern(
                kind="ordered_regex",
                op_regexes=(
                    r"aten\.pow\.Tensor_Scalar",
                    r"aten\.mean\.dim",
                    r"aten\.add\.Tensor",
                    r"aten\.rsqrt\.default",
                    r"aten\.mul\.Tensor",
                ),
                min_ops=5,
                max_ops=5,
            ),
            priority=50,
        ))

        # 7-op bucket: linear_a / pow / mean / add / rsqrt / mul / linear_b
        # with extra non-rms ops for the bucket to be a fragment.
        op_types = [
            "aten.mm.default",          # linear_a
            "aten.pow.Tensor_Scalar",
            "aten.mean.dim",
            "aten.add.Tensor",
            "aten.rsqrt.default",
            "aten.mul.Tensor",
            "aten.mm.default",          # linear_b
        ]
        nodes, edges = _seq_graph(
            op_types, scope="model.layers.0.attn",
            module_class="Attention", call_id=9001,
        )
        # Add a sibling op in the same call_id elsewhere to ensure
        # is_full_forward=False for this bucket.
        sibling = _node("sib", "aten.add.Tensor",
                        scope="model.layers.0.attn",
                        module_class="Attention", layer="0")
        sibling.call_id = 9001
        # We don't connect sibling — call_id totals will be 8 but bucket is 7.
        all_nodes = nodes + [sibling]
        g = _graph(all_nodes, edges)

        cfg = FusionConfig(enabled_rules={"rms_norm_inline"})
        out = fuse(g, _ctx_no_autoload(cfg))
    finally:
        clear_rules()

    fused = [n for n in out.nodes.values() if n.is_fused or n.fused_from]
    assert len(fused) == 1, (
        f"sliding-window scanner should produce one fused node; got "
        f"{[(n.id, n.op_type) for n in out.nodes.values()]}"
    )
    assert fused[0].op_type == "rms_norm_inline"
    # The two linears + the sibling survive as raw aten.
    raw_mm = [n for n in out.nodes.values() if n.op_type == "aten.mm.default"]
    assert len(raw_mm) == 2


def test_six_op_rmsnorm_wins_over_inline():
    """When both 5-op (inline) and 6-op (full) RMSNorm rules are active and
    both share priority=50, the longest match wins on tie-break.
    """
    from python.zrt.transform.context import FusionConfig
    from python.zrt.transform.fusion import (
        MatchPattern,
        ModuleFusionRule,
        register_rule,
    )
    from python.zrt.transform.fusion.pipeline.fuser import fuse
    from python.zrt.transform.fusion.registry import clear_rules

    clear_rules()
    try:
        register_rule(ModuleFusionRule(
            target_class="*",
            op_type="rms_norm_inline",
            name="rms_norm_inline",
            pattern=MatchPattern(
                kind="ordered_regex",
                op_regexes=(
                    r"aten\.pow\.Tensor_Scalar",
                    r"aten\.mean\.dim",
                    r"aten\.add\.Tensor",
                    r"aten\.rsqrt\.default",
                    r"aten\.mul\.Tensor",
                ),
                min_ops=5,
                max_ops=5,
            ),
            priority=50,
        ))
        register_rule(ModuleFusionRule(
            target_class="*",
            op_type="rms_norm",
            name="rms_norm",
            pattern=MatchPattern(
                kind="ordered_regex",
                op_regexes=(
                    r"aten\.pow\.Tensor_Scalar",
                    r"aten\.mean\.dim",
                    r"aten\.add\.Tensor",
                    r"aten\.rsqrt\.default",
                    r"aten\.mul\.Tensor",
                    r"aten\.mul\.Tensor",
                ),
                min_ops=6,
                max_ops=6,
            ),
            priority=50,
        ))

        op_types = [
            "aten.pow.Tensor_Scalar",
            "aten.mean.dim",
            "aten.add.Tensor",
            "aten.rsqrt.default",
            "aten.mul.Tensor",
            "aten.mul.Tensor",
        ]
        nodes, edges = _seq_graph(
            op_types, scope="model.layers.0.norm",
            module_class="SomeBlock", call_id=9101,
        )
        # Make the bucket a fragment so whole-bucket match is bypassed and
        # the sliding-window scanner is the deciding mechanism.
        sibling = _node("sib", "aten.add.Tensor",
                        scope="model.layers.0.norm",
                        module_class="SomeBlock", layer="0")
        sibling.call_id = 9101
        all_nodes = nodes + [sibling]
        g = _graph(all_nodes, edges)

        cfg = FusionConfig(enabled_rules={"rms_norm", "rms_norm_inline"})
        out = fuse(g, _ctx_no_autoload(cfg))
    finally:
        clear_rules()

    fused = [n for n in out.nodes.values() if n.is_fused or n.fused_from]
    assert len(fused) == 1
    assert fused[0].op_type == "rms_norm", (
        f"longer match should win on priority tie; got op_type={fused[0].op_type}"
    )


def test_multi_pass_rms_coef_then_hc_pre():
    """An 11-op bucket containing a 4-op rms_coef prefix + 7 more ops should
    fuse rms_coef in pass 1 and the larger hc_pre_attn (which references
    ``rms_coef`` as its first op) in pass 2.
    """
    from python.zrt.transform.context import FusionConfig
    from python.zrt.transform.fusion import (
        MatchPattern,
        ModuleFusionRule,
        register_rule,
    )
    from python.zrt.transform.fusion.pipeline.fuser import fuse
    from python.zrt.transform.fusion.registry import clear_rules

    clear_rules()
    try:
        register_rule(ModuleFusionRule(
            target_class="*",
            op_type="rms_coef",
            name="rms_coef",
            pattern=MatchPattern(
                kind="ordered_regex",
                op_regexes=(
                    r"aten\.pow\.Tensor_Scalar",
                    r"aten\.mean\.dim",
                    r"aten\.add\.Tensor",
                    r"aten\.rsqrt\.default",
                ),
                min_ops=4,
                max_ops=4,
            ),
            priority=50,
        ))
        register_rule(ModuleFusionRule(
            target_class="*",
            op_type="hc_pre",
            name="hc_pre_attn",
            pattern=MatchPattern(
                kind="ordered_regex",
                op_regexes=(
                    r"rms_coef",
                    r"aten\.mm\.default",
                    r"aten\.mul\.Tensor",
                    r"aten\._softmax\.default",
                    r"aten\.sigmoid\.default",
                    r"aten\._softmax\.default",
                    r"aten\.mul\.Tensor",
                    r"aten\.sum\.dim_IntList",
                ),
                min_ops=8,
                max_ops=8,
            ),
            priority=30,
        ))

        op_types = [
            "aten.pow.Tensor_Scalar",
            "aten.mean.dim",
            "aten.add.Tensor",
            "aten.rsqrt.default",
            "aten.mm.default",
            "aten.mul.Tensor",
            "aten._softmax.default",
            "aten.sigmoid.default",
            "aten._softmax.default",
            "aten.mul.Tensor",
            "aten.sum.dim_IntList",
        ]
        nodes, edges = _seq_graph(
            op_types, scope="model.layers.0.hc_pre_attn",
            module_class="Block", call_id=9201,
        )
        # Fragment: an extra sibling op so is_full_forward=False.
        sibling = _node("sib", "aten.add.Tensor",
                        scope="model.layers.0.hc_pre_attn",
                        module_class="Block", layer="0")
        sibling.call_id = 9201
        all_nodes = nodes + [sibling]
        g = _graph(all_nodes, edges)

        cfg = FusionConfig(enabled_rules={"rms_coef", "hc_pre_attn"})
        out = fuse(g, _ctx_no_autoload(cfg))
    finally:
        clear_rules()

    fused = [n for n in out.nodes.values() if n.is_fused or n.fused_from]
    op_types_out = sorted(n.op_type for n in fused)
    assert "hc_pre" in op_types_out, (
        f"multi-pass fusion should produce hc_pre after rms_coef; got "
        f"op_types={[(n.id, n.op_type) for n in out.nodes.values()]}"
    )
    # Exactly one hc_pre node — the rms_coef from pass 1 was consumed.
    hc_pre_nodes = [n for n in fused if n.op_type == "hc_pre"]
    assert len(hc_pre_nodes) == 1


def test_disabled_rules_disable_partial():
    """``FusionConfig.disabled_rules`` excludes a rule from the active set;
    the sliding-window scanner therefore won't see it either.
    """
    from python.zrt.transform.context import FusionConfig
    from python.zrt.transform.fusion import (
        MatchPattern,
        ModuleFusionRule,
        register_rule,
    )
    from python.zrt.transform.fusion.pipeline.fuser import fuse
    from python.zrt.transform.fusion.registry import clear_rules

    clear_rules()
    try:
        register_rule(ModuleFusionRule(
            target_class="*",
            op_type="rms_norm_inline",
            name="rms_norm_inline",
            pattern=MatchPattern(
                kind="ordered_regex",
                op_regexes=(
                    r"aten\.pow\.Tensor_Scalar",
                    r"aten\.mean\.dim",
                    r"aten\.add\.Tensor",
                    r"aten\.rsqrt\.default",
                    r"aten\.mul\.Tensor",
                ),
                min_ops=5,
                max_ops=5,
            ),
            priority=50,
        ))

        op_types = [
            "aten.mm.default",
            "aten.pow.Tensor_Scalar",
            "aten.mean.dim",
            "aten.add.Tensor",
            "aten.rsqrt.default",
            "aten.mul.Tensor",
            "aten.mm.default",
        ]
        nodes, edges = _seq_graph(
            op_types, scope="model.layers.0.attn",
            module_class="Attention", call_id=9301,
        )
        sibling = _node("sib", "aten.add.Tensor",
                        scope="model.layers.0.attn",
                        module_class="Attention", layer="0")
        sibling.call_id = 9301
        all_nodes = nodes + [sibling]
        g = _graph(all_nodes, edges)

        cfg = FusionConfig(disabled_rules={"rms_norm_inline"})
        out = fuse(g, _ctx_no_autoload(cfg))
    finally:
        clear_rules()

    fused = [n for n in out.nodes.values() if n.is_fused or n.fused_from]
    assert not fused, (
        f"disabled rule must not fuse; got fused="
        f"{[(n.id, n.op_type) for n in fused]}"
    )
    # All raw aten ops survive.
    assert out.num_nodes() == 8


def test_fixed_point_terminates_when_no_change():
    """A bucket with NO matching rule must not cause the multi-pass loop
    to spin forever.  Graph node count stays put.
    """
    from python.zrt.transform.context import FusionConfig
    from python.zrt.transform.fusion.pipeline.fuser import fuse
    from python.zrt.transform.fusion.registry import clear_rules

    clear_rules()
    try:
        op_types = [
            "aten.mm.default",
            "aten.add.Tensor",
            "aten.relu.default",
        ]
        nodes, edges = _seq_graph(
            op_types, scope="model.layers.0.misc",
            module_class="Misc", call_id=9401,
        )
        g = _graph(nodes, edges)

        cfg = FusionConfig()
        out = fuse(g, _ctx_no_autoload(cfg))
    finally:
        clear_rules()

    # No rule registered → no fusion → all 3 nodes survive.
    assert out.num_nodes() == 3
    assert all(not n.is_fused for n in out.nodes.values())
