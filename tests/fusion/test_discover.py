"""Unit tests for templates + joiner + end-to-end YAML round-trip."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from python.zrt.fusion.discover import (
    TEMPLATES,
    discover_fusion_rules,
    get_template,
    join_rules,
)
from python.zrt.fusion.discover.ast_scanner import AstClassInfo, AstScanResult


# ─── templates.get_template ───────────────────────────────────────────────────

class TestTemplates:
    def test_known_class_returns_template(self):
        tpl = get_template("RMSNorm")
        assert tpl is not None
        assert tpl["op_type"] == "rms_norm"

    def test_alias_resolves(self):
        tpl = get_template("LlamaRMSNorm")
        assert tpl is not None
        assert tpl["op_type"] == "rms_norm"

    def test_unknown_returns_none(self):
        assert get_template("DefinitelyNotARealClass") is None

    def test_default_template_has_class_only_match(self):
        d = TEMPLATES["_default"]
        assert d["match"]["kind"] == "class_only"

    def test_template_returns_deep_copy(self):
        a = get_template("RMSNorm")
        a["op_type"] = "MUTATED"
        b = get_template("RMSNorm")
        assert b["op_type"] == "rms_norm"


# ─── joiner.join_rules ────────────────────────────────────────────────────────

def _mk(name: str, *, is_nn_module=True, forward_calls=()) -> AstClassInfo:
    return AstClassInfo(
        name=name,
        bases=["nn.Module"] if is_nn_module else [],
        is_nn_module=is_nn_module,
        forward_calls=list(forward_calls),
    )


def _mk_fn(name: str) -> AstClassInfo:
    return AstClassInfo(name=name, bases=[], is_nn_module=False)


class TestJoiner:
    def test_known_classes_get_their_templates(self):
        ast = AstScanResult(
            classes=[_mk("RMSNorm"), _mk("Linear")],
            top_level_funcs=[],
            file_path="/dummy",
        )
        rules, notes = join_rules(ast, runtime={})
        op_types = {r["op_type"] for r in rules}
        assert "rms_norm" in op_types
        assert "linear" in op_types

    def test_runtime_sequence_attached_to_ordered_regex_rule(self):
        ast = AstScanResult(
            classes=[_mk("RMSNorm")],
            top_level_funcs=[],
            file_path="/dummy",
        )
        runtime = {
            "RMSNorm": [(
                "aten.pow.Tensor_Scalar",
                "aten.mean.dim",
                "aten.add.Scalar",
                "aten.rsqrt.default",
                "aten.mul.Tensor",
                "aten.mul.Tensor",
            )],
        }
        rules, _ = join_rules(ast, runtime)
        rms = next(r for r in rules if r["op_type"] == "rms_norm")
        assert rms["match"]["kind"] == "ordered_regex"
        assert "op_regexes" in rms["match"]
        assert len(rms["match"]["op_regexes"]) == 6

    def test_ordered_regex_without_runtime_degrades_to_class_only(self):
        ast = AstScanResult(
            classes=[_mk("RMSNorm")],
            top_level_funcs=[],
            file_path="/dummy",
        )
        rules, notes = join_rules(ast, runtime={})
        rms = next(r for r in rules if r["op_type"] == "rms_norm")
        assert rms["match"]["kind"] == "class_only"
        assert any("RMSNorm" in n and "degraded" in n for n in notes)

    def test_class_only_template_unaffected_by_runtime(self):
        ast = AstScanResult(
            classes=[_mk("Compressor")],
            top_level_funcs=[],
            file_path="/dummy",
        )
        runtime = {"Compressor": [("aten.mm.default", "aten.add.Tensor")]}
        rules, _ = join_rules(ast, runtime)
        comp = next(r for r in rules if r["op_type"] == "kv_compressor")
        assert comp["match"]["kind"] == "class_only"
        assert "op_regexes" not in comp["match"]

    def test_unknown_class_falls_back_to_default_with_note(self):
        ast = AstScanResult(
            classes=[_mk("MyMysteriousFusion")],
            top_level_funcs=[],
            file_path="/dummy",
        )
        rules, notes = join_rules(ast, runtime={})
        rule = next(r for r in rules if r["target_class"] == "MyMysteriousFusion")
        assert rule["op_type"] == "MyMysteriousFusion"
        assert rule["match"]["kind"] == "class_only"
        assert any("_default" in n for n in notes)
        assert any("review" in n for n in notes)

    def test_top_level_helper_emitted_when_referenced(self):
        # apply_rotary_emb is a known template AND referenced from forward
        ast = AstScanResult(
            classes=[_mk("MLA", forward_calls=["apply_rotary_emb"])],
            top_level_funcs=[_mk_fn("apply_rotary_emb"),
                             _mk_fn("internal_helper")],
            file_path="/dummy",
        )
        rules, _ = join_rules(ast, runtime={})
        op_types = {r["op_type"] for r in rules}
        assert "apply_rotary_emb" in op_types
        # Generic helper not in TEMPLATES → suppressed
        assert "internal_helper" not in op_types

    def test_non_nn_module_classes_skipped(self):
        ast = AstScanResult(
            classes=[
                _mk("Linear"),  # nn.Module — kept
                _mk("ModelArgs", is_nn_module=False),  # dataclass — dropped
            ],
            top_level_funcs=[],
            file_path="/dummy",
        )
        rules, _ = join_rules(ast, runtime={})
        names = {r["target_class"] for r in rules}
        assert "Linear" in names
        assert "ModelArgs" not in names

    def test_rules_sorted_by_priority_then_name(self):
        ast = AstScanResult(
            classes=[_mk("Block"), _mk("RMSNorm"), _mk("MoE")],
            top_level_funcs=[],
            file_path="/dummy",
        )
        rules, _ = join_rules(ast, runtime={})
        # MoE priority=30, RMSNorm priority=20, Block priority=5
        op_types = [r["op_type"] for r in rules]
        assert op_types.index("moe_layer") < op_types.index("rms_norm")
        assert op_types.index("rms_norm") < op_types.index("transformer_block")


# ─── End-to-end: discover → YAML → load_yaml_rules ────────────────────────────

class TestYamlRoundTrip:
    def test_join_output_loads_via_yaml_loader(self, tmp_path: Path):
        from python.zrt.transform.fusion.yaml_loader import load_yaml_rules

        ast = AstScanResult(
            classes=[
                _mk("RMSNorm"),
                _mk("Linear"),
                _mk("Compressor"),
                _mk("Indexer"),
            ],
            top_level_funcs=[],
            file_path="/dummy",
        )
        runtime = {
            "RMSNorm": [(
                "aten.pow.Tensor_Scalar",
                "aten.mean.dim",
                "aten.add.Scalar",
                "aten.rsqrt.default",
                "aten.mul.Tensor",
                "aten.mul.Tensor",
            )],
        }
        rules, _ = join_rules(ast, runtime)

        # Serialise, then read back through the production YAML loader.
        yaml_path = tmp_path / "draft.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(rules, f, default_flow_style=False, sort_keys=False)

        loaded = load_yaml_rules(yaml_path)
        assert len(loaded) >= 4

        # Every loaded rule must have a non-None pattern (the new schema's
        # invariant).  This catches accidental schema drift in templates.
        for r in loaded:
            assert r.pattern is not None, f"{r.target_class} has no pattern"

    def test_discover_fusion_rules_skip_runtime(self, tmp_path: Path):
        # Use a tiny inline source file so the static path is deterministic
        # and self-contained.
        src = (
            "import torch\n"
            "from torch import nn\n"
            "import torch.nn.functional as F\n"
            "\n"
            "class RMSNorm(nn.Module):\n"
            "    def __init__(self, dim):\n"
            "        super().__init__()\n"
            "        self.weight = torch.ones(dim)\n"
            "    def forward(self, x):\n"
            "        return F.rms_norm(x, x.shape[-1:], self.weight)\n"
        )
        p = tmp_path / "tiny.py"
        p.write_text(src, encoding="utf-8")

        rules, notes = discover_fusion_rules(str(p), skip_runtime=True)
        op_types = {r["op_type"] for r in rules}
        assert "rms_norm" in op_types
        # Runtime was skipped → ordered_regex template degraded to class_only.
        rms = next(r for r in rules if r["op_type"] == "rms_norm")
        assert rms["match"]["kind"] == "class_only"

    def test_discover_requires_hf_id_when_runtime_enabled(self, tmp_path: Path):
        src = "from torch import nn\nclass A(nn.Module):\n    pass\n"
        p = tmp_path / "x.py"
        p.write_text(src, encoding="utf-8")
        with pytest.raises(ValueError):
            discover_fusion_rules(str(p))


# ─── Runtime-trace test (skipped — needs HF model) ────────────────────────────

@pytest.mark.skipif(
    True,
    reason="Runtime trace requires a real HF model environment; "
           "skipped in unit-test mode.",
)
def test_run_runtime_trace_smoke():
    from python.zrt.fusion.discover import run_runtime_trace
    out = run_runtime_trace("Qwen/Qwen2.5-0.5B-Instruct", num_layers=2)
    assert isinstance(out, dict)
