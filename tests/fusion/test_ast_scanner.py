"""Unit tests for ``zrt.fusion.discover.ast_scanner``."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from python.zrt.fusion.discover import scan_model_file
from python.zrt.fusion.discover.ast_scanner import (
    AstClassInfo,
    AstScanResult,
)


# ─── Inline-source unit tests ────────────────────────────────────────────────

INLINE_SRC = textwrap.dedent(
    '''
    """Tiny synthetic model file for AST scanner tests."""
    import torch
    from torch import nn
    import torch.nn.functional as F
    import torch.distributed as dist


    def apply_rotary_emb(x, freqs):
        return x * freqs


    class RMSNorm(nn.Module):
        """Root-mean-square layernorm."""
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            return F.rms_norm(x, x.shape[-1:], self.weight, self.eps)


    class Linear(nn.Module):
        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        def forward(self, x):
            return F.linear(x, self.weight, self.bias)


    class ColumnParallelLinear(Linear):
        def forward(self, x):
            y = F.linear(x, self.weight)
            dist.all_reduce(y)
            return y


    class Block(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.norm = RMSNorm(dim)
            self.linear = Linear(dim, dim)

        def forward(self, x):
            return self.linear(self.norm(x))
    '''
).strip()


@pytest.fixture()
def inline_file(tmp_path: Path) -> Path:
    p = tmp_path / "tiny_model.py"
    p.write_text(INLINE_SRC, encoding="utf-8")
    return p


def test_scan_finds_classes_and_top_level_func(inline_file: Path):
    res = scan_model_file(str(inline_file))
    assert isinstance(res, AstScanResult)

    names = [c.name for c in res.classes]
    assert names == ["RMSNorm", "Linear", "ColumnParallelLinear", "Block"]

    fn_names = [f.name for f in res.top_level_funcs]
    assert "apply_rotary_emb" in fn_names


def test_nn_module_subclass_detection(inline_file: Path):
    res = scan_model_file(str(inline_file))
    by_name = {c.name: c for c in res.classes}
    assert by_name["RMSNorm"].is_nn_module
    assert by_name["Linear"].is_nn_module
    # Inherits from already-scanned nn.Module subclass
    assert by_name["ColumnParallelLinear"].is_nn_module
    assert by_name["Block"].is_nn_module


def test_init_params_with_defaults(inline_file: Path):
    res = scan_model_file(str(inline_file))
    by_name = {c.name: c for c in res.classes}

    rms = by_name["RMSNorm"]
    assert "self" not in rms.init_params
    assert rms.init_params["dim"] is None  # no default
    assert rms.init_params["eps"] == "1e-6"

    lin = by_name["Linear"]
    assert lin.init_params["bias"] == "True"


def test_init_attrs_collects_self_assignments(inline_file: Path):
    res = scan_model_file(str(inline_file))
    by_name = {c.name: c for c in res.classes}

    lin = by_name["Linear"]
    # We don't care about the exact source — just that attrs were collected.
    assert "in_features" in lin.init_attrs
    assert "weight" in lin.init_attrs
    assert "bias" in lin.init_attrs


def test_forward_calls_collected(inline_file: Path):
    res = scan_model_file(str(inline_file))
    by_name = {c.name: c for c in res.classes}

    rms = by_name["RMSNorm"]
    assert any(c.endswith("rms_norm") for c in rms.forward_calls), \
        rms.forward_calls

    lin = by_name["Linear"]
    assert "F.linear" in lin.forward_calls


def test_forward_self_calls_and_dist_detection(inline_file: Path):
    res = scan_model_file(str(inline_file))
    by_name = {c.name: c for c in res.classes}

    block = by_name["Block"]
    # Block.forward calls self.linear(self.norm(x)) → both submodules captured
    assert "self.norm" in block.forward_self_calls
    assert "self.linear" in block.forward_self_calls
    assert block.has_dist_call is False

    cpl = by_name["ColumnParallelLinear"]
    assert cpl.has_dist_call is True


def test_top_level_function_summary(inline_file: Path):
    res = scan_model_file(str(inline_file))
    fn = next(f for f in res.top_level_funcs if f.name == "apply_rotary_emb")
    assert fn.is_nn_module is False
    assert fn.bases == []
    assert fn.init_params == {}


# ─── Real-file smoke test (DeepSeek V3.2) ────────────────────────────────────

# DeepSeek-V4's inference/model.py is loaded dynamically and not vendored
# in this repo; V3.2 has the closest class taxonomy and is what we ship,
# so we use it as the realistic AST-scanner target.
_REAL_MODEL_PATH = Path(__file__).resolve().parents[2] / (
    "hf_models/deepseek_v3_2/inference/model.py"
)


@pytest.mark.skipif(
    not _REAL_MODEL_PATH.exists(),
    reason=f"Local model file missing: {_REAL_MODEL_PATH}",
)
def test_scan_real_deepseek_model_file():
    res = scan_model_file(str(_REAL_MODEL_PATH))

    class_names = {c.name for c in res.classes}
    expected_classes = {
        "RMSNorm", "Linear", "Gate", "Expert", "Indexer",
        "ParallelEmbedding", "MoE", "Block",
        "ColumnParallelLinear", "RowParallelLinear", "MLA", "MLP",
    }
    missing = expected_classes - class_names
    assert not missing, f"Missing expected classes: {missing}"

    fn_names = {f.name for f in res.top_level_funcs}
    assert "apply_rotary_emb" in fn_names
    assert "linear" in fn_names

    # nn.Module detection — RMSNorm / Block must be flagged.
    by_name = {c.name: c for c in res.classes}
    assert by_name["RMSNorm"].is_nn_module
    assert by_name["Block"].is_nn_module
    # ColumnParallelLinear inherits from Linear (already scanned) → True
    assert by_name["ColumnParallelLinear"].is_nn_module

    # Forward bodies should populate non-empty call lists for compute heads.
    assert by_name["Linear"].forward_calls, by_name["Linear"].forward_calls
    assert by_name["Expert"].forward_self_calls, \
        by_name["Expert"].forward_self_calls
