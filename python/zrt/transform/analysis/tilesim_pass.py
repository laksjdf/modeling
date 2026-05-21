"""TilesimLatencyPass: overwrite Roofline latency with Tilesim predictions.

When ``--tilesim`` is set, this pass runs after RooflinePass and calls
the Tilesim ``op_latency_predict`` API directly for each mapped op_type.
Only ``latency`` from the Tilesim response is extracted; all other
output fields (component_latency, l2_hit_rate, etc.) are ignored.

Unmapped op_types (e.g. ``comm.*``) keep their Roofline annotations.

Data flow:
  RooflinePass  → annotations["latency_us"] (Roofline value)
  TilesimLatencyPass → op_latency_predict(payload) → overwrite annotations["latency_us"]
  DAGScheduler → reads annotations → Timeline (Tilesim value for mapped ops)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext

logger = logging.getLogger(__name__)


# ── DType → Tilesim precision string ────────────────────────────────────────────

_DTYPE_TO_TILESIM_PRECISION: dict[str, str] = {
    "fp32":     "FLOAT32",
    "fp16":     "FLOAT16",
    "bf16":     "BFLOAT16",
    "fp8_e4m3": "FLOAT8_E4M3",
    "fp8_e5m2": "FLOAT8_E5M2",
    "int8":     "INT8",
    "int4":     "INT4",
    "int32":    "INT32",
    "int64":    "INT64",
    "uint8":    "UINT8",
    "bool":     "BOOL",
    "unknown":  "FLOAT16",
}


# ── op_type → Tilesim op_name mapping ───────────────────────────────────────────

_OP_TYPE_TO_TILESIM_NAME: dict[str, str] = {
    # matmul family
    "aten.mm.default":             "MatMul",
    "aten.mm":                     "MatMul",
    "aten.addmm.default":          "AddMatMul",
    "aten.addmm":                  "AddMatMul",
    "aten.bmm.default":            "BatchMatMul",
    "aten.bmm":                    "BatchMatMul",
    "aten.matmul.default":         "MatMul",
    "aten.matmul":                 "MatMul",
    "aten.linear.default":         "Linear",
    "aten.linear":                 "Linear",
    # elementwise binary
    "aten.add.Tensor":             "Add",
    "aten.add_.Tensor":            "Add",
    "aten.add.Scalar":             "Add",
    "aten.sub.Tensor":             "Sub",
    "aten.sub.Scalar":             "Sub",
    "aten.mul.Tensor":             "Mul",
    "aten.mul.Scalar":             "Mul",
    "aten.div.Tensor":             "Div",
    "aten.div.Scalar":             "Div",
    # elementwise unary
    "aten.neg.default":            "Neg",
    "aten.abs.default":            "Abs",
    "aten.relu.default":           "Relu",
    "aten.relu_.default":          "Relu",
    "aten.silu.default":           "Silu",
    "aten.silu_.default":          "Silu",
    "aten.gelu.default":           "Gelu",
    "aten.sigmoid.default":        "Sigmoid",
    "aten.tanh.default":           "Tanh",
    "aten.exp.default":            "Exp",
    "aten.log.default":            "Log",
    "aten.sqrt.default":           "Sqrt",
    "aten.rsqrt.default":          "Rsqrt",
    "aten.sin.default":            "Sin",
    "aten.cos.default":            "Cos",
    "aten.reciprocal.default":     "Reciprocal",
    "aten.pow.Tensor_Scalar":      "Pow",
    "aten.pow.Tensor_Tensor":      "Pow",
    "aten.clamp.default":          "Clamp",
    "aten.clamp.Scalar":           "Clamp",
    # masked ops
    "aten.masked_fill.Scalar":     "MaskedFill",
    "aten.masked_fill_.Scalar":    "MaskedFill",
    "aten.masked_fill.Tensor":     "MaskedFill",
    # norm
    "aten.layer_norm.default":     "LayerNorm",
    "aten.layer_norm":             "LayerNorm",
    "aten.native_layer_norm.default": "LayerNorm",
    "rms_norm":                    "RmsNorm",
    "add_rms_norm":                "AddRmsNorm",
    "add_layer_norm":              "AddLayerNorm",
    # softmax
    "aten._softmax.default":       "Softmax",
    "aten.softmax.int":            "Softmax",
    # attention
    "aten._scaled_dot_product_flash_attention.default": "FlashAttention",
    "aten.scaled_dot_product_attention.default":        "FlashAttention",
    "flash_attn":                 "FlashAttention",
    "sdpa":                       "FlashAttention",
    # embedding
    "aten.embedding.default":      "Embedding",
    "embedding":                   "Embedding",
    # gather / scatter / index
    "aten.index.Tensor":           "Index",
    "aten.index_select.default":   "IndexSelect",
    "aten.gather.default":         "Gather",
    "aten.scatter.src":            "Scatter",
    # reduction
    "aten.mean.dim":               "Mean",
    "aten.mean.default":           "Mean",
    "aten.sum.dim_IntList":        "Sum",
    "aten.sum.default":            "Sum",
    "aten.var.correction":         "Var",
    "aten.amax.default":           "Amax",
    "aten.amin.default":           "Amin",
    # cumulative
    "aten.cumsum.default":         "CumSum",
    "aten.cumsum.dim":             "CumSum",
    # sort / topk
    "aten.sort.default":           "Sort",
    "aten.topk.default":           "TopK",
    # dtype cast
    "aten._to_copy.default":       "Cast",
    # copy / reshape
    "aten.copy_.default":          "Copy",
    "aten.clone.default":          "Clone",
    "aten.view.default":           "View",
    "aten.reshape.default":        "Reshape",
    "aten.permute.default":        "Permute",
    "aten.transpose.int":          "Transpose",
    "aten.contiguous.memory_format": "Contiguous",
    # fused semantic labels
    "swiglu":                      "SwiGLU",
    "gated_mlp":                   "GatedMLP",
    "Linear":                      "Linear",
    "lm_head":                     "Linear",
    "rope":                        "RoPE",
    "moe_gate":                    "MoEGate",
    "moe_gate_topk":               "MoEGateTopK",
    "moe_dispatch":                "MoEDispatch",
}


# ── Hardware config name → Tilesim accelerator path ────────────────────────────

_HW_TO_TILESIM_ACCELERATOR: dict[str, str] = {
    "ascend_910b": "910B1/910B1.yaml",
    "ascend_910c": "910C/910C.yaml",
    "nvidia_h100_sxm": "H100/H100.yaml",
    "nvidia_h800": "H800/H800.yaml",
    "nvidia_a100_80g": "A100/A100.yaml",
}


def _op_type_to_tilesim_name(op_type: str) -> str | None:
    if op_type in _OP_TYPE_TO_TILESIM_NAME:
        return _OP_TYPE_TO_TILESIM_NAME[op_type]
    if op_type.startswith("aten."):
        short = op_type.split(".", 2)[1]
        if short:
            return short[0].upper() + short[1:] if len(short) > 1 else short.upper()
    if op_type.startswith("comm."):
        return None
    return None


def _build_payload(node, hw, accelerator_override: str = "") -> dict | None:
    """Convert OpNode + HardwareSpec into a Tilesim API input dict.

    Returns None if the op_type cannot be mapped to a Tilesim op_name.
    """
    op_name = _op_type_to_tilesim_name(node.op_type)
    if op_name is None:
        return None
    if not node.inputs or not node.outputs:
        return None

    input_shapes = [list(t.shape) for t in node.inputs]
    output_shapes = [list(t.shape) for t in node.outputs]
    input_precision = [
        _DTYPE_TO_TILESIM_PRECISION.get(t.dtype.value, "FLOAT16")
        for t in node.inputs
    ]
    output_precision = [
        _DTYPE_TO_TILESIM_PRECISION.get(t.dtype.value, "FLOAT16")
        for t in node.outputs
    ]

    num_devices = getattr(hw.interconnect, "intra_node", None)
    core_num = getattr(num_devices, "num_devices", 24) if num_devices else 24

    if accelerator_override:
        accelerator = accelerator_override
    else:
        accelerator = _HW_TO_TILESIM_ACCELERATOR.get(hw.name, "")

    return {
        "op_name": op_name,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "input_precision": input_precision,
        "output_precision": output_precision,
        "core_num": core_num,
        "backend_type": "theo",
        "accelerator": accelerator,
    }


# ── TilesimLatencyPass ──────────────────────────────────────────────────────────

class TilesimLatencyPass(GraphPass):
    """Overwrite Roofline annotations with Tilesim latency predictions.

    For each node whose op_type can be mapped to a Tilesim op_name,
    calls ``op_latency_predict`` directly and overwrites
    ``annotations["latency_us"]``.  Unmapped nodes (comm.*, etc.)
    keep their Roofline values.
    """

    name = "tilesim_latency"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        from api.operator_api.op_latency_predict import op_latency_predict

        hw = ctx.hw_spec
        accelerator_override = ctx.tilesim_accelerator or ""
        g = graph.clone()

        overridden = 0
        skipped = 0
        for node in g.nodes.values():
            payload = _build_payload(node, hw, accelerator_override)
            if payload is None:
                skipped += 1
                continue

            result = op_latency_predict(payload)
            latency_ms = result.get("latency", 0.0)
            if latency_ms <= 0:
                continue

            latency_us = latency_ms * 1_000.0
            node.annotations["latency_us"] = latency_us
            node.annotations["sim_backend"] = "tilesim"
            overridden += 1

        logger.info(
            "TilesimLatencyPass: %d nodes overridden, %d skipped (unmapped)",
            overridden, skipped,
        )
        return g