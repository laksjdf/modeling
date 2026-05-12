"""Short-name to full aten op-type resolver.

Maps short op names used in YAML rules to full aten op-type strings
(e.g. ``"mm"`` → ``"aten.mm.default"``).  Names already containing a
dot are passed through unchanged.
"""
from __future__ import annotations

from typing import Sequence

# ── Short-name → full aten name mapping ──────────────────────────────────────

_SHORT_NAME_MAP: dict[str, str] = {
    # Arithmetic
    "add": "aten.add.Tensor",
    "add_": "aten.add_.Scalar",
    "mul": "aten.mul.Tensor",
    "div": "aten.div.Tensor",
    "div_": "aten.div_.Scalar",
    "sub": "aten.sub.Tensor",
    "neg": "aten.neg.default",

    # Linear algebra
    "mm": "aten.mm.default",
    "addmm": "aten.addmm.default",
    "bmm": "aten.bmm.default",
    "linear": "aten.linear.default",
    "matmul": "aten.matmul.default",

    # Shape / memory
    "view": "aten.view.default",
    "reshape": "aten.reshape.default",
    "permute": "aten.permute.default",
    "transpose": "aten.transpose.int",
    "contiguous": "aten.contiguous.memory_format",
    "clone": "aten.clone.default",
    "expand": "aten.expand.default",
    "squeeze": "aten.squeeze.dim",
    "unsqueeze": "aten.unsqueeze.default",
    "flatten": "aten.flatten.using_ints",
    "cat": "aten.cat.default",
    "stack": "aten.stack.default",
    "slice": "aten.slice.Tensor",
    "select": "aten.select.int",
    "split": "aten.split.Tensor",
    "chunk": "aten.chunk.default",
    "copy_": "aten.copy_.default",
    "as_strided": "aten.as_strided.default",
    "t": "aten.t.default",

    # Activation functions
    "gelu": "aten.gelu.default",
    "silu": "aten.silu.default",
    "relu": "aten.relu.default",
    "tanh": "aten.tanh.default",
    "sigmoid": "aten.sigmoid.default",
    "mish": "aten.mish.default",
    "elu": "aten.elu.default",
    "leaky_relu": "aten.leaky_relu.default",
    "softmax": "aten._softmax.default",
    "log_softmax": "aten._log_softmax.default",

    # Normalization
    "native_layer_norm": "aten.native_layer_norm.default",
    "native_batch_norm": "aten.native_batch_norm.default",
    "native_group_norm": "aten.native_group_norm.default",

    # Pooling
    "max_pool2d_with_indices": "aten.max_pool2d_with_indices.default",
    "avg_pool2d": "aten.avg_pool2d.default",

    # Power / statistics
    "pow": "aten.pow.Tensor_Scalar",
    "mean": "aten.mean.dim",
    "sum": "aten.sum.dim_IntList",
    "rsqrt": "aten.rsqrt.default",
    "norm": "aten.norm.ScalarOpt_dim",

    # Random / dropout
    "bernoulli_": "aten.bernoulli_.float",
    "empty_like": "aten.empty_like.default",

    # Embedding / loss
    "embedding": "aten.embedding.default",
    "nll_loss_forward": "aten.nll_loss_forward.default",

    # Conv
    "convolution": "aten.convolution.default",
}


def resolve_short_name(name: str) -> str:
    """Resolve a short op name to its full aten name.

    If *name* already contains a ``.`` it is returned unchanged.
    Raises ``KeyError`` for unknown short names.
    """
    if "." in name:
        return name
    full = _SHORT_NAME_MAP.get(name)
    if full is None:
        raise KeyError(f"Unknown short op name: {name!r}")
    return full


def resolve_short_names(seq: Sequence[str]) -> tuple[str, ...]:
    """Resolve a sequence of short names to a tuple of full aten names."""
    return tuple(resolve_short_name(n) for n in seq)
