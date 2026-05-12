"""Tensor helpers and zero-cost op filter list."""
from __future__ import annotations

from typing import Any, List

import torch

# Ops that are pure metadata / autograd book-keeping, zero compute.
ALWAYS_TRANSPARENT: set[str] = {
    "aten.detach.default",
    "aten.alias.default",
    "aten.is_same_size.default",
    "prim.device.default",
}

# Shape-only ops: change stride/size metadata but do not move data.
SHAPE_OPS: set[str] = {
    "aten.view.default",
    "aten._unsafe_view.default",
    "aten.expand.default",
    "aten.expand_as.default",
    "aten.squeeze.default",
    "aten.squeeze.dim",
    "aten.unsqueeze.default",
    "aten.permute.default",
    "aten.transpose.int",
    "aten.as_strided.default",
    "aten.select.int",
    "aten.slice.Tensor",
    "aten.t.default",
    "aten.split.Tensor",
    "aten.split_with_sizes.default",
    "aten.unbind.int",
    "aten.diagonal.default",
    "aten.slice_backward.default",
}

SKIP_OPS: set[str] = ALWAYS_TRANSPARENT | SHAPE_OPS | {
    "aten._to_copy.default",
}

def shape_str(t: torch.Tensor) -> str:
    return str(list(t.shape))


def collect_tensors(args: tuple, kwargs: dict) -> List[torch.Tensor]:
    tensors = []
    for a in args:
        if isinstance(a, torch.Tensor):
            tensors.append(a)
        elif isinstance(a, (list, tuple)):
            for item in a:
                if isinstance(item, torch.Tensor):
                    tensors.append(item)
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            tensors.append(v)
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, torch.Tensor):
                    tensors.append(item)
    return tensors


def collect_output_tensors(out: Any) -> List[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [item for item in out if isinstance(item, torch.Tensor)]
    return []
