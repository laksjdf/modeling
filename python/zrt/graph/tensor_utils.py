"""Tensor helpers and zero-cost op filter list."""
from __future__ import annotations

from typing import Any, List

import torch

from python.zrt.transform.fusion.rules import ALWAYS_TRANSPARENT, SHAPE_OPS

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
