"""TorchDispatchMode-based recorder that intercepts every aten op."""
from __future__ import annotations

import json
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from python.zrt.graph.tensor_utils import (
    SKIP_OPS,
    collect_tensors,
    collect_output_tensors,
    shape_str,
)
from python.zrt.graph.tracker import ModuleTracker
from python.zrt.graph.classifier import extract_layer_idx, classify_component

# Path fragments that identify torch/stdlib/our-own-dispatch internals —
# frames matching any of these are skipped when searching for the model call site.
_SKIP_PATH_MARKERS = (
    "\\torch\\", "/torch/",
    "\\zrt\\graph\\", "/zrt/graph/",
    "_bootstrap", "<frozen", "<string",
)


def _capture_call_site() -> Tuple[str, int, str, str]:
    """Return (filename, lineno, source_line, func_name) from the innermost
    model-code frame that triggered the current aten op."""
    frames = traceback.extract_stack()
    # Walk from most-recent (deepest) outward; skip torch/our internals.
    for frame in reversed(frames):
        fname = frame.filename or ""
        if fname and not any(m in fname for m in _SKIP_PATH_MARKERS):
            return (
                os.path.basename(fname),
                frame.lineno,
                (frame.line or "").strip(),
                frame.name,
            )
    return "", 0, "", ""


def _collect_extra_args(func, args, kwargs) -> str:
    """Return JSON KV string of non-tensor, non-None args, with schema-derived param names."""
    def _has_tensor(v) -> bool:
        if isinstance(v, torch.Tensor):
            return True
        if isinstance(v, (list, tuple)):
            return any(isinstance(i, torch.Tensor) for i in v)
        return False

    def _serialize(v):
        if isinstance(v, (bool, int, float, str)):
            return v
        return repr(v)

    result = {}
    try:
        params = func._schema.arguments
        for param, val in zip(params, args):
            if not _has_tensor(val) and val is not None:
                result[param.name] = _serialize(val)
    except Exception:
        for i, val in enumerate(args):
            if not _has_tensor(val) and val is not None:
                result[f"arg{i}"] = _serialize(val)

    for k, v in kwargs.items():
        if not _has_tensor(v) and v is not None:
            result[k] = _serialize(v)

    return json.dumps(result, ensure_ascii=False) if result else ""


class TensorTracker:
    """Assign stable unique IDs to tensors seen during tracing.

    id(tensor) is not reliable on meta device, so we maintain our own counter.
    """

    def __init__(self):
        self._counter = 0
        self._id_map: Dict[int, int] = {}

    def reset(self):
        self._counter = 0
        self._id_map.clear()

    def get_id(self, t: torch.Tensor) -> int:
        oid = id(t)
        if oid not in self._id_map:
            self._id_map[oid] = self._counter
            self._counter += 1
        return self._id_map[oid]


class RecordingDispatch(TorchDispatchMode):
    """Intercept every aten op and record its metadata."""

    def __init__(self, tensor_tracker: TensorTracker,
                 module_tracker: Optional[ModuleTracker] = None,
                 skip_reshapes: bool = True,
                 active: bool = True,
                 target_layers: Optional[List[int]] = None):
        super().__init__()
        self.tensor_tracker = tensor_tracker
        self.records: List[Dict[str, Any]] = []
        self._module_tracker = module_tracker
        self._skip_reshapes = skip_reshapes
        self.active = active
        self._target_layers: Optional[set] = (
            set(target_layers) if target_layers is not None else None
        )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)

        # Always track tensor IDs even when paused — this ensures forward tensors
        # seen during a paused-forward pass of train_backward get IDs that persist
        # when those same tensor objects appear as inputs to backward ops.
        input_tensors = collect_tensors(args, kwargs)
        output_tensors = collect_output_tensors(out)
        for t in input_tensors:
            self.tensor_tracker.get_id(t)
        for t in output_tensors:
            self.tensor_tracker.get_id(t)

        if not self.active:
            return out

        func_name = str(func.overloadpacket) + "." + func._overloadname
        try:
            op_short = func.overloadpacket.__name__
        except AttributeError:
            op_short = str(func.overloadpacket).split(".")[-1]

        input_ids = [self.tensor_tracker.get_id(t) for t in input_tensors]
        output_ids = [self.tensor_tracker.get_id(t) for t in output_tensors]

        if self._skip_reshapes and func_name in SKIP_OPS:
            return out

        module_path = ""
        module_class = ""
        if self._module_tracker:
            module_path = self._module_tracker.current_module
            module_class = self._module_tracker.current_module_class

        if self._target_layers is not None:
            layer_str = extract_layer_idx(module_path)
            if layer_str and int(layer_str) not in self._target_layers:
                return out

        input_shapes = [shape_str(t) for t in input_tensors]
        input_dtypes = [str(t.dtype) for t in input_tensors]
        output_shapes = [shape_str(t) for t in output_tensors]
        output_dtypes = [str(t.dtype) for t in output_tensors]

        src_file, src_line, src_code, src_func = _capture_call_site()
        extra_args = _collect_extra_args(func, args, kwargs)
        in_recompute = (
            self._module_tracker.in_recompute if self._module_tracker else False
        )

        self.records.append({
            "node_id": len(self.records),
            "op_short": op_short,
            "aten_op": func_name,
            "module_path": module_path,
            "module_class": module_class,
            "layer": extract_layer_idx(module_path),
            "component": classify_component(module_path, func_name),
            "src_file": src_file,
            "src_line": src_line,
            "src_code": src_code,
            "src_func": src_func,
            "extra_args": extra_args,
            "input_shapes": ", ".join(input_shapes),
            "input_dtypes": ", ".join(input_dtypes),
            "output_shapes": ", ".join(output_shapes),
            "output_dtypes": ", ".join(output_dtypes),
            "num_inputs": len(input_tensors),
            "num_outputs": len(output_tensors),
            "_input_ids": input_ids,
            "_output_ids": output_ids,
            "recompute": in_recompute,
        })

        return out
