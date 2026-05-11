"""Track which nn.Module is currently executing via forward hooks."""
from __future__ import annotations

from typing import Any, Dict, List

import torch.nn as nn


class ModuleTracker:
    """Track which nn.Module is currently executing via forward hooks.

    Also records module metadata (class name, parent-child relationships)
    for automatic fusion rule discovery.
    """

    def __init__(self, root: nn.Module):
        self._stack: List[str] = []
        self._class_stack: List[str] = []
        self._handles: List[Any] = []
        self.path_to_class: Dict[str, str] = {}
        self.path_to_children: Dict[str, List[str]] = {}
        self._forward_depth: int = 0       # >0 while inside a module forward
        self._in_backward_phase: bool = False  # set externally before loss.backward()
        self._pre_backward_module: str = ""    # module path at backward entry
        self._bwd_expected_pop: int = 0    # pending backward post-hook pops
        self._install(root, "")

    def _install(self, module: nn.Module, prefix: str):
        child_paths = []
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            class_name = type(child).__name__
            self.path_to_class[full_name] = class_name
            child_paths.append(full_name)

            def _pre_hook(m, inp, _fn=full_name, _cls=class_name):
                self._stack.append(_fn)
                self._class_stack.append(_cls)
                self._forward_depth += 1

            def _post_hook(m, inp, out, _fn=full_name):
                if self._stack and self._stack[-1] == _fn:
                    self._stack.pop()
                    if self._class_stack:
                        self._class_stack.pop()
                self._forward_depth = max(0, self._forward_depth - 1)

            def _pre_bwd_hook(m, grad_out, _fn=full_name, _cls=class_name):
                # Track backward module entry (fires regardless of output type).
                # Clean up stale entries from sibling modules whose
                # register_full_backward_hook never fired (e.g. when output is
                # a NamedTuple like BaseModelOutputWithPast).
                while self._bwd_expected_pop > 0:
                    if self._stack and self._class_stack:
                        self._stack.pop()
                        self._class_stack.pop()
                    self._bwd_expected_pop -= 1
                self._stack.append(_fn)
                self._class_stack.append(_cls)
                self._bwd_expected_pop += 1

            def _post_bwd_hook(m, grad_in, grad_out, _fn=full_name):
                if self._stack and self._stack[-1] == _fn:
                    self._stack.pop()
                    if self._class_stack:
                        self._class_stack.pop()
                    self._bwd_expected_pop = max(0, self._bwd_expected_pop - 1)

            h1 = child.register_forward_pre_hook(_pre_hook)
            h2 = child.register_forward_hook(_post_hook)
            h3 = child.register_full_backward_pre_hook(_pre_bwd_hook)
            h4 = child.register_full_backward_hook(_post_bwd_hook)
            self._handles.extend([h1, h2, h3, h4])
            self._install(child, full_name)
        self.path_to_children[prefix] = child_paths

    @property
    def current_module(self) -> str:
        return self._stack[-1] if self._stack else ""

    @property
    def current_module_class(self) -> str:
        return self._class_stack[-1] if self._class_stack else ""

    @property
    def in_recompute(self) -> bool:
        """True when a forward pass is re-running inside backward (activation checkpointing).

        Relies on ``_in_backward_phase`` being set to True externally (by _trace_phase)
        before loss.backward() is called, and ``_forward_depth`` being reset to 0 at
        the same time.  Any forward hook that fires during backward means recompute.
        """
        return self._in_backward_phase and self._forward_depth > 0

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


class NullModuleTracker:
    """Drop-in for ModuleTracker when torch.compile graph capture is used.

    torch.compile tracing does not run forward hooks, so there is no live
    module context.  ExcelWriter and FusionEngine only read
    ``path_to_class`` / ``path_to_children`` — both are provided as empty
    dicts so all downstream code works without modification.
    """

    path_to_class: Dict[str, str] = {}
    path_to_children: Dict[str, List[str]] = {}
    current_module: str = ""
    current_module_class: str = ""

    def remove(self) -> None:
        pass
