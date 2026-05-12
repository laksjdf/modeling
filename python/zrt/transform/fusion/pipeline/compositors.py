"""Post-fusion compositors (Add+Norm)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph


_ADD_NORM_RULE_NAME = "add_norm"


def _compose_add_norm(graph: "OpGraph") -> "OpGraph":
    """Merge adjacent Add + RMSNorm/LayerNorm nodes into AddNorm.

    Operates on already-fused nodes (not raw aten ops).  Looks for
    ``op_type`` ending in "add" followed by one ending in "norm".
    """
    from python.zrt.ir.node import OpNode

    topo = graph.topo_sort()
    if len(topo) < 2:
        return graph

    norm_types = {"rms_norm", "layer_norm", "RMSNorm", "LayerNorm"}
    add_types = {"add", "residual_add"}

    pairs: list[tuple[int, int]] = []
    for i in range(len(topo) - 1):
        a, b = topo[i], topo[i + 1]
        a_label = a.op_type.lower()
        b_label = b.op_type.lower()
        if (a_label in add_types or a_label.endswith("add")) and \
           (b_label in norm_types or b_label.endswith("norm")):
            has_edge = any(
                e.src == a.id and e.dst == b.id for e in graph.edges
            )
            if has_edge:
                pairs.append((i, i + 1))

    if not pairs:
        return graph

    used: set[str] = set()
    for add_i, norm_i in reversed(pairs):
        add_node = topo[add_i]
        norm_node = topo[norm_i]
        if add_node.id in used or norm_node.id in used:
            continue

        merged = OpNode(
            id=f"composed_{add_node.id}_{norm_node.id}",
            op_type="AddNorm",
            inputs=add_node.inputs,
            outputs=norm_node.outputs,
            scope=norm_node.scope,
            category="compute",
            module_class=norm_node.module_class,
            layer=norm_node.layer,
            component=norm_node.component,
            fused_from=add_node.fused_from + norm_node.fused_from,
            num_sub_ops=add_node.num_sub_ops + norm_node.num_sub_ops,
            fusion_level="parent",
            name=norm_node.name,
        )
        merged.annotations.update(add_node.annotations)
        merged.annotations.update(norm_node.annotations)
        merged.annotations["source_op_ids"] = (
            list(add_node.annotations.get("source_op_ids", [add_node.id]))
            + list(norm_node.annotations.get("source_op_ids", [norm_node.id]))
        )
        merged.annotations["fused_by_rule"] = _ADD_NORM_RULE_NAME
        graph.replace_subgraph({add_node.id, norm_node.id}, merged)
        used.add(add_node.id)
        used.add(norm_node.id)

    return graph
