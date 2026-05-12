from __future__ import annotations
import json, logging, sys
logging.basicConfig(level=logging.WARNING)

from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.ir.edge import Edge
from python.zrt.ir.types import TensorMeta, DType

with open('output/deepseek_v4/deepseek_v4_train_forward_raw_graph.json') as f:
    raw = json.load(f)
records = raw.get('nodes', raw)

def _t(tid, shape=(1,128)):
    return TensorMeta.from_shape_dtype(tid, shape, DType.BF16)

nodes = {}
for i, rec in enumerate(records):
    nid = f"op_{i}"
    n = OpNode(
        id=nid, op_type=rec['op_type'],
        inputs=[_t(f"{nid}_in")], outputs=[_t(f"{nid}_out")],
        scope=rec.get('module_path', ''), layer=rec.get('layer', ''),
        module_class=rec.get('module_class', ''),
        category=rec.get('component', 'compute'),
    )
    nodes[nid] = n

edges = []
for i in range(len(records)-1):
    edges.append(Edge(src=f"op_{i}", src_idx=0, dst=f"op_{i+1}", dst_idx=0, tensor=_t('e')))

graph = OpGraph(name='test', phase='train_forward', nodes=nodes, edges=edges)

from python.zrt.transform.fusion.algorithm import bucket_nodes_by_leaf_module, _merge_parent_groups, _find_rule
from python.zrt.transform.fusion.platforms import load_platform_rules
load_platform_rules('hf_models/deepseek_v4')

groups = bucket_nodes_by_leaf_module(graph)
merged = _merge_parent_groups(groups, graph)
print(f"Leaf groups: {len(groups)}, Merged: {len(merged)}")

matched = unmatched = 0
for g in merged:
    if len(g.child_ops) <= 1:
        continue
    seq = tuple(op.op_type for op in g.child_ops)
    rule = _find_rule(g)
    if rule:
        matched += 1
    else:
        unmatched += 1
        if unmatched <= 12:
            print(f"\nUNMATCHED {g.module_class} [{len(seq)}] @ {g.scope}")
            for s in seq[:5]:
                print(f"  {s}")
            if len(seq) > 10:
                print(f"  ...({len(seq)-10} more)...")
                for s in seq[-5:]:
                    print(f"  {s}")
            else:
                for s in seq[5:]:
                    print(f"  {s}")

print(f"\nMatched: {matched}, Unmatched: {unmatched}")
