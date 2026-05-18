"""Context Parallel pass: splits seq dimension by CP factor.

Design (Simplified):
- 规则：只要 tensor shape 中包含 seq_len，就切分
- 排除：权重节点(is_param)、通信节点、optimizer节点
- 切分方式：将所有匹配 seq_len 的维度切分为 seq_len/cp

会被切分的算子（shape包含seq_len）:
  - Embedding (输出)
  - QKV 线性层
  - Attention 全套
  - LayerNorm/RMSNorm
  - MLP/FFN
  - Dropout
  - RoPE
  - Activations (SiLU, etc.)
  
不会被切分的算子:
  - 权重/参数 (is_param=True)
  - Loss/Optimizer
  - 标量/常量操作
  - 通信节点
"""
from __future__ import annotations

import logging
from typing import Dict, List

from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.ir.types import TensorMeta
from python.zrt.transform.base import GraphPass
from python.zrt.transform.context import TransformContext

logger = logging.getLogger(__name__)


class ContextParallelPass(GraphPass):
    """Context Parallel pass: split all tensors containing seq_len.
    
    Simplified rule: if any tensor shape dimension equals seq_len, split it.
    """
    name = "context_parallel"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        if ctx.parallel.cp <= 1:
            return graph
        
        g = graph.clone()
        cp = ctx.parallel.cp
        
        if ctx.training:
            cp_kind = ctx.training.resolve_cp_kind(ctx.model_id, cp)
        else:
            cp_kind = "ulysses"
        
        seq_len = ctx.training.seq_len if ctx.training else 2048
        
        nodes_to_split = self._identify_nodes_with_seq_len(g, seq_len)
        
        g, split_nodes = self._split_tensor_shapes(g, nodes_to_split, seq_len, cp, cp_kind)
        
        for node in split_nodes:
            node.annotations["cp_split"] = {
                "kind": cp_kind,
                "cp": cp,
            }
        
        return g

    def _identify_nodes_with_seq_len(self, graph: OpGraph, seq_len: int) -> List[OpNode]:
        """识别所有含 seq_len tensor 的节点，切分这些 tensor。
        
        规则（简化）：
        1. 节点的输入/输出中，只要有含 seq_len 的 tensor，就处理
        2. 不过滤权重算子（w1/w2/linear 等），让激活 tensor 被切分
        3. 权重 tensor（不含 seq_len）自然不会被切分
        4. 排除通信节点
        
        这样：
        - w1/w2/linear: 激活 tensor 被切分，权重 tensor 不变
        - silu/layernorm: 所有 tensor 被切分
        """
        candidates = []
        
        for node in graph.topo_sort():
            # 排除通信节点
            if node.category == "communication":
                continue
            
            # 检查是否有含 seq_len 的 tensor
            has_seq = any(seq_len in tensor.shape for tensor in node.inputs + node.outputs)
            
            if has_seq:
                candidates.append(node)
        
        logger.info(f"ContextParallelPass identified {len(candidates)} nodes with seq_len tensor")
        if candidates:
            sample_shapes = []
            for n in candidates[:5]:
                shapes = [t.shape for t in n.inputs[:2] if seq_len in t.shape]
                if shapes:
                    sample_shapes.append(f"{n.op_type}: {shapes[0]}")
            logger.info(f"  Sample: {sample_shapes}")
        
        return candidates

    def _split_tensor_shapes(
        self, 
        graph: OpGraph, 
        nodes_to_split: List[OpNode],
        seq_len: int,
        cp: int,
        cp_kind: str,
    ) -> tuple[OpGraph, List[OpNode]]:
        seq_local = seq_len // cp
        tensor_map: Dict[str, TensorMeta] = {}
        split_nodes: List[OpNode] = []
        
        for node in nodes_to_split:
            new_inputs = [
                self._split_seq_dim(t, seq_len, seq_local, tensor_map)
                for t in node.inputs
            ]
            new_outputs = [
                self._split_seq_dim(t, seq_len, seq_local, tensor_map)
                for t in node.outputs
            ]
            
            new_node = OpNode(
                id=node.id,
                op_type=node.op_type,
                inputs=new_inputs,
                outputs=new_outputs,
                attrs=node.attrs,
                scope=node.scope,
                layer=node.layer,
                category=node.category,
                annotations=node.annotations,
                op_short=node.op_short,
                module_class=node.module_class,
                component=node.component,
                name=node.name,
                provenance=node.provenance,
                src_file=node.src_file,
                src_line=node.src_line,
                src_code=node.src_code,
                call_id=node.call_id,
                fused_from=node.fused_from,
                num_sub_ops=node.num_sub_ops,
                fusion_level=node.fusion_level,
            )
            graph.nodes[node.id] = new_node
            
            if any(t.id in tensor_map for t in node.inputs + node.outputs):
                split_nodes.append(new_node)
        
        for edge in graph.edges:
            if edge.tensor and edge.tensor.id in tensor_map:
                edge.tensor = tensor_map[edge.tensor.id]
        
        return graph, split_nodes

    def _split_seq_dim(
        self, 
        tensor: TensorMeta, 
        seq_len: int,
        seq_local: int,
        tensor_map: Dict[str, TensorMeta],
    ) -> TensorMeta:
        """Split dimensions matching seq_len.
        
        规则：
        - 2D tensor (seq, hidden): dim[0] 是 seq，切分
        - 3D+ tensor: dim[0] 是 batch/heads，不切；从 dim[1] 开始检查
        """
        if tensor.id in tensor_map:
            return tensor_map[tensor.id]
        
        shape = tensor.shape
        new_shape = list(shape)
        found_seq = False
        
        # 2D tensor: dim[0] 可能是 seq，检查并切分
        if len(shape) == 2 and shape[0] == seq_len:
            new_shape[0] = seq_local
            found_seq = True
        
        # 3D+ tensor: dim[0] 是 batch/heads，跳过；从 dim[1] 开始检查
        for i in range(1, len(shape)):
            if shape[i] == seq_len:
                new_shape[i] = seq_local
                found_seq = True
        
        if not found_seq:
            return tensor
        
        from math import prod
        new_shape_tuple = tuple(new_shape)
        new_numel = prod(new_shape_tuple) if new_shape_tuple else 1
        new_bytes = int(new_numel * tensor.dtype.itemsize)
        
        new_tensor = TensorMeta(
            id=tensor.id,
            shape=new_shape_tuple,
            dtype=tensor.dtype,
            mem_bytes=new_bytes,
        )
        
        tensor_map[tensor.id] = new_tensor
        return new_tensor