from __future__ import annotations

from python.zrt.ir.graph import OpGraph
from python.zrt.transform.base import GraphPass
from python.zrt.transform.context import TransformContext


class ContextParallelPass(GraphPass):
    """Context Parallel pass for attention operations."""
    name = "context_parallel"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        """Run context parallel pass on the graph.
        
        Args:
            graph: Input OpGraph
            ctx: TransformContext with parallel config and training config
            
        Returns:
            New OpGraph with context parallel annotations
        """
        if ctx.parallel.cp <= 1:
            return graph
        
        g = graph.clone()
        cp = ctx.parallel.cp
        cp_kind = ctx.training.cp_kind if ctx.training else "none"

        for node in g.topo_sort():
            if not self._is_attention_op(node):
                continue

            if cp_kind == "ulysses":
                # Head-dim sharding: inputs arrive seq-sharded; A2A scatter-seq / gather-heads
                # before attn; inverse A2A after.
                node.annotations["cp_split"] = {
                    "kind": "ulysses",
                    "cp": cp
                }
                # Halve heads dim on Q/K/V/O view; multiply by CP on seq dim for attn core input
                # This will be handled by CommInserterPass later

            elif cp_kind == "ring":
                # Seq-dim sharding inside attention: N rounds of P2P send/recv of KV chunks
                # each round overlaps with a flash-attn tile computation
                node.annotations["cp_split"] = {
                    "kind": "ring",
                    "cp": cp,
                    "p2p_rounds": cp
                }
                # Ring attention will be handled by CommInserterPass later

            elif cp_kind == "hybrid":
                # Hybrid strategy combining Ulysses and Ring
                node.annotations["cp_split"] = {
                    "kind": "hybrid",
                    "cp": cp
                }

            elif cp_kind == "compressed":
                # DeepSeek-V4 two-stage compressed CP:
                # Stage 1: P2P boundary exchange (last m/m' uncompressed KV entries)
                # Stage 2: All-Gather to collect compressed KV from all CP ranks
                # CSA: compression_ratio=4, HCA: compression_ratio=128
                node.annotations["cp_split"] = {
                    "kind": "compressed",
                    "cp": cp,
                    "stage1": "p2p_boundary_exchange",
                    "stage2": "allgather_compressed_kv",
                }

        return g

    def _is_attention_op(self, node: OpNode) -> bool:
        """Check if a node is an attention operation.
        
        Args:
            node: OpNode to check
            
        Returns:
            True if the node is an attention operation, False otherwise
        """
        op_type = node.op_type
        scope = node.scope.lower()
        
        # Check for attention-related operations
        attention_op_types = {
            "aten._scaled_dot_product_attention",
            "aten.softmax",
            "flash_attn.flash_attn_func",
            "flash_attn.flash_attn_qkvpacked_func",
            "flash_attn.flash_attn_varlen_func",
            "flash_attn.flash_attn_varlen_qkvpacked_func"
        }
        
        # Check for attention-related scopes
        attention_scopes = {
            "self_attn",
            "attention",
            "attn"
        }
        
        return (
            op_type in attention_op_types or
            any(scope_part in scope for scope_part in attention_scopes)
        )


from python.zrt.ir.node import OpNode
