from __future__ import annotations

import logging

from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.transform.base import GraphPass
from python.zrt.transform.context import TransformContext

logger = logging.getLogger(__name__)


class TrainFlopsPass(GraphPass):
    """TrainFlopsPass for training-aware FLOPs calculation."""
    name = "flops_train"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        """Run TrainFlopsPass on the graph.
        
        Args:
            graph: Input OpGraph
            ctx: TransformContext with training config
            
        Returns:
            New OpGraph with training FLOPs annotations
        """
        g = graph.clone()
        
        for node in g.nodes.values():
            # Calculate forward FLOPs and memory
            fwd_flops, read_bytes, write_bytes = self._calculate_fwd_flops(node, g)
            
            # Calculate gradient FLOPs (dx and dw)
            dx_flops, dw_flops = self._calculate_grad_flops(node, fwd_flops)
            
            # Apply recompute multiplier if needed
            rec_mult = 2.0 if node.annotations.get("recompute") else 1.0
            
            # Update node annotations
            node.annotations.update({
                "flops_fwd": int(fwd_flops * rec_mult),
                "flops_dx": int(dx_flops),
                "flops_dw": int(dw_flops),
                "read_bytes": int(read_bytes),
                "write_bytes": int(write_bytes),
                # Legacy key — sum of fwd/dx/dw for the relevant phase
                "flops": int(fwd_flops * rec_mult) if graph.phase == "train_forward"
                         else int(dx_flops + dw_flops),
            })
        
        return g

    def _calculate_fwd_flops(self, node: OpNode, graph: OpGraph) -> tuple[float, float, float]:
        """Calculate forward FLOPs and memory for a node.
        
        Args:
            node: OpNode to analyze
            
        Returns:
            Tuple of (fwd_flops, read_bytes, write_bytes)
        """
        op_type = node.op_type
        inputs = node.inputs
        outputs = node.outputs
        
        # Default values
        fwd_flops = 0.0
        read_bytes = 0.0
        write_bytes = 0.0
        
        # Calculate read bytes
        for inp in inputs:
            read_bytes += inp.mem_bytes

        # Calculate write bytes
        for out in outputs:
            write_bytes += out.mem_bytes
        
        # Calculate FLOPs based on op type
        if op_type in ("aten.mm", "aten.linear", "aten.addmm"):
            # Matmul: FLOPs = 2 * M * N * K
            if len(inputs) >= 2:
                # Assuming inputs[0] is (M, K), inputs[1] is (K, N)
                if len(inputs[0].shape) == 2 and len(inputs[1].shape) == 2:
                    M, K = inputs[0].shape
                    _, N = inputs[1].shape
                    fwd_flops = 2 * M * N * K
        
        elif "attention" in op_type.lower() or "attn" in op_type.lower():
            # Flash attention: FLOPs ≈ 2 * batch * seq_len^2 * heads * head_dim
            # Approximate based on input/output sizes
            if inputs and outputs:
                batch, seq_len, heads, head_dim = self._attention_dims(node, graph)
                if seq_len > 0 and heads > 0 and head_dim > 0:
                    compression = self._attention_compression_ratio(node, graph)
                    fwd_flops = 2 * batch * (seq_len ** 2) * heads * head_dim * compression
        
        elif "layer_norm" in op_type.lower() or "ln" in op_type.lower():
            # Layer norm: FLOPs ≈ 5 * N
            if inputs:
                N = 1
                for dim in inputs[0].shape:
                    N *= dim
                fwd_flops = 5 * N
        
        elif "softmax" in op_type.lower():
            # Softmax: FLOPs ≈ 5 * N
            if inputs:
                N = 1
                for dim in inputs[0].shape:
                    N *= dim
                fwd_flops = 5 * N
        
        elif "swiglu" in op_type.lower() or "gelu" in op_type.lower():
            # Activation functions: FLOPs ≈ N
            if inputs:
                N = 1
                for dim in inputs[0].shape:
                    N *= dim
                fwd_flops = N
        
        elif op_type.startswith("comm."):
            # Communication nodes: no FLOPs
            fwd_flops = 0.0
        
        elif op_type.startswith("optimizer."):
            # Optimizer nodes: FLOPs are already calculated in OptimizerPass
            fwd_flops = node.attrs.get("step_flops", 0.0)
        
        return fwd_flops, read_bytes, write_bytes

    def _attention_dims(self, node: OpNode, graph: OpGraph) -> tuple[int, int, int, int]:
        attrs = node.attrs
        metadata = graph.metadata
        inputs = node.inputs
        outputs = node.outputs

        batch = self._int_or_none(attrs.get("batch", attrs.get("b")))
        seq_len = self._int_or_none(attrs.get("seq_len", attrs.get("s")))
        heads = self._int_or_none(attrs.get("heads", attrs.get("num_heads")))
        head_dim = self._int_or_none(attrs.get("head_dim"))

        if batch is None:
            batch = self._int_or_none(metadata.get("batch", metadata.get("batch_size")))
        if seq_len is None:
            seq_len = self._int_or_none(metadata.get("seq_len"))
        if heads is None:
            heads = self._int_or_none(metadata.get("num_heads", metadata.get("heads")))
        if head_dim is None:
            head_dim = self._int_or_none(metadata.get("head_dim"))

        if inputs:
            q_shape = inputs[0].shape
            if len(q_shape) >= 4:
                batch = batch or int(q_shape[0])
                heads = heads or int(q_shape[1])
                seq_len = seq_len or int(q_shape[2])
                head_dim = head_dim or int(q_shape[3])
            elif len(q_shape) >= 3:
                batch = batch or int(q_shape[0])
                seq_len = seq_len or int(q_shape[1])

        hidden_size = None
        if outputs and outputs[0].shape:
            hidden_size = int(outputs[0].shape[-1])
        elif inputs and inputs[0].shape:
            hidden_size = int(inputs[0].shape[-1])

        if head_dim is None and heads and hidden_size:
            head_dim = max(1, hidden_size // heads)
        if head_dim is None:
            head_dim = 64
        if heads is None and hidden_size and head_dim > 0:
            heads = max(1, hidden_size // head_dim)

        return batch or 1, seq_len or 1, heads or 0, head_dim or 0

    def _attention_compression_ratio(self, node: OpNode, graph: OpGraph) -> float:
        value = node.annotations.get("attn_compression_ratio")
        if value is None:
            value = node.attrs.get("attn_compression_ratio")
        if value is None:
            value = graph.metadata.get("attn_compression_ratio", 1.0)
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid attn_compression_ratio=%r on node %s; using dense attention ratio 1.0",
                value, node.id,
            )
            return 1.0
        if not (0.0 < ratio <= 1.0):
            logger.warning(
                "Invalid attn_compression_ratio=%r on node %s; using dense attention ratio 1.0",
                value, node.id,
            )
            return 1.0
        return ratio

    @staticmethod
    def _int_or_none(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _calculate_grad_flops(self, node: OpNode, fwd_flops: float) -> tuple[float, float]:
        """Calculate gradient FLOPs (dx and dw) for a node.
        
        Args:
            node: OpNode to analyze
            fwd_flops: Forward FLOPs
            
        Returns:
            Tuple of (dx_flops, dw_flops)
        """
        op_type = node.op_type
        
        # Default values
        dx_flops = 0.0
        dw_flops = 0.0
        
        if op_type in ("aten.mm", "aten.linear", "aten.addmm"):
            # Matmul: dx = 2 * M * N * K, dw = 2 * M * N * K
            dx_flops = fwd_flops
            dw_flops = fwd_flops
        
        elif "attention" in op_type.lower() or "attn" in op_type.lower():
            # Flash attention: dx ≈ 2.5 * fwd, dw = 0
            dx_flops = 2.5 * fwd_flops
            dw_flops = 0.0
        
        elif "layer_norm" in op_type.lower() or "ln" in op_type.lower():
            # Layer norm: dx = fwd, dw = 0
            dx_flops = fwd_flops
            dw_flops = 0.0
        
        elif "softmax" in op_type.lower():
            # Softmax: dx = fwd, dw = 0
            dx_flops = fwd_flops
            dw_flops = 0.0
        
        elif "swiglu" in op_type.lower() or "gelu" in op_type.lower():
            # Activation functions: dx = fwd, dw = 0
            dx_flops = fwd_flops
            dw_flops = 0.0
        
        elif op_type.startswith("comm."):
            # Communication nodes: no gradient FLOPs
            dx_flops = 0.0
            dw_flops = 0.0
        
        elif op_type.startswith("optimizer."):
            # Optimizer nodes: no gradient FLOPs (handled separately)
            dx_flops = 0.0
            dw_flops = 0.0
        
        return dx_flops, dw_flops
