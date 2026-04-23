from __future__ import annotations

from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.transform.base import GraphPass
from python.zrt.transform.context import TransformContext


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
            fwd_flops, read_bytes, write_bytes = self._calculate_fwd_flops(node)
            
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

    def _calculate_fwd_flops(self, node: OpNode) -> tuple[float, float, float]:
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
                batch = inputs[0].shape[0] if len(inputs[0].shape) > 0 else 1
                seq_len = inputs[0].shape[1] if len(inputs[0].shape) > 1 else 1
                # Estimate heads and head_dim from output shape
                if len(outputs[0].shape) == 3:
                    _, _, hidden_size = outputs[0].shape
                    # Assume head_dim = 64 (common value)
                    head_dim = 64
                    heads = hidden_size // head_dim
                    fwd_flops = 2 * batch * (seq_len ** 2) * heads * head_dim
        
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
