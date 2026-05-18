"""End-to-end test for training modeling pipeline (Path 2).

Tests DeepSeek V3, V3.2, and V4 models to verify:
1. TP shape splitting correctness across different TP degrees (1, 2, 4)
2. Communication operator insertion and ordering
3. Complete training report generation

Run with:
    .\run_pytest.bat tests/IT/test_training_tp_e2e.py -v
"""
import pytest
from pathlib import Path
import tempfile

from python.zrt.pipeline import run_trace_phases
from python.zrt.transform.analysis import estimate_training_from_graphs
import python.zrt.hardware.registry as hw_registry


MODEL_CONFIGS = {
    "deepseek_v3": {
        "model_id": "hf_models/deepseek_v3",
        "hidden_size": 7168,
        "num_heads": 128,
        "description": "DeepSeek V3 (MLA attention)",
        "key_ops": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "attention_type": "MLA",
        "has_moe": True,
    },
    "deepseek_v3_2": {
        "model_id": "hf_models/deepseek_v3_2",
        "hidden_size": 7168,
        "num_heads": 128,
        "description": "DeepSeek V3.2 (MLA + Index attention)",
        "key_ops": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "attention_type": "MLA + Index",
        "has_moe": True,
    },
    "deepseek_v4": {
        "model_id": "hf_models/deepseek_v4",
        "hidden_size": 7168,
        "num_heads": 128,
        "description": "DeepSeek V4 (MLA 2.0 + LongMoE)",
        "key_ops": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "attention_type": "MLA 2.0",
        "has_moe": True,
    },
}


# Row parallel operators that require all_reduce after them
# From: python/zrt/transform/parallel/tensor_parallel.py
ROW_PARALLEL_OPS = ("o_proj", "down_proj", "w2")


@pytest.fixture(scope="module", params=list(MODEL_CONFIGS.keys()))
def training_graphs(request):
    """Capture training graphs once per model and reuse across tests."""
    model_key = request.param
    config = MODEL_CONFIGS[model_key]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_trace_phases(
            model_id=config["model_id"],
            num_layers=2,
            batch_size=1,
            seq_len=128,
            phases=["train_forward", "train_backward"],
            output_dir=tmpdir,
        )
        yield model_key, result.graphs


def run_tp_analysis(
    forward_graph,
    backward_graph,
    tp: int,
    hidden: int,
    num_layers: int,
):
    """Run training analysis with specified TP degree."""
    hw_spec = hw_registry.load("nvidia_h100_sxm")
    report, ctx, transformed = estimate_training_from_graphs(
        forward_graph=forward_graph,
        backward_graph=backward_graph,
        hw_spec=hw_spec,
        tp=tp,
        pp=1,
        dp=1,
        seq_len=128,
        batch_size=1,
        hidden=hidden,
        num_layers=num_layers,
        return_transformed=True,
    )
    return report, transformed


def extract_projection_nodes(graph, proj_type: str):
    """Extract projection nodes by type (q_proj, k_proj, v_proj, o_proj)."""
    nodes = []
    for node_id, node in graph.nodes.items():
        scope = getattr(node, 'scope', '')
        if proj_type in scope and "self_attn" in scope:
            nodes.append({
                "node_id": node_id,
                "node": node,
                "op_type": node.op_type,
                "scope": scope,
                "inputs": [list(inp.shape) if hasattr(inp, 'shape') else [] for inp in getattr(node, 'inputs', [])],
                "outputs": [list(out.shape) if hasattr(out, 'shape') else [] for out in getattr(node, 'outputs', [])],
            })
    return nodes


def extract_communication_nodes(graph):
    """Extract communication operators with their context."""
    comm_nodes = []
    for node_id, node in graph.nodes.items():
        op_type = getattr(node, 'op_type', '')
        if op_type.startswith("comm.") or "allreduce" in node_id.lower():
            comm_nodes.append({
                "node_id": node_id,
                "node": node,
                "op_type": op_type,
                "scope": getattr(node, 'scope', ''),
                "inputs": [list(inp.shape) if hasattr(inp, 'shape') else [] for inp in getattr(node, 'inputs', [])],
                "outputs": [list(out.shape) if hasattr(out, 'shape') else [] for out in getattr(node, 'outputs', [])],
            })
    return comm_nodes


def find_row_parallel_nodes(graph):
    """Find row parallel nodes (o_proj, down_proj, w2) that need all_reduce."""
    row_parallel_nodes = []
    for node_id, node in graph.nodes.items():
        scope = getattr(node, 'scope', '').lower()
        # Check if this node is a row parallel operator (excluding allreduce nodes)
        if any(op in scope for op in ROW_PARALLEL_OPS) and "allreduce" not in node_id.lower():
            row_parallel_nodes.append({
                "node_id": node_id,
                "node": node,
                "scope": getattr(node, 'scope', ''),
                "op_type": getattr(node, 'op_type', ''),
            })
    return row_parallel_nodes


def find_allreduce_nodes(graph):
    """Find all all_reduce communication nodes."""
    all_reduce_nodes = []
    for node_id, node in graph.nodes.items():
        op_type = getattr(node, 'op_type', '')
        # Check for comm.all_reduce or nodes with allreduce in their id
        if op_type == "comm.all_reduce" or "allreduce" in node_id.lower():
            all_reduce_nodes.append({
                "node_id": node_id,
                "node": node,
                "op_type": op_type,
                "scope": getattr(node, 'scope', ''),
            })
    return all_reduce_nodes


class TestTPComparison:
    """End-to-end TP validation tests with comprehensive shape comparison."""

    def test_tp_shape_splitting_comparison(self, training_graphs):
        """Compare TP=1, TP=2, TP=4 shape splitting for projection layers."""
        model_key, graphs = training_graphs
        config = MODEL_CONFIGS[model_key]
        hidden_size = config["hidden_size"]
        
        fwd_graph = graphs["train_forward"]
        bwd_graph = graphs["train_backward"]

        # Run analysis for different TP degrees
        results = {}
        for tp in [1, 2, 4]:
            report, transformed = run_tp_analysis(
                fwd_graph, bwd_graph, tp=tp,
                hidden=hidden_size, num_layers=2
            )
            unified_graph = transformed.get("unified")
            assert unified_graph is not None, f"Unified graph should exist for TP={tp}"
            
            results[tp] = {
                "report": report,
                "unified_graph": unified_graph,
                "q_proj": extract_projection_nodes(unified_graph, "q_proj"),
                "k_proj": extract_projection_nodes(unified_graph, "k_proj"),
                "v_proj": extract_projection_nodes(unified_graph, "v_proj"),
                "o_proj": extract_projection_nodes(unified_graph, "o_proj"),
                "comm": extract_communication_nodes(unified_graph),
            }

        # Verify shape splitting for column parallel ops (q, k, v projections)
        for proj_type in ["q_proj", "k_proj", "v_proj"]:
            tp1_nodes = results[1][proj_type]
            if not tp1_nodes:
                continue  # Skip if no nodes found
            
            tp1_dim = tp1_nodes[0]["outputs"][0][-1] if (tp1_nodes[0]["outputs"] and tp1_nodes[0]["outputs"][0]) else 0
            
            for tp in [1, 2, 4]:
                expected_dim = tp1_dim // tp
                nodes = results[tp][proj_type]
                
                for node in nodes:
                    if node["outputs"]:
                        actual_dim = node["outputs"][0][-1] if node["outputs"][0] else 0
                        assert actual_dim == expected_dim, \
                            f"{proj_type} output dim should be {expected_dim} for TP={tp}, got {actual_dim}"

        # Verify communication operators scale with TP
        tp1_comm_count = len(results[1]["comm"])
        tp2_comm_count = len(results[2]["comm"])
        tp4_comm_count = len(results[4]["comm"])
        
        assert tp1_comm_count == 0, f"TP=1 should have 0 comm ops, got {tp1_comm_count}"
        assert tp2_comm_count > 0, f"TP=2 should have comm ops"
        assert tp4_comm_count >= tp2_comm_count, f"TP=4 should have >= TP=2 comm ops"

    def test_tp_communication_operator_ordering(self, training_graphs):
        """Verify communication operators are inserted and have correct structure."""
        model_key, graphs = training_graphs
        config = MODEL_CONFIGS[model_key]
        hidden_size = config["hidden_size"]
        
        fwd_graph = graphs["train_forward"]
        bwd_graph = graphs["train_backward"]

        # Run TP=2 analysis
        _, transformed = run_tp_analysis(
            fwd_graph, bwd_graph, tp=2,
            hidden=hidden_size, num_layers=2
        )
        unified_graph = transformed.get("unified")

        # Extract all nodes with their order
        nodes_ordered = list(unified_graph.nodes.values())
        
        # Find communication nodes and their positions
        comm_positions = []
        for idx, node in enumerate(nodes_ordered):
            op_type = getattr(node, 'op_type', '')
            if op_type.startswith("comm."):
                comm_positions.append({
                    "index": idx,
                    "node_id": node.id,
                    "op_type": op_type,
                    "scope": getattr(node, 'scope', ''),
                })

        # Verify communication operators exist
        assert len(comm_positions) > 0, "Should have communication operators for TP=2"

        # Verify all_reduce operators exist
        all_reduce_count = sum(1 for c in comm_positions if "all_reduce" in c["op_type"])
        assert all_reduce_count > 0, "Should have all_reduce operators for TP=2"

        # Verify all_reduce operators have correct structure
        for comm in comm_positions:
            if "all_reduce" in comm["op_type"]:
                node = nodes_ordered[comm["index"]]
                inputs = getattr(node, 'inputs', [])
                outputs = getattr(node, 'outputs', [])
                assert len(inputs) > 0, f"all_reduce should have inputs"
                assert len(outputs) > 0, f"all_reduce should have outputs"

    def test_tp_allreduce_inserted_after_row_parallel_ops(self, training_graphs):
        """Verify all_reduce operators are inserted after row-parallel ops (o_proj, down_proj, w2).
        
        According to CommInserterPass logic:
        - Row-parallel linear nodes (o_proj, down_proj, w2) have tp_split["comm_after"] = "all_reduce"
        - CommInserterPass inserts all_reduce after these nodes
        """
        model_key, graphs = training_graphs
        config = MODEL_CONFIGS[model_key]
        hidden_size = config["hidden_size"]
        
        fwd_graph = graphs["train_forward"]
        bwd_graph = graphs["train_backward"]

        # Run TP=2 analysis
        _, transformed = run_tp_analysis(
            fwd_graph, bwd_graph, tp=2,
            hidden=hidden_size, num_layers=2
        )
        unified_graph = transformed.get("unified")

        # Find row parallel nodes (excluding nodes that are already allreduce)
        row_parallel_nodes = find_row_parallel_nodes(unified_graph)
        
        # Find all_reduce nodes
        all_reduce_nodes = find_allreduce_nodes(unified_graph)
        all_nodes = {node["node_id"]: node["node"] for node in all_reduce_nodes}
        
        # Also add all regular nodes to all_nodes for lookup
        for node_id, node in unified_graph.nodes.items():
            if node_id not in all_nodes:
                all_nodes[node_id] = node

        assert len(all_reduce_nodes) > 0, "Should have all_reduce nodes for TP=2"

        # Build adjacency for reachability check
        reachable_from = {}
        for node_id in unified_graph.nodes:
            reachable_from[node_id] = set()
            stack = [node_id]
            visited = set([node_id])
            while stack:
                current = stack.pop()
                successors = unified_graph.successors(current)
                for succ in successors:
                    if succ not in visited:
                        visited.add(succ)
                        reachable_from[node_id].add(succ)
                        stack.append(succ)

        # Verify all_reduce nodes are reachable from row-parallel ops
        row_parallel_ids = [n["node_id"] for n in row_parallel_nodes]
        all_reduce_ids = [n["node_id"] for n in all_reduce_nodes]
        
        reachable_all_reduce = set()
        for row_id in row_parallel_ids:
            reachable_all_reduce.update(reachable_from[row_id] & set(all_reduce_ids))
        
        assert len(reachable_all_reduce) > 0, \
            f"At least one all_reduce should be reachable from row-parallel nodes"

        # Verify each all_reduce has a row-parallel predecessor in the scope
        for all_reduce in all_reduce_nodes:
            predecessors = list(unified_graph.predecessors(all_reduce["node_id"]))
            
            has_row_parallel_predecessor = False
            for pred_id in predecessors:
                pred_node = all_nodes.get(pred_id)
                if pred_node:
                    scope = getattr(pred_node, 'scope', '').lower()
                    # Check if predecessor is a row-parallel op or has similar scope
                    if any(op in scope for op in ROW_PARALLEL_OPS):
                        has_row_parallel_predecessor = True
                        break
            
            # Also check by scope similarity
            if not has_row_parallel_predecessor:
                reduce_scope = all_reduce["scope"].lower()
                for row_node in row_parallel_nodes:
                    row_scope = row_node["scope"].lower()
                    # Check if they share the same layer scope
                    if "layer" in row_scope and "layer" in reduce_scope:
                        # Extract layer number from both scopes
                        import re
                        row_layer = re.search(r'layer[s]?\.(\d+)', row_scope)
                        reduce_layer = re.search(r'layer[s]?\.(\d+)', reduce_scope)
                        if row_layer and reduce_layer and row_layer.group(1) == reduce_layer.group(1):
                            has_row_parallel_predecessor = True
                            break
            
            assert has_row_parallel_predecessor, \
                f"all_reduce {all_reduce['node_id']} should have row-parallel op in the same layer"

    def test_tp_allreduce_annotations(self, training_graphs):
        """Verify all_reduce nodes have correct annotations from CommInserterPass."""
        model_key, graphs = training_graphs
        config = MODEL_CONFIGS[model_key]
        hidden_size = config["hidden_size"]
        
        fwd_graph = graphs["train_forward"]
        bwd_graph = graphs["train_backward"]

        # Run TP=2 analysis
        _, transformed = run_tp_analysis(
            fwd_graph, bwd_graph, tp=2,
            hidden=hidden_size, num_layers=2
        )
        unified_graph = transformed.get("unified")

        # Find all_reduce nodes and verify annotations
        for node_id, node in unified_graph.nodes.items():
            op_type = getattr(node, 'op_type', '')
            if op_type == "comm.all_reduce":
                # Verify annotations from CommInserterPass
                inserted_by = node.annotations.get("inserted_by")
                assert inserted_by == "tp_pass", \
                    f"all_reduce {node_id} should have inserted_by='tp_pass', got {inserted_by}"
                
                # Verify attributes
                group_size = node.attrs.get("group_size")
                assert group_size == 2, \
                    f"all_reduce {node_id} should have group_size=2 for TP=2, got {group_size}"
                
                collective = node.attrs.get("collective")
                assert collective == "all_reduce", \
                    f"all_reduce {node_id} should have collective='all_reduce', got {collective}"

    def test_tp_communication_operator_execution_order(self, training_graphs):
        """Verify communication operators appear in correct execution order sequence.
        
        According to topological order:
        - Row-parallel op (e.g., o_proj) should come BEFORE its all_reduce
        - all_reduce should come AFTER its row-parallel predecessor
        """
        model_key, graphs = training_graphs
        config = MODEL_CONFIGS[model_key]
        hidden_size = config["hidden_size"]
        
        fwd_graph = graphs["train_forward"]
        bwd_graph = graphs["train_backward"]

        # Run TP=2 analysis
        _, transformed = run_tp_analysis(
            fwd_graph, bwd_graph, tp=2,
            hidden=hidden_size, num_layers=2
        )
        unified_graph = transformed.get("unified")

        # Get topological order
        try:
            topo_order = list(unified_graph.topological_sort())
        except AttributeError:
            topo_order = list(unified_graph.nodes.keys())

        # Build position map
        node_positions = {node_id: idx for idx, node_id in enumerate(topo_order)}

        # Find row parallel and all_reduce nodes
        row_parallel_positions = {}
        all_reduce_positions = {}
        
        for node_id, node in unified_graph.nodes.items():
            scope = getattr(node, 'scope', '').lower()
            op_type = getattr(node, 'op_type', '')
            node_id_lower = node_id.lower()
            
            # Row parallel ops (excluding allreduce nodes)
            if any(op in scope for op in ROW_PARALLEL_OPS) and "allreduce" not in node_id_lower:
                row_parallel_positions[node_id] = node_positions[node_id]
            # All_reduce ops
            elif op_type == "comm.all_reduce" or "allreduce" in node_id_lower:
                all_reduce_positions[node_id] = node_positions[node_id]

        # At least one of these should exist
        total_found = len(row_parallel_positions) + len(all_reduce_positions)
        assert total_found > 0, "Should have row-parallel or all_reduce nodes"

        # If we have both types, verify execution order
        if row_parallel_positions and all_reduce_positions:
            import re
            
            # Group by layer
            layer_nodes = {}
            
            for node_id, pos in {**row_parallel_positions, **all_reduce_positions}.items():
                node = unified_graph.nodes[node_id]
                scope = getattr(node, 'scope', '')
                match = re.search(r'layer[s]?\.(\d+)', scope.lower())
                if match:
                    layer_num = int(match.group(1))
                    if layer_num not in layer_nodes:
                        layer_nodes[layer_num] = {'row': [], 'reduce': []}
                    
                    if node_id in row_parallel_positions:
                        layer_nodes[layer_num]['row'].append((node_id, pos))
                    else:
                        layer_nodes[layer_num]['reduce'].append((node_id, pos))

            # For each layer, verify all_reduce comes after row-parallel ops
            for layer_num, nodes in layer_nodes.items():
                if nodes['row'] and nodes['reduce']:
                    min_row_pos = min(pos for _, pos in nodes['row'])
                    min_reduce_pos = min(pos for _, pos in nodes['reduce'])
                    
                    assert min_reduce_pos > min_row_pos, \
                        f"In layer {layer_num}: all_reduce should come after row-parallel ops, " \
                        f"but row at pos {min_row_pos} and reduce at pos {min_reduce_pos}"

    def test_tp_communication_operator_properties(self, training_graphs):
        """Verify communication operators have correct properties for row parallel computation."""
        model_key, graphs = training_graphs
        config = MODEL_CONFIGS[model_key]
        hidden_size = config["hidden_size"]
        
        fwd_graph = graphs["train_forward"]
        bwd_graph = graphs["train_backward"]

        # Run TP=2 analysis
        _, transformed = run_tp_analysis(
            fwd_graph, bwd_graph, tp=2,
            hidden=hidden_size, num_layers=2
        )
        unified_graph = transformed.get("unified")

        # Find all communication nodes
        comm_nodes = extract_communication_nodes(unified_graph)

        assert len(comm_nodes) > 0, "Should have communication nodes for TP=2"

        # Verify each comm node has proper connections
        for comm in comm_nodes:
            # Check predecessors exist
            predecessors = list(unified_graph.predecessors(comm["node_id"]))
            assert len(predecessors) > 0, \
                f"Comm node {comm['node_id']} should have predecessors"
            
            # Check successors exist
            successors = list(unified_graph.successors(comm["node_id"]))
            assert len(successors) > 0, \
                f"Comm node {comm['node_id']} should have successors"

        # For all_reduce specifically, verify input/output shapes match
        for comm in comm_nodes:
            if "all_reduce" in comm["op_type"] or "allreduce" in comm["node_id"].lower():
                inputs = comm["inputs"]
                outputs = comm["outputs"]
                
                assert len(inputs) > 0, f"all_reduce {comm['node_id']} should have inputs"
                assert len(outputs) > 0, f"all_reduce {comm['node_id']} should have outputs"
                
                if inputs and outputs:
                    input_shape = list(inputs[0].shape) if hasattr(inputs[0], 'shape') else []
                    output_shape = list(outputs[0].shape) if hasattr(outputs[0], 'shape') else []
                    
                    assert input_shape == output_shape, \
                        f"all_reduce {comm['node_id']} input shape {input_shape} should match output shape {output_shape}"

    def test_tp_report_metrics(self, training_graphs):
        """Verify report metrics change correctly with TP configuration."""
        model_key, graphs = training_graphs
        config = MODEL_CONFIGS[model_key]
        hidden_size = config["hidden_size"]
        
        fwd_graph = graphs["train_forward"]
        bwd_graph = graphs["train_backward"]

        # Get metrics for different TP degrees
        tp1_report, _ = run_tp_analysis(fwd_graph, bwd_graph, tp=1, hidden=hidden_size, num_layers=2)
        tp2_report, _ = run_tp_analysis(fwd_graph, bwd_graph, tp=2, hidden=hidden_size, num_layers=2)
        tp4_report, _ = run_tp_analysis(fwd_graph, bwd_graph, tp=4, hidden=hidden_size, num_layers=2)

        # Verify reports are valid
        assert tp1_report.step_time_ms > 0
        assert tp2_report.step_time_ms > 0
        assert tp4_report.step_time_ms > 0

        # Verify TP scaling trends - different TP degrees should have different step times
        assert tp1_report.step_time_ms != tp2_report.step_time_ms, \
            f"TP=1 and TP=2 should have different step times"
        assert tp2_report.step_time_ms != tp4_report.step_time_ms, \
            f"TP=2 and TP=4 should have different step times"

        # Verify MFU trends
        assert tp1_report.mfu > 0
        assert tp2_report.mfu > 0
        assert tp4_report.mfu > 0

    def test_tp_no_communication_for_tp1(self, training_graphs):
        """Verify TP=1 has no communication operators."""
        model_key, graphs = training_graphs
        config = MODEL_CONFIGS[model_key]
        hidden_size = config["hidden_size"]
        
        fwd_graph = graphs["train_forward"]
        bwd_graph = graphs["train_backward"]

        _, transformed = run_tp_analysis(
            fwd_graph, bwd_graph, tp=1,
            hidden=hidden_size, num_layers=2
        )
        unified_graph = transformed.get("unified")

        comm_nodes = [n for n in unified_graph.nodes.values() 
                      if getattr(n, 'op_type', '').startswith("comm.")]
        assert len(comm_nodes) == 0, \
            f"TP=1 should have no communication operators, got {len(comm_nodes)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])