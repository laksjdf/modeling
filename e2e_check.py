# -*- coding: utf-8 -*-
from python.zrt.graph import run_trace_phases, load_model
from python.zrt.transform import (
    build_default_pipeline, TransformContext,
    ParallelConfig, StreamConfig,
    export_training_graphs,
)
from python.zrt.executor import DAGScheduler
from python.zrt.simulator import SimulatorHub
from python.zrt.report import build_training_summary
from python.zrt.transform.context import TrainingConfig
import python.zrt.hardware.registry as hw_registry
from pathlib import Path

model_id = "deepseek-ai/DeepSeek-V3"

# Step 0: 准备输出目录
from python.zrt.graph.main import _make_model_slug
slug = _make_model_slug(model_id)
output_dir_base = Path("output") / "graph" / slug

# Step 1: 抓图（train_forward + train_backward）
result = run_trace_phases(
    model_id=model_id,
    num_layers=4,
    batch_size=1,
    seq_len=128,
    phases=("train_forward", "train_backward"),
)
raw_fwd, fused_fwd = result.graphs["train_forward"]
raw_bwd, fused_bwd = result.graphs["train_backward"]

_, config, _ = load_model(model_id, num_hidden_layers=4)
print(f"\n[1] 抓图完成:")
print(f"    train_forward raw={raw_fwd}  fused={fused_fwd}")
print(f"    train_backward raw={raw_bwd}  fused={fused_bwd}")

# Step 2: Transform（TP=1 baseline，前向 + 后向各独立变换）
print(f"\n[2] Transform pipeline (TP=1 baseline):")
hw = hw_registry.load("nvidia_h100_sxm")
ctx = TransformContext(
    hw_spec=hw,
    parallel=ParallelConfig(tp=1),
    stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
)
pipe = build_default_pipeline()

fwd_transformed = pipe.run(raw_fwd, ctx)
bwd_transformed = pipe.run(raw_bwd, ctx)
print(f"    fwd: {raw_fwd.num_nodes()} -> {fwd_transformed.num_nodes()} nodes")
print(f"    bwd: {raw_bwd.num_nodes()} -> {bwd_transformed.num_nodes()} nodes")

# Step 3: 调度（DAGScheduler 分别调度前向和后向）
print(f"\n[3] DAGScheduler:")
scheduler = DAGScheduler(hw_spec=hw)
fwd_tl = scheduler.schedule(fwd_transformed)
bwd_tl = scheduler.schedule(bwd_transformed)
print(f"    fwd: total={fwd_tl.total_latency_ms:.3f} ms"
      f"  compute={fwd_tl.compute_time_us/1e3:.3f} ms"
      f"  comm={fwd_tl.comm_time_us/1e3:.3f} ms")
print(f"    bwd: total={bwd_tl.total_latency_ms:.3f} ms"
      f"  compute={bwd_tl.compute_time_us/1e3:.3f} ms"
      f"  comm={bwd_tl.comm_time_us/1e3:.3f} ms")

# Step 4: 模拟（SimulatorHub 产出 SimResult）
print(f"\n[4] SimulatorHub (TP=1):")
hub = SimulatorHub.default()
fwd_sim = hub.simulate_graph(fwd_transformed, hw)
bwd_sim = hub.simulate_graph(bwd_transformed, hw)
print(f"    fwd: {len(fwd_sim)} ops simulated")
print(f"    bwd: {len(bwd_sim)} ops simulated")

# Step 5: 节点注解验证
print(f"\n[5] 节点注解验证 (fwd 前5节点):")
missing_annot = []
for node in list(fwd_transformed.topo_sort())[:5]:
    lat   = node.annotations.get("latency_us")
    flops = node.annotations.get("flops", "N/A")
    bound = node.annotations.get("bound", "N/A")
    sid   = node.annotations.get("stream_id", "N/A")
    print(f"    {node.op_type:35s}  lat={str(lat):>10}us  flops={str(flops):>12}"
          f"  bound={bound}  stream={sid}")
    if lat is None:
        missing_annot.append(node.op_type)
assert not missing_annot, f"缺少 latency_us 注解: {missing_annot}"
print(f"    OK 所有采样节点均含 latency_us / flops / bound / stream_id 注解")

# Step 6: 内存估算（TrainingMemoryPass）
print(f"\n[6] 训练内存估算 (ZeRO-1, TP=1):")
from python.zrt.transform.analysis.training import TrainingMemoryPass
ctx_train = TransformContext(
    hw_spec=hw,
    parallel=ParallelConfig(tp=1),
    stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
    training=TrainingConfig(optimizer="adam", zero_stage=1, micro_batch=1,
                            global_batch=1),
)
fwd_with_mem = TrainingMemoryPass().run(fwd_transformed, ctx_train)
mem_bd = fwd_with_mem.metadata.get("memory_breakdown")
if mem_bd:
    print(f"    weights     {mem_bd.weights/1e9:.2f} GB")
    print(f"    grads       {mem_bd.grads/1e9:.2f} GB")
    print(f"    opt_state   {mem_bd.opt_state/1e9:.2f} GB")
    print(f"    activations {mem_bd.activations/1e9:.2f} GB")
    print(f"    total       {mem_bd.total/1e9:.2f} GB")
else:
    print("    (memory_breakdown not available)")

# Step 7: TrainingSummary
print(f"\n[7] TrainingSummary (TP=1):")
train_summary = build_training_summary(
    model        = model_id,
    hardware     = "nvidia_h100_sxm",
    batch_size   = 1,
    seq_len      = 128,
    fwd_graph    = fwd_transformed,
    bwd_graph    = bwd_transformed,
    fwd_results  = fwd_sim,
    bwd_results  = bwd_sim,
    fwd_timeline = fwd_tl,
    bwd_timeline = bwd_tl,
    hw_spec      = hw,
    parallel_desc= "TP1",
    memory_breakdown = mem_bd,
)
summary_str = str(train_summary).replace("µ", "u")
print(summary_str)

# Step 8: TP=4 对比
print(f"\n[8] Transform pipeline (TP=4):")
ctx4 = TransformContext(
    hw_spec=hw,
    parallel=ParallelConfig(tp=4),
    stream_config=StreamConfig(num_compute_streams=1, num_comm_streams=1),
)
fwd_tp4 = pipe.run(raw_fwd, ctx4)
bwd_tp4 = pipe.run(raw_bwd, ctx4)
fwd_tl4 = scheduler.schedule(fwd_tp4)
bwd_tl4 = scheduler.schedule(bwd_tp4)
fwd_sim4 = hub.simulate_graph(fwd_tp4, hw)
bwd_sim4 = hub.simulate_graph(bwd_tp4, hw)

train_summary_tp4 = build_training_summary(
    model        = model_id,
    hardware     = "nvidia_h100_sxm",
    batch_size   = 1,
    seq_len      = 128,
    fwd_graph    = fwd_tp4,
    bwd_graph    = bwd_tp4,
    fwd_results  = fwd_sim4,
    bwd_results  = bwd_sim4,
    fwd_timeline = fwd_tl4,
    bwd_timeline = bwd_tl4,
    hw_spec      = hw,
    parallel_desc= "TP4",
)
print(f"    TP=4 step={train_summary_tp4.step_ms:.3f} ms"
      f"  fwd={train_summary_tp4.forward_ms:.3f} ms"
      f"  bwd={train_summary_tp4.backward_ms:.3f} ms"
      f"  MFU={train_summary_tp4.mfu:.2%}")

# Step 9: 正确性断言
print(f"\n[9] 正确性断言:")
assert fwd_tl.comm_time_us == 0.0,  f"TP=1 fwd 不应有通信时间"
assert bwd_tl.comm_time_us == 0.0,  f"TP=1 bwd 不应有通信时间"
assert len(fwd_tp4.comm_nodes()) > 0, f"TP=4 fwd 应有 comm 节点"
assert len(bwd_tp4.comm_nodes()) > 0, f"TP=4 bwd 应有 comm 节点"
assert train_summary.step_ms > 0,          "step latency 应为正"
assert train_summary.mfu >= 0.0,           "MFU 应为非负"
assert train_summary.forward_ms > 0,       "forward latency 应为正"
assert train_summary.backward_ms > 0,      "backward latency 应为正"
assert abs(train_summary.step_ms - train_summary.forward_ms - train_summary.backward_ms) < 0.01
assert raw_fwd.num_nodes() > fused_fwd.num_nodes(), \
    f"fused {fused_fwd.num_nodes()} 应小于 raw {raw_fwd.num_nodes()}"
assert raw_bwd.num_nodes() > fused_bwd.num_nodes(), \
    f"fused bwd {fused_bwd.num_nodes()} 应小于 raw bwd {raw_bwd.num_nodes()}"
print(f"    OK TP=1: fwd_comm=0us, bwd_comm=0us")
print(f"    OK TP=4: fwd_comm_nodes={len(fwd_tp4.comm_nodes())} bwd_comm_nodes={len(bwd_tp4.comm_nodes())}")
print(f"    OK step={train_summary.step_ms:.3f} ms = fwd {train_summary.forward_ms:.3f} + bwd {train_summary.backward_ms:.3f}")
print(f"    OK MFU={train_summary.mfu:.4f}")
print(f"    OK fusion: fwd {raw_fwd.num_nodes()} -> {fused_fwd.num_nodes()}"
      f"  bwd {raw_bwd.num_nodes()} -> {fused_bwd.num_nodes()}")

# Step 10: 导出训练图（Excel + JSON）
print(f"\n[10] 导出训练图 (TP=1):")
export_result = export_training_graphs(
    fwd_graph        = fwd_transformed,
    bwd_graph        = bwd_transformed,
    ctx              = ctx,
    output_dir       = output_dir_base,
    training_summary = train_summary,
)
print(f"    Excel:    {export_result['excel']}")
print(f"    JSON fwd: {export_result['json_fwd']}")
print(f"    JSON bwd: {export_result['json_bwd']}")
assert export_result["excel"].exists(),    "训练 Excel 导出失败"
assert export_result["json_fwd"].exists(), "fwd JSON 导出失败"
assert export_result["json_bwd"].exists(), "bwd JSON 导出失败"
print(f"    OK 导出文件完整性检查通过")
print(f"    OK Excel sheets: Metadata / Transformed Operators / Communication Ops /"
      f" Parallelism Summary / Stream Assignment / Backward Operators /"
      f" Recompute Ops / Training Summary")
