"""Tests for graph_mode=True (torch.compile graph capture path).

Uses DeepSeek-V3 local config — no network required.
"""
import pytest
from python.zrt.pipeline import run_trace_phases

MODEL_ID = "deepseek-ai/DeepSeek-V3"
_COMMON = dict(
    model_id=MODEL_ID,
    num_layers=2,
    batch_size=1,
    seq_len=16,
    graph_mode=True,
    auto_layers=False,
)


def test_dsv3_graph_mode_train_backward_records():
    """graph_mode train_backward 应捕获前向 + 梯度算子。"""
    result = run_trace_phases(**_COMMON, phases=("train_backward",))
    records = result.phase_records["train_backward"]
    assert len(records) > 0, "Expected compile-mode to capture at least one op"


@pytest.mark.xfail(
    strict=False,
    reason="pre-existing: graph_mode 抓图成功（records 测试通过）但 Excel/JSON 未输出。tracked in #66",
)
def test_dsv3_graph_mode_train_backward_files(tmp_path):
    """graph_mode 应输出 Excel 和 JSON 文件。"""
    run_trace_phases(**_COMMON, phases=("train_backward",), output_dir=tmp_path)
    assert list(tmp_path.glob("*train_backward_ops.xlsx")), "Excel file not created"
    assert list(tmp_path.glob("*train_backward_raw_graph.json")), "JSON file not created"
