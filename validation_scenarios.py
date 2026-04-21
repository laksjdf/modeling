"""
标准验证场景定义及参数提取
用于 e2e_validate_with_public_data.py
"""
from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class HardwareConfig:
    """硬件配置"""
    device_name: str  # "A100", "B200", "Ascend 910C"
    num_devices: int
    interconnect: str  # "NVLink", "HCCS", "RoCE"
    interconnect_bandwidth_gbs: float
    hbm_bandwidth_gbs: float  # 单卡 HBM 带宽
    total_memory_gb: int
    tensor_parallel_size: int


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    num_params_b: int
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    quantization: str  # "FP16", "FP8", "NVFP4", "W8A8"
    is_moe: bool = False
    num_experts: Optional[int] = None


@dataclass
class InferenceConfig:
    """推理配置"""
    batch_size: int
    input_seq_len: int
    output_seq_len: int
    phase: str = "prefill"  # "prefill" | "decode" | "both"


@dataclass
class ValidationScenario:
    """完整验证场景"""
    scenario_id: str
    hardware: HardwareConfig
    model: ModelConfig
    inference: InferenceConfig
    measured_latency_ms: Optional[float] = None
    measured_throughput_tok_s: Optional[float] = None
    framework: Optional[str] = None
    source: Optional[str] = None

    def to_dict(self):
        return asdict(self)


# ============================================================================
# 标准验证场景定义
# ============================================================================

SCENARIO_1_A100_LLAMA2_70B_TP4 = ValidationScenario(
    scenario_id="A100_Llama2_70B_TP4",
    hardware=HardwareConfig(
        device_name="A100 40GB",
        num_devices=4,
        interconnect="NVLink",
        interconnect_bandwidth_gbs=900,
        hbm_bandwidth_gbs=1555,
        total_memory_gb=160,
        tensor_parallel_size=4,
    ),
    model=ModelConfig(
        name="Llama-2 70B",
        num_params_b=70,
        num_hidden_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_kv_heads=8,
        quantization="FP16",
    ),
    inference=InferenceConfig(
        batch_size=1,
        input_seq_len=1500,
        output_seq_len=100,
        phase="prefill+decode",
    ),
    measured_latency_ms=7400.0,
    measured_throughput_tok_s=3245.0,  # vLLM
    framework="vLLM",
    source="vLLM Paper + TrueFoundry benchmark",
)

SCENARIO_2_A100_QWEN_72B_TP4 = ValidationScenario(
    scenario_id="A100_Qwen_72B_TP4",
    hardware=HardwareConfig(
        device_name="A100 40GB",
        num_devices=4,
        interconnect="NVLink",
        interconnect_bandwidth_gbs=900,
        hbm_bandwidth_gbs=1555,
        total_memory_gb=160,
        tensor_parallel_size=4,
    ),
    model=ModelConfig(
        name="Qwen 72B",
        num_params_b=72,
        num_hidden_layers=80,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=32,
        quantization="FP16",
    ),
    inference=InferenceConfig(
        batch_size=1,
        input_seq_len=512,
        output_seq_len=100,
        phase="decode",
    ),
    measured_latency_ms=None,  # TPOT 不直接转换为延迟
    measured_throughput_tok_s=449.51,
    framework="vLLM",
    source="DatabaseMart vLLM benchmark",
)

SCENARIO_3_ASCEND910B_16CARD_DEEPSEEK = ValidationScenario(
    scenario_id="Ascend910B_16Card_DeepSeek",
    hardware=HardwareConfig(
        device_name="Ascend 910B",
        num_devices=16,
        interconnect="HCCS",
        interconnect_bandwidth_gbs=800,  # HCCS 估计带宽
        hbm_bandwidth_gbs=1500,
        total_memory_gb=1024,
        tensor_parallel_size=8,  # 2×8卡，假设 TP=8
    ),
    model=ModelConfig(
        name="DeepSeek-R1 蒸馏系列",
        num_params_b=14,  # 蒸馏版本
        num_hidden_layers=32,
        hidden_size=2048,
        num_attention_heads=32,
        num_kv_heads=4,
        quantization="W8A8",
        is_moe=True,
        num_experts=16,
    ),
    inference=InferenceConfig(
        batch_size=100,  # 100 并发请求
        input_seq_len=512,
        output_seq_len=100,
        phase="decode",
    ),
    measured_latency_ms=None,
    measured_throughput_tok_s=618.0,
    framework="CloudMatrix-Infer",
    source="SegmentFault + CSDN Ascend 910B 部署指南",
)

SCENARIO_4_B200_8CARD_LLAMA4 = ValidationScenario(
    scenario_id="B200_8Card_Llama4",
    hardware=HardwareConfig(
        device_name="B200 Blackwell",
        num_devices=8,
        interconnect="NVLink 5",
        interconnect_bandwidth_gbs=1800,
        hbm_bandwidth_gbs=2400,
        total_memory_gb=1440,
        tensor_parallel_size=8,
    ),
    model=ModelConfig(
        name="Llama 4 Maverick",
        num_params_b=400,
        num_hidden_layers=128,
        hidden_size=16384,
        num_attention_heads=128,
        num_kv_heads=8,
        quantization="FP8",
    ),
    inference=InferenceConfig(
        batch_size=1,
        input_seq_len=1024,
        output_seq_len=256,
        phase="prefill+decode",
    ),
    measured_latency_ms=None,
    measured_throughput_tok_s=1000.0,  # per user, sustained
    framework="TensorRT-LLM",
    source="NVIDIA Developer Blog - Blackwell",
)

# ============================================================================
# 场景集合
# ============================================================================

VALIDATION_SCENARIOS = [
    SCENARIO_1_A100_LLAMA2_70B_TP4,
    SCENARIO_2_A100_QWEN_72B_TP4,
    SCENARIO_3_ASCEND910B_16CARD_DEEPSEEK,
    SCENARIO_4_B200_8CARD_LLAMA4,
]


def export_scenarios_to_json(output_path: str = "validation_scenarios.json"):
    """导出验证场景为 JSON"""
    data = {
        "metadata": {
            "description": "标准验证场景 - 用于多卡推理模型准确度评估",
            "num_scenarios": len(VALIDATION_SCENARIOS),
            "updated": "2026-04-20",
        },
        "scenarios": [asdict(s) for s in VALIDATION_SCENARIOS],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Validation scenarios exported: {output_path}")


if __name__ == "__main__":
    export_scenarios_to_json()

    # Print scenario overview
    print("\n" + "=" * 80)
    print("Validation Scenarios Overview")
    print("=" * 80)
    for scenario in VALIDATION_SCENARIOS:
        print(f"\n[{scenario.scenario_id}]")
        print(f"  Hardware: {scenario.hardware.num_devices}x {scenario.hardware.device_name} ({scenario.hardware.interconnect})")
        print(f"  Model: {scenario.model.name} ({scenario.model.num_params_b}B)")
        print(f"  Config: TP={scenario.hardware.tensor_parallel_size}, batch={scenario.inference.batch_size}")
        if scenario.measured_throughput_tok_s:
            print(f"  Measured: {scenario.measured_throughput_tok_s:.1f} tok/s")
        if scenario.measured_latency_ms:
            print(f"  Measured: {scenario.measured_latency_ms:.1f} ms")
        print(f"  Source: {scenario.source}")
