#!/usr/bin/env python3
# 测试所有算子的功能

import sys
import os

# 添加python目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

from zrt.layers import *
from zrt.tensor_base import TensorBase
from zrt.input_param import InputParam
from zrt.layers.op_base import OP_CLASS_REGISTRY

class DummyModel:
    pass

dummy_model = DummyModel()
input_param = InputParam(batch_size=64, seq_len=2048)

# 测试所有注册的算子
def test_all_operators():
    print("开始测试所有算子...")
    print(f"总共有 {len(OP_CLASS_REGISTRY)} 个算子")
    
    failures = []
    
    for op_name, op_class in OP_CLASS_REGISTRY.items():
        try:
            # 测试创建算子实例
            op = op_class(dummy_model, op_name)
            print(f"[OK] {op_name}: 创建成功")
            
            # 测试__call__方法
            # 根据算子类型创建不同的输入
            if op_name in ["Add", "AddInplace", "Mul", "MulInplace"]:
                # 二元算子，需要两个输入
                input_tensor = [
                    TensorBase([32, 1024, 768]),
                    TensorBase([32, 1024, 768])
                ]
            elif op_name in ["ScaledDotProductAttention"]:
                # 注意力算子，需要三个输入 (query, key, value)
                input_tensor = [
                    TensorBase([32, 1024, 768]),
                    TensorBase([32, 1024, 768]),
                    TensorBase([32, 1024, 768])
                ]
            elif op_name in ["Embedding"]:
                # Embedding算子，需要一个输入
                input_tensor = [TensorBase([32, 1024])]
            else:
                # 其他算子，默认一个输入
                input_tensor = [TensorBase([32, 1024, 768])]
            
            output = op(input_tensor)
            print(f"[OK] {op_name}: 调用成功")
            
            # 测试build_dynamic_input方法
            dynamic_input = op_class.build_dynamic_input(input_tensor, input_param)
            print(f"[OK] {op_name}: build_dynamic_input成功")
            
            # 测试属性
            assert hasattr(op, 'name'), f"{op_name} 缺少name属性"
            assert hasattr(op, 'inputs'), f"{op_name} 缺少inputs属性"
            assert hasattr(op, 'outputs'), f"{op_name} 缺少outputs属性"
            assert hasattr(op, 'compute_formula'), f"{op_name} 缺少compute_formula属性"
            assert hasattr(op, 'compute_flops'), f"{op_name} 缺少compute_flops属性"
            assert hasattr(op, 'memory_bytes'), f"{op_name} 缺少memory_bytes属性"
            print(f"[OK] {op_name}: 属性检查成功")
            
        except Exception as e:
            print(f"[FAIL] {op_name}: 测试失败 - {e}")
            failures.append(f"{op_name}: {e}")
    
    print(f"\n测试完成: 成功 {len(OP_CLASS_REGISTRY) - len(failures)}, 失败 {len(failures)}")
    assert not failures, "\n".join(failures)

if __name__ == "__main__":
    test_all_operators()
