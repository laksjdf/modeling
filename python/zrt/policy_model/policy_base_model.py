from abc import ABC, abstractmethod
from typing import List
from zrt.runtime_config import RuntimeConfig
from zrt.layers.op_base import OperatorBase
from zrt.tensor_base import TensorBase


class PolicyBaseModel(ABC):
    def __init__(self, rt_config: RuntimeConfig):
        self.rt_config = rt_config

    @abstractmethod
    def predict(self, op: OperatorBase, input_tensor: List[TensorBase], **kwargs) -> float:
        pass
