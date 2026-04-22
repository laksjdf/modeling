from enum import Enum


class Dtype(Enum):
    FP32 = 4
    BF16 = 2
    FP16 = 2
    FP8 = 1

    @property
    def bytes(self) -> int:
        return self.value
