from enum import Enum


class Dtype(Enum):
    """Element dtype for parameters/grads/activations.

    Note: ``.bytes`` and ``.stored_bytes`` return ``float`` (FP4 = 0.5).
    Callers that multiply by element counts and need an int byte total
    must round explicitly.
    """
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    FP4 = "fp4"

    @property
    def bytes(self) -> float:
        return _BYTES[self]

    @property
    def block_overhead_bytes_per_elem(self) -> float:
        # FP4 uses MXFP-style block=32 with one BF16 (2B) scale per block.
        return 2.0 / 32.0 if self is Dtype.FP4 else 0.0

    @property
    def stored_bytes(self) -> float:
        return self.bytes + self.block_overhead_bytes_per_elem


_BYTES: dict[Dtype, float] = {
    Dtype.FP32: 4.0,
    Dtype.BF16: 2.0,
    Dtype.FP16: 2.0,
    Dtype.FP8_E4M3: 1.0,
    Dtype.FP8_E5M2: 1.0,
    Dtype.FP4: 0.5,
}

# Back-compat alias: callers that use ``Dtype.FP8`` get E4M3 (V4 forward GEMM).
Dtype.FP8 = Dtype.FP8_E4M3  # type: ignore[attr-defined]
