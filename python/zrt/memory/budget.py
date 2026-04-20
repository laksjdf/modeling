"""Memory budget breakdown dataclass."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryBudget:
    """Per-device memory budget estimate."""

    weights_mb: float
    kv_cache_mb: float
    activation_peak_mb: float
    comm_buffer_mb: float
    framework_overhead_mb: float
    total_mb: float
    capacity_mb: float
    is_feasible: bool

    def breakdown(self) -> dict[str, float]:
        """Return the absolute MB contribution of each memory bucket."""
        return {
            "weights_mb": self.weights_mb,
            "kv_cache_mb": self.kv_cache_mb,
            "activation_peak_mb": self.activation_peak_mb,
            "comm_buffer_mb": self.comm_buffer_mb,
            "framework_overhead_mb": self.framework_overhead_mb,
            "total_mb": self.total_mb,
            "capacity_mb": self.capacity_mb,
        }

    @property
    def utilization(self) -> float:
        if self.capacity_mb <= 0:
            return 0.0
        return self.total_mb / self.capacity_mb
