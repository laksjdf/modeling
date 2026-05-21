"""Tilesim backend — placeholder (not used in current TilesimLatencyPass design).

Tilesim latency prediction is now handled directly by
``TilesimLatencyPass`` (tilesim_pass.py), which calls
``op_latency_predict`` in-process.  This empty shell exists only
because ``backend_register.py`` references it.
"""
from python.zrt.hardware import HardwareSpec
from python.zrt.ir import OpNode
from python.zrt.simulator import OpSimulator, SimResult

class TilesimSimulator(OpSimulator):
    name = "tilesim"
    priority = 2

    def can_simulate(self, node: "OpNode", hw: "HardwareSpec") -> bool:
        return False

    def simulate(self, node: "OpNode", hw: "HardwareSpec") -> SimResult:
        pass