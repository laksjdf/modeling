"""Stream abstraction for executor."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stream:
    """Represents a single execution stream (compute or communication).

    A stream is a sequential execution context on a device. Operations on the same
    stream cannot execute in parallel; operations on different streams can.
    """
    stream_id: int
    stream_type: str  # "compute" | "comm"
