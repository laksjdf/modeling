"""Compute-Communication overlap analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.executor.scheduler import Timeline


@dataclass
class OverlapReport:
    """Detailed compute-communication overlap analysis."""
    compute_us: float      # Total compute latency (sum of all compute ops)
    comm_us: float         # Total communication latency (sum of all comm ops)
    overlap_us: float      # Communication time hidden behind compute
    exposed_comm_us: float  # Communication time NOT hidden (critical for latency)
    overlap_ratio: float   # overlap_us / comm_us (0.0 to 1.0)
    critical_path_us: float  # Total wall-clock latency


class OverlapAnalyzer:
    """Analyze compute-communication overlap using interval intersection.

    The key insight: overlap = intersection of compute and comm time intervals.
    This is more accurate than the approximation (compute + comm - total).
    """

    def analyze(self, timeline: "Timeline") -> OverlapReport:
        """Analyze overlap in a timeline.

        Uses a sweep-line algorithm to compute the exact intersection of compute
        and communication intervals.
        """
        compute = self._intervals(timeline, "compute")
        comm = self._intervals(timeline, "comm")

        total_compute = self._sum_duration(compute)
        total_comm = self._sum_duration(comm)
        overlap = self._intersection(compute, comm)

        critical_path = timeline.total_latency_us
        exposed = total_comm - overlap

        overlap_ratio = 0.0
        if total_comm > 0:
            overlap_ratio = min(1.0, overlap / total_comm)

        return OverlapReport(
            compute_us=total_compute,
            comm_us=total_comm,
            overlap_us=overlap,
            exposed_comm_us=exposed,
            overlap_ratio=overlap_ratio,
            critical_path_us=critical_path,
        )

    def _intervals(self, timeline: "Timeline",
                   stream_type: str) -> list[tuple[float, float]]:
        """Extract time intervals for ops of a given type.

        Returns list of (start_us, end_us) tuples, not necessarily sorted.
        """
        return [
            (op.start_us, op.end_us)
            for op in timeline.scheduled_ops
            if op.stream_type == stream_type
        ]

    def _sum_duration(self, intervals: list[tuple[float, float]]) -> float:
        """Sum the duration of all intervals (they may or may not overlap)."""
        return sum(end - start for start, end in intervals)

    def _intersection(self, a: list[tuple[float, float]],
                      b: list[tuple[float, float]]) -> float:
        """Compute total intersection time of two sets of intervals.

        Uses sweep-line algorithm:
        - Create events for all interval boundaries
        - Sort by time
        - Sweep through, tracking how many intervals from each set are active
        - When both sets are active, add to overlap
        """
        if not a or not b:
            return 0.0

        events = []
        for s, e in a:
            events.append((s, 1, "a"))     # interval starts
            events.append((e, -1, "a"))    # interval ends
        for s, e in b:
            events.append((s, 1, "b"))
            events.append((e, -1, "b"))

        events.sort()

        overlap = 0.0
        active_a = 0
        active_b = 0
        prev_t = 0.0

        for t, delta, src in events:
            # If both sets are active, this interval contributes to overlap
            if active_a > 0 and active_b > 0:
                overlap += t - prev_t
            # Update active counts
            if src == "a":
                active_a += delta
            else:
                active_b += delta
            prev_t = t

        return max(0.0, overlap)  # Clamp to 0 in case of numerical error
