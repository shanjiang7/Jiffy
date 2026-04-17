"""
Core segment and supersegment types for the Hermes pipeline.

Classes
-------
Step            – single timestep position/angle record
Segment         – ordered sequence of Steps; carries power, velocity, timing
SuperSegment    – fixed-size group of consecutive Segments (inter-node unit)

Functions
---------
build_supersegments  – group a Segment list into fixed-size SuperSegments
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ── Step ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Step:
    x_nd: float
    y_nd: float
    dt_m: float
    phi_rad: float


# ── Segment ───────────────────────────────────────────────────────────────────

class Segment:
    """
    Ordered sequence of Steps representing one laser segment.

    Uses __slots__ for fast attribute access and low memory footprint.
    All fields are set once in __init__ and treated as read-only.
    """

    __slots__ = (
        "id", "steps", "power_W", "V_mps", "t_start_s", "width_m",
        "_duration_s", "_path_bounds_nd", "_aabb_nd",
    )

    def __init__(
        self,
        id: int,
        steps: tuple[Step, ...],
        power_W: float,
        V_mps: float,
        t_start_s: float = 0.0,
        width_m: float = 0.0,
    ) -> None:
        V = float(V_mps)
        if V <= 0.0:
            raise ValueError(f"V_mps must be > 0; got {V!r}")

        self.id = int(id)
        self.steps = steps
        self.power_W = float(power_W)
        self.V_mps = V
        self.t_start_s = float(t_start_s)
        self.width_m = float(width_m)

        # Cache expensive computations once at construction
        self._duration_s: float = float(sum(float(s.dt_m) for s in steps)) / V
        self._path_bounds_nd: tuple[float, float, float, float] = self._compute_path_bounds()
        self._aabb_nd: tuple[float, float, float, float] = self._compute_aabb_nd()

    # ── cached properties ─────────────────────────────────────────────────────

    @property
    def duration_s(self) -> float:
        return self._duration_s

    @property
    def path_bounds_nd(self) -> tuple[float, float, float, float]:
        """Line-based bounding box: (x_min, x_max, y_min, y_max) in nd units."""
        return self._path_bounds_nd

    @property
    def aabb_nd(self) -> tuple[float, float, float, float]:
        """ROI-based AABB: (x_min, x_max, y_min, y_max) in nd units."""
        return self._aabb_nd

    # ── other properties ──────────────────────────────────────────────────────

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def n_moves(self) -> int:
        """Number of dt-moves (= points - 1)."""
        return max(0, len(self.steps) - 1)

    @property
    def time_window_s(self) -> tuple[float, float]:
        t0 = self.t_start_s
        return (t0, t0 + self._duration_s)

    # ── equality / hashing (replicate dataclass frozen behaviour) ─────────────

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Segment):
            return NotImplemented
        return (self.id == other.id and self.steps == other.steps
                and self.power_W == other.power_W and self.V_mps == other.V_mps
                and self.t_start_s == other.t_start_s and self.width_m == other.width_m)

    def __hash__(self) -> int:
        return hash((self.id, self.steps, self.power_W, self.V_mps,
                     self.t_start_s, self.width_m))

    def __repr__(self) -> str:
        return (f"Segment(id={self.id}, n_steps={len(self.steps)}, "
                f"power_W={self.power_W}, V_mps={self.V_mps})")

    # ── private helpers ───────────────────────────────────────────────────────

    def _compute_path_bounds(self) -> tuple[float, float, float, float]:
        if not self.steps:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [float(s.x_nd) for s in self.steps]
        ys = [float(s.y_nd) for s in self.steps]
        return (min(xs), max(xs), min(ys), max(ys))

    def _compute_aabb_nd(self) -> tuple[float, float, float, float]:
        if len(self.steps) < 2:
            return self._path_bounds_nd

        width_m = self.width_m
        x_all, y_all = [], []

        for i in range(len(self.steps) - 1):
            s0, s1 = self.steps[i], self.steps[i + 1]
            dt_m = float(s0.dt_m)
            if dt_m <= 0.0:
                continue
            phi = float(s0.phi_rad)
            cx = (float(s0.x_nd) + float(s1.x_nd)) * 0.5
            cy = (float(s0.y_nd) + float(s1.y_nd)) * 0.5
            hl, hw = dt_m * 0.5, width_m * 0.5
            c, s = math.cos(phi), math.sin(phi)
            lx, ly = hl * c, hl * s   # half-length vector
            wx, wy = -hw * s, hw * c  # half-width vector (perpendicular)
            # 4 corners of the oriented rectangle
            for sl, sw in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                x_all.append(cx + sl * lx + sw * wx)
                y_all.append(cy + sl * ly + sw * wy)

        if not x_all:
            return self._path_bounds_nd
        return (min(x_all), max(x_all), min(y_all), max(y_all))


# ── SuperSegment ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class SuperSegment:
    """Fixed-size group of consecutive Segments (inter-node unit)."""
    id: int
    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        assert len(self.segments) > 0, "SuperSegment must contain at least one segment"
        seg_ids = [s.id for s in self.segments]
        for i in range(1, len(seg_ids)):
            assert seg_ids[i] == seg_ids[0] + i, "Segments must have sequential ids"
            assert (self.segments[i].time_window_s[0]
                    >= self.segments[i - 1].time_window_s[1]), \
                "Segments must have non-decreasing time windows"

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def segment_ids(self) -> tuple[int, ...]:
        return tuple(s.id for s in self.segments)

    @property
    def segment_id_range(self) -> tuple[int, int]:
        return (self.segments[0].id, self.segments[-1].id)

    @property
    def time_window_s(self) -> tuple[float, float]:
        return (self.segments[0].time_window_s[0],
                self.segments[-1].time_window_s[1])

    @property
    def duration_s(self) -> float:
        t0, t1 = self.time_window_s
        return t1 - t0

    @property
    def t_start_s(self) -> float:
        return self.segments[0].t_start_s

    @property
    def V_mps(self) -> float:
        return self.segments[0].V_mps

    @property
    def power_W(self) -> float:
        return self.segments[0].power_W

    @property
    def aabb_nd(self) -> tuple[float, float, float, float]:
        xmins, xmaxs, ymins, ymaxs = zip(*(seg.aabb_nd for seg in self.segments))
        return (min(xmins), max(xmaxs), min(ymins), max(ymaxs))


# ── Factory ───────────────────────────────────────────────────────────────────

def build_supersegments(
    segments: list[Segment],
    *,
    segments_per_supersegment: int,
    start_supersegment_id: int = 0,
) -> list[SuperSegment]:
    """Group a sequential list of Segments into fixed-size SuperSegments."""
    k = int(segments_per_supersegment)
    assert k >= 1, "segments_per_supersegment must be >= 1"
    assert segments, "No segments to build SuperSegments from"
    ids = [s.id for s in segments]
    for i in range(1, len(ids)):
        assert ids[i] == ids[0] + i, "Segments must be sequential by id"

    out, super_id = [], int(start_supersegment_id)
    for i in range(0, len(segments), k):
        out.append(SuperSegment(id=super_id, segments=tuple(segments[i:i + k])))
        super_id += 1
    return out


__all__ = ["Step", "Segment", "SuperSegment", "build_supersegments"]
