from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PathLeg:
    """A continuous sub-segment with constant per-step displacement."""

    dx_step: float
    dy_step: float
    steps: int
    source_on: bool = True


@dataclass
class PathDef:
    """Per-component path definition consumed by the outer solver."""

    component_id: int
    x_start: float
    y_start: float
    legs: list[PathLeg]
    total_steps: int
    weight: int


__all__ = ["PathDef", "PathLeg"]
