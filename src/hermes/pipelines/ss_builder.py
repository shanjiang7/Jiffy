"""
The common core (config load → waypoints → trajectory → fine segments → SS
grouping) is factored here so that both the DAG pipeline and the simulation
pipeline can reuse it without duplication.

Public API
----------
build_ss_from_cfg :returns grouped Segment lists + config metadata
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from hermes.laser_path.path_loader import build_path_sections_nd_from_ini
from hermes.laser_path.path_segment import (
    sample_traj_with_sections,
    compute_dt_phi_with_source,
    build_segments_from_program,
)
from hermes.pipelines.config import PipelineConfig
from hermes.utils.segment_types import Segment, SuperSegment, build_supersegments


@dataclass(frozen=True)
class SSBuildResult:
    segments: list[Segment]
    supersegments: list[SuperSegment]
    len_scale: float
    num_layers: int
    segments_per_layer: int
    ss_per_layer: int


def _replicate_segments(
    base_segments: List[Segment],
    layer_idx: int,
    n_segs_per_layer: int,
    time_offset: float,
) -> List[Segment]:
    """Create a copy of base segments with offset IDs and start times.

    Times are accumulated sequentially (rather than adding a flat offset to
    each stored t_start) so that seg[i].t_start == seg[i-1].time_window[1]
    exactly, avoiding IEEE 754 non-associativity violations.
    """
    result: List[Segment] = []
    t = base_segments[0].t_start_s + time_offset
    for seg in base_segments:
        new_seg = Segment(
            id=layer_idx * n_segs_per_layer + seg.id,
            steps=seg.steps,
            power_W=seg.power_W,
            V_mps=seg.V_mps,
            t_start_s=t,
            width_m=seg.width_m,
        )
        result.append(new_seg)
        t = new_seg.time_window_s[1]
    return result


def build_ss_from_cfg(
    cfg: PipelineConfig,
) -> SSBuildResult:
    step_nd = cfg.require_effective_motion_step_nd()

    sections = build_path_sections_nd_from_ini(cfg.cfg_path, len_scale=cfg.len_scale, step_nd=step_nd)
    traj_nd, move_source_on = sample_traj_with_sections(sections, ds=step_nd)
    laser_path_nd = compute_dt_phi_with_source(traj_nd, move_source_on, len_scale=cfg.len_scale)

    seg_program = laser_path_nd[:cfg.first_n_points]

    base_segments = build_segments_from_program(
        seg_program,
        steps_per_segment=cfg.steps_per_segment,
        power_W=cfg.segment.P_W,
        V_mps=cfg.segment.V_mps,
        t0_s=cfg.segment.t0_s,
        width_roi_m=cfg.width_roi_m,
    )
    
    base_ss = build_supersegments(base_segments, segments_per_supersegment=cfg.segments_per_supersegment)
    ss_per_layer = len(base_ss)
    n_segs = len(base_segments)
    layer_duration = base_segments[-1].time_window_s[1] if base_segments else 0.0

    num_layers = cfg.layers.num_layers
    all_segments: list[Segment] = list(base_segments)
    all_ss: list[SuperSegment] = list(base_ss)

    for layer_idx in range(1, num_layers):
        layer_segs = _replicate_segments(base_segments, layer_idx, n_segs, layer_idx * layer_duration)
        all_segments.extend(layer_segs)
        layer_ss = build_supersegments(
            layer_segs,
            segments_per_supersegment=cfg.segments_per_supersegment,
            start_supersegment_id=layer_idx * ss_per_layer,
        )
        all_ss.extend(layer_ss)

    return SSBuildResult(
        segments=all_segments,
        supersegments=all_ss,
        len_scale=cfg.len_scale,
        num_layers=num_layers,
        segments_per_layer=n_segs,
        ss_per_layer=ss_per_layer,
    )


__all__ = ["build_ss_from_cfg", "SSBuildResult"]
