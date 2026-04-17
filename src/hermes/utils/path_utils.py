from __future__ import annotations

import math
from argparse import Namespace
from pathlib import Path
from typing import List

from hermes.motion.types import PathDef, PathLeg
from hermes.utils.dag_utils import Component
from hermes.utils.segment_types import SuperSegment

def resolve_path(project_root: Path, arg_path: str | None, default_rel: str) -> Path:
    if arg_path is None:
        return (project_root / default_rel).resolve()
    p = Path(arg_path).expanduser()
    if p.is_absolute():
        return p.resolve()
    if p.exists():
        return p.resolve()
    p2 = project_root / p
    if p2.exists():
        return p2.resolve()
    return (project_root / "configs" / p).resolve()


def build_path_defs_from_components(
    supersegments: List[SuperSegment],
    components: List[Component],
    dt_s: float,
    len_scale: float,
    coord_scale: float = 1.0,
) -> List[PathDef]:
    """
    Build one PathDef per component.

    Stitches all steps contained in the SuperSegments of a Component into
    continuous `PathLeg`s, merging identical movements across Segment and
    SuperSegment boundaries.

    Parameters
    ----------
    len_scale : float
        The HERMES solver's nondimensionalization length scale (phys.len_scale,
        in metres). Kept for API compatibility; geometric leg displacements are
        reconstructed from the path step coordinates rather than solver dt.
    coord_scale : float, optional
        Multiplicative factor applied to raw x_nd / y_nd waypoint coordinates
        to convert them from the DAG pipeline's ND system into the HERMES
        solver's ND system.
        coord_scale = dag_len_scale / hermes_len_scale  (default 1.0 = no conversion)
    """
    _ = dt_s, len_scale
    phi_tol = 1e-6
    path_defs: List[PathDef] = []
    
    for comp in components:
        comp_ss_list = supersegments[comp.start_ss : comp.end_ss + 1]
        
        first_seg = comp_ss_list[0].segments[0]
        x_start = float(first_seg.steps[0].x_nd) * coord_scale
        y_start = float(first_seg.steps[0].y_nd) * coord_scale
        
        legs: List[PathLeg] = []
        total_steps = 0
        
        cur_dx = None
        cur_dy = None
        cur_src = None
        cur_count = 0
        
        for ss in comp_ss_list:
            for seg in ss.segments:
                src = float(seg.power_W) > 0.0
                if len(seg.steps) <= 1:
                    continue

                for step_a, step_b in zip(seg.steps[:-1], seg.steps[1:]):
                    dx = (float(step_b.x_nd) - float(step_a.x_nd)) * coord_scale
                    dy = (float(step_b.y_nd) - float(step_a.y_nd)) * coord_scale
                    
                    if cur_dx is None:
                        cur_dx = dx
                        cur_dy = dy
                        cur_src = src
                        cur_count = 1
                    else:
                        if abs(dx - cur_dx) < phi_tol and abs(dy - cur_dy) < phi_tol and src == cur_src:
                            cur_count += 1
                        else:
                            legs.append(PathLeg(
                                dx_step=cur_dx,
                                dy_step=cur_dy,
                                steps=cur_count,
                                source_on=cur_src,
                            ))
                            total_steps += cur_count
                            cur_dx = dx
                            cur_dy = dy
                            cur_src = src
                            cur_count = 1

        # Flush the final leg if it exists
        if cur_count > 0:
            legs.append(PathLeg(
                dx_step=cur_dx,
                dy_step=cur_dy,
                steps=cur_count,
                source_on=cur_src,
            ))
            total_steps += cur_count
                    
        weight = comp.end_ss - comp.start_ss + 1
        path_defs.append(PathDef(
            component_id=comp.id,
            x_start=x_start,
            y_start=y_start,
            legs=legs,
            total_steps=total_steps,
            weight=weight,
        ))

    return path_defs

__all__ = [
    "resolve_path",
    "build_path_defs_from_components",
]
