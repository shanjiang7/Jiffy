# Modified from path_loader.py in hermes-gpu-heat repository.
from __future__ import annotations

from pathlib import Path

import numpy as np

from .path_builders import (
    build_single_line_nd,
    build_segments_nd,
    build_raster_nd,
    build_island_raster_sections_nd,
    build_hilbert_nd,
    build_waypoints_nd,
    build_picture_nd,
    build_continuous_hybrid_sections_nd,
    concatenate_sections_nd,
)
from hermes.config.units import parse_length_expr
from hermes.utils.laser_io import load_cfg


def build_path_sections_nd_from_ini(
    path_ini: str | Path,
    len_scale: float,
    step_nd: float | None = None,
) -> list[tuple[np.ndarray, bool]]:
    """
    Returns a list of path sections in non-dimensional units.

    Each section is `(waypoints_nd, source_on)`. Legacy path modes emit a single
    source-on section; hybrid modes may emit multiple sections with explicit
    source-off connectors.
    """
    path_ini = Path(path_ini).expanduser().resolve()
    cfg = load_cfg(path_ini)

    # 1) single
    if "path.single" in cfg:
        sec = cfg["path.single"]
        x0 = parse_length_expr(sec.get("x0", "0"))
        y0 = parse_length_expr(sec.get("y0", "0"))
        length_m = parse_length_expr(sec["length"])
        dir_str = sec.get("dir", "+y")
        return [(
            build_single_line_nd(
                start_xy_m=(x0, y0),
                length_m=length_m,
                dir_str=dir_str,
                len_scale=len_scale,
            ),
            True,
        )]

    # 2) raster x-major
    if "path.raster_x" in cfg:
        sec = cfg["path.raster_x"]
        x0 = parse_length_expr(sec.get("x0", "0"))
        y0 = parse_length_expr(sec.get("y0", "0"))
        width_m = parse_length_expr(sec["width"])
        height_m = parse_length_expr(sec["height"])
        line_pitch_m = parse_length_expr(sec["line_pitch"]) if "line_pitch" in sec else None
        passes = int(sec["passes"]) if "passes" in sec else None
        x_dir = int(sec.get("x_dir", "1"))
        return [(
            build_raster_nd(
                origin_m=(x0, y0), width_m=width_m, height_m=height_m,
                line_pitch_m=line_pitch_m, passes=passes,
                x_dir_sign=x_dir, x_major=True, len_scale=len_scale,
            ),
            True,
        )]

    # 3) raster y-major
    if "path.raster_y" in cfg:
        sec = cfg["path.raster_y"]
        x0 = parse_length_expr(sec.get("x0", "0"))
        y0 = parse_length_expr(sec.get("y0", "0"))
        width_m = parse_length_expr(sec["width"])
        height_m = parse_length_expr(sec["height"])
        line_pitch_m = parse_length_expr(sec["line_pitch"]) if "line_pitch" in sec else None
        passes = int(sec["passes"]) if "passes" in sec else None
        y_dir = int(sec.get("y_dir", "1"))
        return [(
            build_raster_nd(
                origin_m=(x0, y0), width_m=width_m, height_m=height_m,
                line_pitch_m=line_pitch_m, passes=passes,
                x_dir_sign=y_dir, x_major=False, len_scale=len_scale,
            ),
            True,
        )]

    # 4) segments
    if "path.segments" in cfg:
        sec = cfg["path.segments"]
        x0 = parse_length_expr(sec.get("x0", "0"))
        y0 = parse_length_expr(sec.get("y0", "0"))
        repeat = int(sec.get("repeat", "1"))
        segments = []
        for key in sorted(
            (k for k in sec if k.startswith("segment.")),
            key=lambda s: int(s.split(".")[1]),
        ):
            line = sec.get(key, "").strip()
            if not line:
                continue
            toks = [t.strip() for t in line.split(",")]
            seg = {}
            for t in toks:
                if "=" not in t:
                    continue
                k, v = [x.strip() for x in t.split("=", 1)]
                if k == "dir":
                    seg["dir"] = v
                elif k == "length":
                    seg["length_m"] = parse_length_expr(v)
                elif k == "vx":
                    seg["vx"] = float(v)
                elif k == "vy":
                    seg["vy"] = float(v)
            if "length_m" not in seg:
                raise ValueError(f"{key} missing length=...")
            if "dir" not in seg and not {"vx", "vy"} <= set(seg.keys()):
                raise ValueError(f"{key} needs either dir=... or (vx,vy)=...")
            segments.append(seg)
        return [(
            build_segments_nd(
                start_xy_m=(x0, y0),
                segments=segments,
                repeat=repeat,
                len_scale=len_scale,
            ),
            True,
        )]

    # 5) explicit waypoints
    if "path.waypoints" in cfg:
        sec = cfg["path.waypoints"]
        raw = sec["points"]

        def _parse_pts(s: str):
            out = []
            for chunk in s.split(")"):
                c = chunk.strip().lstrip(",").strip()
                if not c:
                    continue
                if c[0] == "(":
                    c = c[1:]
                xy = [t.strip() for t in c.split(",")]
                if len(xy) != 2:
                    continue
                out.append((parse_length_expr(xy[0]), parse_length_expr(xy[1])))
            return out

        pts_m = _parse_pts(raw)
        close_loop = sec.get("close_loop", "false").strip().lower() in {"1", "true", "on", "yes"}
        return [(
            build_waypoints_nd(waypoints_m=pts_m, close_loop=close_loop, len_scale=len_scale),
            True,
        )]

    # 6) island raster
    if "path.island_raster" in cfg:
        sec = cfg["path.island_raster"]
        x0 = parse_length_expr(sec.get("x0", "0"))
        y0 = parse_length_expr(sec.get("y0", "0"))
        island_width_m = parse_length_expr(sec["island_width"])
        island_height_m = parse_length_expr(sec["island_height"])
        line_pitch_m = parse_length_expr(sec["line_pitch"])
        rows = int(sec["rows"])
        cols = int(sec["cols"])
        island_gap_x_m = parse_length_expr(sec.get("island_gap_x", "0"))
        island_gap_y_m = parse_length_expr(sec.get("island_gap_y", "0"))
        scan_axis = sec.get("scan_axis", "x").strip().lower()
        if scan_axis not in {"x", "y"}:
            raise ValueError("path.island_raster scan_axis must be 'x' or 'y'")
        raster_dir_sign = int(sec.get("raster_dir", "1"))
        raster_dir_mode = sec.get("raster_dir_mode", "fixed").strip().lower()
        snake = sec.get("snake", "true").strip().lower() in {"1", "true", "yes", "on"}
        return build_island_raster_sections_nd(
            origin_m=(x0, y0),
            island_width_m=island_width_m,
            island_height_m=island_height_m,
            line_pitch_m=line_pitch_m,
            rows=rows,
            cols=cols,
            island_gap_x_m=island_gap_x_m,
            island_gap_y_m=island_gap_y_m,
            x_major=(scan_axis == "x"),
            raster_dir_sign=raster_dir_sign,
            raster_dir_mode=raster_dir_mode,
            snake=snake,
            len_scale=len_scale,
        )

    # 7) hilbert
    if "path.hilbert" in cfg:
        sec = cfg["path.hilbert"]
        x0 = parse_length_expr(sec.get("x0", "0"))
        y0 = parse_length_expr(sec.get("y0", "0"))
        size_m = parse_length_expr(sec["size"])
        order = int(sec["order"])
        return [(
            build_hilbert_nd(
                origin_m=(x0, y0),
                size_m=size_m,
                order=order,
                len_scale=len_scale,
            ),
            True,
        )]

    # 8) picture
    if "path.picture" in cfg:
        sec = cfg["path.picture"]
        img_raw = sec["image"]
        img_path = Path(img_raw)
        if not img_path.is_absolute():
            img_path = (path_ini.parent / img_path).resolve()

        horizontal_m = parse_length_expr(sec["horizontal"])
        vertical_m = parse_length_expr(sec["vertical"]) if "vertical" in sec else None

        column_res_x_m = None
        column_res_y_m = None
        column_res_m = None

        if "column_res_x" in sec and "column_res_y" in sec:
            column_res_x_m = parse_length_expr(sec["column_res_x"])
            column_res_y_m = parse_length_expr(sec["column_res_y"])
        elif "column_res" in sec:
            column_res_m = parse_length_expr(sec["column_res"])

        n = int(sec["n"]) if "n" in sec else None
        anchor_to_first = sec.get("anchor_to_first", "true").strip().lower() in {"1", "true", "yes", "on"}
        return [(
            build_picture_nd(
                img_path=str(img_path),
                horizontal_length_m=horizontal_m,
                vertical_length_m=vertical_m,
                len_scale=len_scale,
                column_res_x_m=column_res_x_m,
                column_res_y_m=column_res_y_m,
                column_res_m=column_res_m,
                anchor_to_first=anchor_to_first,
                n=n,
                step_nd=step_nd,
            ),
            True,
        )]

    # 9) continuous hybrid: interleaved double square spiral + raster,
    #     one uninterrupted laser-on stroke (no travel moves, no connectors)
    if "path.continuous_hybrid" in cfg:
        sec = cfg["path.continuous_hybrid"]
        x0 = parse_length_expr(sec.get("x0", "0"))
        y0 = parse_length_expr(sec.get("y0", "0"))
        spiral_side_m = parse_length_expr(sec["spiral_side"])
        track_pitch_m = parse_length_expr(sec["track_pitch"])
        raster_width_m = parse_length_expr(sec["raster_width"])
        raster_line_pitch_m = parse_length_expr(sec["raster_line_pitch"])
        gap_m = parse_length_expr(sec.get("gap", "0"))
        tile_gap_m = parse_length_expr(sec.get("tile_gap", "0"))
        repeats = int(sec.get("repeats", "1"))
        return build_continuous_hybrid_sections_nd(
            origin_m=(x0, y0),
            spiral_side_m=spiral_side_m,
            track_pitch_m=track_pitch_m,
            raster_width_m=raster_width_m,
            raster_line_pitch_m=raster_line_pitch_m,
            gap_m=gap_m,
            tile_gap_m=tile_gap_m,
            repeats=repeats,
            len_scale=len_scale,
        )

    raise ValueError("No [path.*] section found in the provided INI.")


def build_waypoints_nd_from_ini(path_ini: str | Path, len_scale: float, step_nd: float | None = None) -> "np.ndarray":
    """
    Returns non-dimensional waypoints (shape (P,2)).

    Supported path modes:
    - [path.single]: Single-line scan
    - [path.raster_x]: X-major raster/serpentine scan
    - [path.raster_y]: Y-major raster/serpentine scan
    - [path.segments]: Piecewise segments
    - [path.waypoints]: Explicit waypoints
    - [path.island_raster]: Grid of raster-filled islands with source-off travel between islands
    - [path.hilbert]: Continuous Hilbert space-filling curve
    - [path.picture]: Image-based path with column scanning
    - [path.continuous_hybrid]: Interleaved double square spiral + raster, one laser-on stroke
    """
    sections = build_path_sections_nd_from_ini(path_ini, len_scale=len_scale, step_nd=step_nd)
    return concatenate_sections_nd(sections)
