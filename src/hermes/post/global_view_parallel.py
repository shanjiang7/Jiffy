#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a parallel execution view from segment-correction snapshots.

Frames are grouped by per-rank execution index recorded during the parallel run.
Each output frame contains:
1) cumulative global melt history up to that execution frame,
2) union of all active moving domains across ranks for that frame.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tvtk.api import tvtk, write_data
from vtk.util import numpy_support

from hermes.runtime.config import load_config
from hermes.physics.material import phys_parameter


@dataclass(frozen=True)
class SnapMeta:
    file: str
    rank: int
    component_id: int
    exec_index: int
    layer: int
    kind: str
    index: int
    center_x_nd: float
    center_y_nd: float


def _resolve_base(path: Path) -> tuple[Path, Path, Path]:
    path = path.expanduser().resolve()
    if path.name == "snapshots_par" and path.is_dir():
        base = path.parent
        meta = base / "snapshots_par_meta"
        return base, path, meta
    snap = path / "snapshots_par"
    meta = path / "snapshots_par_meta"
    if snap.is_dir() and meta.is_dir():
        return path, snap, meta
    raise FileNotFoundError(f"Could not find snapshots_par and snapshots_par_meta under: {path}")


def _load_meta(meta_dir: Path) -> list[SnapMeta]:
    out: list[SnapMeta] = []
    for p in sorted(meta_dir.glob("rank_*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out.append(
                    SnapMeta(
                        file=str(d["file"]),
                        rank=int(d["rank"]),
                        component_id=int(d["component_id"]),
                        exec_index=int(d["exec_index"]),
                        layer=int(d["layer"]),
                        kind=str(d["kind"]),
                        index=int(d["index"]),
                        center_x_nd=float(d["center_x_nd"]),
                        center_y_nd=float(d["center_y_nd"]),
                    )
                )
    return out


def _write_pvd(pvd_path: Path, datasets: list[tuple[float, str]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for t, rel in datasets:
        lines.append(f'    <DataSet timestep="{t:.9f}" group="" part="0" file="{rel}"/>')
    lines += ["  </Collection>", "</VTKFile>"]
    pvd_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _add_scalar(img: tvtk.ImageData, data: np.ndarray, name: str) -> None:
    vtk_arr = numpy_support.numpy_to_vtk(np.ascontiguousarray(data.ravel(order="F")), deep=True)
    vtk_arr.SetName(name)
    img.point_data.scalars = vtk_arr


def _map_to_global_indices(
    snap_nx: int,
    snap_ny: int,
    snap_nz: int,
    h_m: float,
    cx_m: float,
    cy_m: float,
    cz_m: float,
    x_min: float,
    y_min: float,
    z_min: float,
    global_hx: float,
    global_hy: float,
    global_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = np.arange(snap_nx, dtype=np.float64)
    iy = np.arange(snap_ny, dtype=np.float64)
    iz = np.arange(snap_nz, dtype=np.float64)
    lx = (ix - 0.5 * (snap_nx - 1)) * h_m
    ly = (iy - 0.5 * (snap_ny - 1)) * h_m
    # crop_snapshot keeps the last z cells in original order: index -1 is the top surface.
    lz = (iz - float(snap_nz - 1)) * h_m
    gx = np.round((lx + cx_m - x_min) / global_hx).astype(np.int64)
    gy = np.round((ly + cy_m - y_min) / global_hy).astype(np.int64)
    gz = np.round((lz + cz_m - z_min) / global_hz).astype(np.int64)
    return gx, gy, gz


def _build_active_plane_union(
    gx_line: np.ndarray,
    gy_line: np.ndarray,
    gz_line: np.ndarray,
    global_nx: int,
    global_ny: int,
    global_nz: int,
    out: np.ndarray,
) -> None:
    valid_x = (gx_line >= 0) & (gx_line < global_nx)
    valid_y = (gy_line >= 0) & (gy_line < global_ny)
    valid_z = (gz_line >= 0) & (gz_line < global_nz)
    gx_valid = gx_line[valid_x]
    gy_valid = gy_line[valid_y]
    gz_valid = gz_line[valid_z]
    if gx_valid.size == 0 or gy_valid.size == 0 or gz_valid.size == 0:
        return
    z_top = int(gz_valid[-1])
    out[np.ix_(gx_valid, gy_valid, np.asarray([z_top], dtype=np.int64))] = 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Build parallel execution VTK series: melt_history + active_domain_union.")
    ap.add_argument("--output-path", required=True, help="Parallel run dir containing snapshots_par and snapshots_par_meta.")
    ap.add_argument("--sim-config", default="configs/examples/sim_ex1.ini", help="Simulation config path.")
    ap.add_argument("--melt-threshold-nd", type=float, default=1.0, help="ND melting threshold.")
    ap.add_argument("--global-nx", type=int, default=512, help="Global grid points in X.")
    ap.add_argument("--global-ny", type=int, default=512, help="Global grid points in Y.")
    ap.add_argument("--max-frames", type=int, default=0, help="Optional cap on execution frames (0=all).")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    base_path, snap_dir, meta_dir = _resolve_base(project_root / args.output_path)
    sim_cfg = (project_root / args.sim_config).resolve()
    if not sim_cfg.is_file():
        raise FileNotFoundError(f"sim config not found: {sim_cfg}")

    meta = _load_meta(meta_dir)
    if not meta:
        raise RuntimeError(f"No metadata records found in: {meta_dir}")

    sample = np.load(snap_dir / meta[0].file)
    if sample.ndim != 3:
        raise ValueError(f"Expected 3D snapshots, got shape {sample.shape}")
    snap_nx, snap_ny, snap_nz = map(int, sample.shape)

    rc = load_config(sim_cfg)
    mat_override = rc.material.to_override_dict()
    t_spot_on = 2.0 * rc.laser.x_span_m / rc.laser.v
    phys = phys_parameter(rc.laser.Q, rc.laser.x_span_m, t_spot_on, mat_ch=mat_override)
    h_m = float(rc.level3.h_tuple[0])
    layer_thickness_m = float(rc.layers.layer_thickness)

    half_wx = 0.5 * (snap_nx - 1) * h_m
    half_wy = 0.5 * (snap_ny - 1) * h_m
    margin = 2.0 * h_m
    xs_m = [m.center_x_nd * phys.len_scale for m in meta]
    ys_m = [m.center_y_nd * phys.len_scale for m in meta]
    x_min = min(xs_m) - half_wx - margin
    x_max = max(xs_m) + half_wx + margin
    y_min = min(ys_m) - half_wy - margin
    y_max = max(ys_m) + half_wy + margin

    num_layers = max(m.layer for m in meta) + 1
    z_min = -(snap_nz - 1) * h_m
    z_max = (num_layers - 1) * layer_thickness_m

    global_nx = int(args.global_nx)
    global_ny = int(args.global_ny)
    global_nz = int(round((z_max - z_min) / h_m)) + 1
    global_hx = (x_max - x_min) / max(global_nx - 1, 1)
    global_hy = (y_max - y_min) / max(global_ny - 1, 1)
    global_hz = h_m

    by_exec: dict[int, list[SnapMeta]] = {}
    for rec in meta:
        by_exec.setdefault(int(rec.exec_index), []).append(rec)
    exec_keys = sorted(by_exec)
    if args.max_frames > 0:
        exec_keys = exec_keys[: args.max_frames]

    cumulative = np.zeros((global_nx, global_ny, global_nz), dtype=np.uint8)
    out_root = base_path / "VTK_parallel_global"
    out_melt = out_root / "melt_history"
    out_union = out_root / "active_domain_union"
    out_melt.mkdir(parents=True, exist_ok=True)
    out_union.mkdir(parents=True, exist_ok=True)
    melt_datasets: list[tuple[float, str]] = []
    union_datasets: list[tuple[float, str]] = []

    print(f"[info] snapshots: {snap_dir}")
    print(f"[info] metadata: {meta_dir}")
    print(f"[info] execution frames: {len(exec_keys)}")
    print(f"[info] global grid: ({global_nx}, {global_ny}, {global_nz})")

    for frame, exec_idx in enumerate(exec_keys):
        active_union = np.zeros((global_nx, global_ny, global_nz), dtype=np.uint8)
        for rec in by_exec[exec_idx]:
            u = np.load(snap_dir / rec.file)
            cx_m = rec.center_x_nd * phys.len_scale
            cy_m = rec.center_y_nd * phys.len_scale
            cz_m = float(rec.layer) * layer_thickness_m
            gx_line, gy_line, gz_line = _map_to_global_indices(
                snap_nx=snap_nx,
                snap_ny=snap_ny,
                snap_nz=snap_nz,
                h_m=h_m,
                cx_m=cx_m,
                cy_m=cy_m,
                cz_m=cz_m,
                x_min=x_min,
                y_min=y_min,
                z_min=z_min,
                global_hx=global_hx,
                global_hy=global_hy,
                global_hz=global_hz,
            )
            _build_active_plane_union(gx_line, gy_line, gz_line, global_nx, global_ny, global_nz, active_union)

            melt = np.argwhere(u > float(args.melt_threshold_nd))
            if melt.size == 0:
                continue
            gx = gx_line[melt[:, 0]]
            gy = gy_line[melt[:, 1]]
            gz = gz_line[melt[:, 2]]
            valid = (
                (gx >= 0) & (gx < global_nx)
                & (gy >= 0) & (gy < global_ny)
                & (gz >= 0) & (gz < global_nz)
            )
            cumulative[gx[valid], gy[valid], gz[valid]] = 1

        img_melt = tvtk.ImageData(
            spacing=(global_hx, global_hy, global_hz),
            origin=(x_min, y_min, z_min),
            dimensions=(global_nx, global_ny, global_nz),
        )
        _add_scalar(img_melt, cumulative, "melt_history")

        img_union = tvtk.ImageData(
            spacing=(global_hx, global_hy, global_hz),
            origin=(x_min, y_min, z_min),
            dimensions=(global_nx, global_ny, global_nz),
        )
        _add_scalar(img_union, active_union, "active_domain_union")

        name = f"parallel_{frame:07d}.vti"
        write_data(img_melt, str(out_melt / name))
        write_data(img_union, str(out_union / name))
        melt_datasets.append((float(exec_idx), f"melt_history/{name}"))
        union_datasets.append((float(exec_idx), f"active_domain_union/{name}"))

    melt_pvd = out_root / "global_melt_history_parallel.pvd"
    union_pvd = out_root / "active_domain_union_parallel.pvd"
    _write_pvd(melt_pvd, melt_datasets)
    _write_pvd(union_pvd, union_datasets)
    print(f"[ok] wrote melt-history series: {melt_pvd}")
    print(f"[ok] wrote active-domain series: {union_pvd}")


if __name__ == "__main__":
    main()
