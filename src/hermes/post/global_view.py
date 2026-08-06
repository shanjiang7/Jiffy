#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a global melt-history view from segment-correction temperature snapshots.

For each snapshot in snapshots_par:
1) detect melted cells (u_nd > threshold),
2) estimate snapshot center from reconstructed path position,
3) map local melt voxels into one global 3D grid,
4) write separate VTK time series for melt history and moving source plane.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hermes.DAG.dependency import LookupRuntime
from hermes.pipelines.components import compute_dag_and_components
from hermes.pipelines.config import PipelineConfig
from hermes.runtime.config import load_config
from hermes.physics.material import phys_parameter
from hermes.utils.path_utils import build_path_defs_from_components


_LAYER_SS_RE = re.compile(r"layer_(\d+)_ss_(\d+)\.npy$")
_LAYER_STEP_RE = re.compile(r"layer_(\d+)_step_(\d+)\.npy$")
_SS_RE = re.compile(r"ss_(\d+)\.npy$")


@dataclass(frozen=True)
class SnapItem:
    layer: int
    kind: str  # "ss" | "step"
    idx: int
    path: Path


@dataclass(frozen=True)
class CompRange:
    comp_id: int
    start_ss: int
    end_ss: int  # exclusive
    start_step: int
    end_step: int  # exclusive


def _resolve_base(path: Path) -> tuple[Path, Path]:
    path = path.expanduser().resolve()
    snap_names = {"snapshots_par", "snapshots_ser", "snapshots_rank_base"}
    if path.name in snap_names and path.is_dir():
        return path.parent, path
    for snap_name in ("snapshots_par", "snapshots_ser", "snapshots_rank_base"):
        snap = path / snap_name
        if snap.is_dir():
            return path, snap
    raise FileNotFoundError(
        f"Could not find snapshots_par, snapshots_ser, or snapshots_rank_base under: {path}"
    )


def _list_snapshots(snapshot_dir: Path) -> list[SnapItem]:
    items: list[SnapItem] = []

    for p in sorted(snapshot_dir.glob("layer_*_step_*.npy")):
        m = _LAYER_STEP_RE.match(p.name)
        if m:
            items.append(SnapItem(layer=int(m.group(1)), kind="step", idx=int(m.group(2)), path=p))
    if items:
        return sorted(items, key=lambda x: (x.layer, x.idx))

    for p in sorted(snapshot_dir.glob("layer_*_ss_*.npy")):
        m = _LAYER_SS_RE.match(p.name)
        if m:
            items.append(SnapItem(layer=int(m.group(1)), kind="ss", idx=int(m.group(2)), path=p))
    if items:
        return sorted(items, key=lambda x: (x.layer, x.idx))

    for p in sorted(snapshot_dir.glob("ss_*.npy")):
        m = _SS_RE.match(p.name)
        if m:
            items.append(SnapItem(layer=0, kind="ss", idx=int(m.group(1)), path=p))
    return sorted(items, key=lambda x: (x.layer, x.idx))


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


def _position_after_steps(path_def, n_steps: int) -> tuple[float, float]:
    steps_left = max(0, min(int(n_steps), int(path_def.total_steps)))
    x = float(path_def.x_start)
    y = float(path_def.y_start)
    if steps_left == 0:
        return x, y
    for leg in path_def.legs:
        if steps_left <= 0:
            break
        take = min(steps_left, int(leg.steps))
        x += float(leg.dx_step) * take
        y += float(leg.dy_step) * take
        steps_left -= take
    return x, y


def _build_component_ranges(path_defs) -> list[CompRange]:
    out: list[CompRange] = []
    acc_ss = 0
    acc_step = 0
    for pd in path_defs:
        weight_ss = int(pd.weight)
        total_steps = int(pd.total_steps)
        out.append(
            CompRange(
                comp_id=int(pd.component_id),
                start_ss=acc_ss,
                end_ss=acc_ss + weight_ss,
                start_step=acc_step,
                end_step=acc_step + total_steps,
            )
        )
        acc_ss += weight_ss
        acc_step += total_steps
    return out


def _find_component(global_idx: int, ranges: list[CompRange], *, use_steps: bool) -> CompRange:
    for r in ranges:
        start = r.start_step if use_steps else r.start_ss
        end = r.end_step if use_steps else r.end_ss
        if start <= global_idx < end:
            return r
    key = "global_step" if use_steps else "global_ss"
    raise KeyError(f"{key}={global_idx} not found in component ranges")


def _write_legacy_vtk(
    path: Path,
    data: np.ndarray,
    *,
    scalar_name: str,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> None:
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {data.shape} for {path}")
    nx, ny, nz = (int(v) for v in data.shape)
    values = np.ascontiguousarray(data.ravel(order="F"))
    with path.open("w", encoding="ascii") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{scalar_name}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN {origin[0]:.12e} {origin[1]:.12e} {origin[2]:.12e}\n")
        f.write(f"SPACING {spacing[0]:.12e} {spacing[1]:.12e} {spacing[2]:.12e}\n")
        f.write(f"POINT_DATA {values.size}\n")
        f.write(f"SCALARS {scalar_name} unsigned_char 1\n")
        f.write("LOOKUP_TABLE default\n")
        np.savetxt(f, values.astype(np.uint8, copy=False), fmt="%d")


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


def _build_source_plane(
    gx_line: np.ndarray,
    gy_line: np.ndarray,
    gz_line: np.ndarray,
    global_nx: int,
    global_ny: int,
    global_nz: int,
) -> np.ndarray:
    plane = np.zeros((global_nx, global_ny, global_nz), dtype=np.uint8)

    valid_x = (gx_line >= 0) & (gx_line < global_nx)
    valid_y = (gy_line >= 0) & (gy_line < global_ny)
    valid_z = (gz_line >= 0) & (gz_line < global_nz)
    gx_valid = gx_line[valid_x]
    gy_valid = gy_line[valid_y]
    gz_valid = gz_line[valid_z]
    if gx_valid.size == 0 or gy_valid.size == 0 or gz_valid.size == 0:
        return plane

    x0, x1 = int(gx_valid[0]), int(gx_valid[-1])
    y0, y1 = int(gy_valid[0]), int(gy_valid[-1])
    # Laser acts on the top surface of the moving domain.
    z_top = int(gz_valid[-1])

    plane[x0 : x1 + 1, y0 : y1 + 1, z_top] = 1
    return plane


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Map local snapshots to a global melt-history voxel grid (VTK time series)."
    )
    ap.add_argument("--output-path", required=True, help="Run dir containing snapshots_par.")
    ap.add_argument("--sim-config", default="configs/examples/sim_ex1.ini", help="Simulation config path.")
    ap.add_argument("--path-config", default="configs/examples/bull.ini", help="Path/DAG config path.")
    ap.add_argument("--dt-us", type=float, help="Override solver dt in microseconds.")
    ap.add_argument("--num-layers", type=int, default=None, help="Override num_layers from path config.")
    ap.add_argument("--melt-threshold-nd", type=float, default=1.0, help="ND melting threshold.")
    ap.add_argument("--global-nx", type=int, default=512, help="Global grid points in X.")
    ap.add_argument("--global-ny", type=int, default=512, help="Global grid points in Y.")
    ap.add_argument("--write-every", type=int, default=1, help="Write VTK every N snapshots.")
    ap.add_argument(
        "--write-final-only",
        action="store_true",
        help="Accumulate all snapshots but write only the final melt-history frame.",
    )
    ap.add_argument("--max-files", type=int, default=0, help="Optional cap for quick checks (0=all).")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    base_path, snapshots = _resolve_base(project_root / args.output_path)
    sim_cfg = (project_root / args.sim_config).resolve()
    path_cfg = (project_root / args.path_config).resolve()

    if not sim_cfg.is_file():
        raise FileNotFoundError(f"sim config not found: {sim_cfg}")
    if not path_cfg.is_file():
        raise FileNotFoundError(f"path config not found: {path_cfg}")

    items = _list_snapshots(snapshots)
    if not items:
        raise RuntimeError(f"No supported snapshot files found in: {snapshots}")
    if args.max_files > 0:
        items = items[: args.max_files]

    first_u = np.load(items[0].path)
    if first_u.ndim != 3:
        raise ValueError(f"Expected 3D snapshots, got shape {first_u.shape} in {items[0].path}")
    snap_nx, snap_ny, snap_nz = map(int, first_u.shape)

    rc = load_config(sim_cfg)
    mat_override = rc.material.to_override_dict()
    t_spot_on = 2.0 * rc.laser.x_span_m / rc.laser.v
    phys = phys_parameter(rc.laser.Q, rc.laser.x_span_m, t_spot_on, mat_ch=mat_override)

    if args.dt_us is not None:
        dt_s = float(args.dt_us) * 1e-6
    elif rc.time.CFL is not None:
        dt_s = (float(rc.time.CFL) * (float(rc.level1.h_tuple[0]) ** 2)) / float(phys.kappa)
    elif rc.time.dt is not None:
        dt_s = float(rc.time.dt)
    else:
        raise ValueError("Need either [time].CFL or [time].dt in sim config.")

    h_m = float(rc.level3.h_tuple[0])
    layer_thickness_m = float(rc.layers.layer_thickness)

    pipeline_cfg = PipelineConfig.from_ini(path_cfg, num_layers=args.num_layers)
    pipeline_cfg = pipeline_cfg.with_solver_motion(
        dt_s=dt_s,
        solver_velocity_mps=rc.laser.v,
    )
    import cupy as cp

    float_type = cp.float64 if rc.float_type_str.lower() == "float64" else cp.float32
    lookup_runtime = LookupRuntime(
        rc=rc,
        phys=phys,
        float_type=float_type,
        solver_mode="fused",
        source_on_steps=pipeline_cfg.dependency.lookup_source_on_steps,
        source_substeps=pipeline_cfg.dependency.mock_numerical_source_steps,
    )
    dag = compute_dag_and_components(pipeline_cfg, lookup_runtime=lookup_runtime)
    if dag is None:
        raise RuntimeError("DAG pipeline returned no supersegments/components.")
    steps_per_ss = int(pipeline_cfg.steps_per_segment * pipeline_cfg.segments_per_supersegment)

    path_defs = build_path_defs_from_components(
        supersegments=dag.supersegments,
        components=dag.components,
        dt_s=dt_s,
        len_scale=phys.len_scale,
        coord_scale=dag.len_scale / phys.len_scale,
    )
    comp_by_id = {int(pd.component_id): pd for pd in path_defs}
    comp_ranges = _build_component_ranges(path_defs)

    # Estimate global x/y bounds from full path envelope + half local snapshot size.
    x_min_nd = float("inf")
    x_max_nd = float("-inf")
    y_min_nd = float("inf")
    y_max_nd = float("-inf")
    for pd in path_defs:
        x = float(pd.x_start)
        y = float(pd.y_start)
        x_min_nd = min(x_min_nd, x)
        x_max_nd = max(x_max_nd, x)
        y_min_nd = min(y_min_nd, y)
        y_max_nd = max(y_max_nd, y)
        for leg in pd.legs:
            x += float(leg.dx_step) * int(leg.steps)
            y += float(leg.dy_step) * int(leg.steps)
            x_min_nd = min(x_min_nd, x)
            x_max_nd = max(x_max_nd, x)
            y_min_nd = min(y_min_nd, y)
            y_max_nd = max(y_max_nd, y)

    half_wx = 0.5 * (snap_nx - 1) * h_m
    half_wy = 0.5 * (snap_ny - 1) * h_m
    margin = 2.0 * h_m
    x_min = x_min_nd * phys.len_scale - half_wx - margin
    x_max = x_max_nd * phys.len_scale + half_wx + margin
    y_min = y_min_nd * phys.len_scale - half_wy - margin
    y_max = y_max_nd * phys.len_scale + half_wy + margin

    max_layer_in_files = max(it.layer for it in items)
    num_layers = max(int(dag.num_layers), max_layer_in_files + 1)
    z_min = -(snap_nz - 1) * h_m
    z_max = (num_layers - 1) * layer_thickness_m

    global_nx = int(args.global_nx)
    global_ny = int(args.global_ny)
    global_nz = int(round((z_max - z_min) / h_m)) + 1
    global_hx = (x_max - x_min) / max(global_nx - 1, 1)
    global_hy = (y_max - y_min) / max(global_ny - 1, 1)
    global_hz = h_m
    steps_per_layer = int(dag.ss_per_layer) * int(steps_per_ss)

    cumulative = np.zeros((global_nx, global_ny, global_nz), dtype=np.uint8)

    out_root = base_path / "VTK_global"
    out_melt = out_root / "melt_history"
    out_domain = out_root / "snapshot_domain"
    out_melt.mkdir(parents=True, exist_ok=True)
    out_domain.mkdir(parents=True, exist_ok=True)
    melt_datasets: list[tuple[float, str]] = []
    domain_datasets: list[tuple[float, str]] = []

    print(f"[info] snapshots dir: {snapshots}")
    print(f"[info] snapshot count: {len(items)}")
    print(f"[info] snapshot shape: ({snap_nx}, {snap_ny}, {snap_nz})")
    print(f"[info] global grid: ({global_nx}, {global_ny}, {global_nz})")
    print(f"[info] x range [m]: {x_min:.6e} .. {x_max:.6e}")
    print(f"[info] y range [m]: {y_min:.6e} .. {y_max:.6e}")
    print(f"[info] z range [m]: {z_min:.6e} .. {z_max:.6e}")
    print(f"[info] threshold ND: {args.melt_threshold_nd:.6f}")

    frame = 0
    for i, item in enumerate(items):
        u = np.load(item.path)
        if u.shape != first_u.shape:
            raise ValueError(f"Snapshot shape mismatch in {item.path}: got {u.shape}, expected {first_u.shape}")

        if item.kind == "ss":
            global_ss = int(item.layer) * int(dag.ss_per_layer) + int(item.idx)
            cr = _find_component(global_ss, comp_ranges, use_steps=False)
            pd = comp_by_id[cr.comp_id]
            local_ss = global_ss - cr.start_ss
            n_steps = local_ss * steps_per_ss
            cx_nd, cy_nd = _position_after_steps(pd, n_steps)
        else:
            global_step = int(item.layer) * steps_per_layer + int(item.idx)
            cr = _find_component(global_step, comp_ranges, use_steps=True)
            pd = comp_by_id[cr.comp_id]
            local_step = global_step - cr.start_step
            cx_nd, cy_nd = _position_after_steps(pd, local_step)

        cx_m = cx_nd * phys.len_scale
        cy_m = cy_nd * phys.len_scale
        cz_m = float(item.layer) * layer_thickness_m
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

        snapshot_domain = _build_source_plane(
            gx_line=gx_line,
            gy_line=gy_line,
            gz_line=gz_line,
            global_nx=global_nx,
            global_ny=global_ny,
            global_nz=global_nz,
        )

        melt = np.argwhere(u > float(args.melt_threshold_nd))
        if melt.size > 0:
            gx = gx_line[melt[:, 0]]
            gy = gy_line[melt[:, 1]]
            gz = gz_line[melt[:, 2]]

            valid = (
                (gx >= 0) & (gx < global_nx)
                & (gy >= 0) & (gy < global_ny)
                & (gz >= 0) & (gz < global_nz)
            )
            cumulative[gx[valid], gy[valid], gz[valid]] = 1

        should_write = (
            i == len(items) - 1
            if bool(args.write_final_only)
            else i % max(1, int(args.write_every)) == 0 or i == len(items) - 1
        )
        if should_write:
            name = f"global_melt_{frame:07d}.vtk"
            melt_file = out_melt / name
            domain_file = out_domain / name
            _write_legacy_vtk(
                melt_file,
                cumulative,
                scalar_name="melt_history",
                spacing=(global_hx, global_hy, global_hz),
                origin=(x_min, y_min, z_min),
            )
            _write_legacy_vtk(
                domain_file,
                snapshot_domain,
                scalar_name="snapshot_domain",
                spacing=(global_hx, global_hy, global_hz),
                origin=(x_min, y_min, z_min),
            )
            melt_datasets.append((float(i), f"melt_history/{name}"))
            domain_datasets.append((float(i), f"snapshot_domain/{name}"))
            frame += 1

    melt_pvd = out_root / "global_melt_history.pvd"
    domain_pvd = out_root / "snapshot_domain_series.pvd"
    _write_pvd(melt_pvd, melt_datasets)
    _write_pvd(domain_pvd, domain_datasets)
    print(f"[ok] wrote {len(melt_datasets)} melt-history VTK files under: {out_melt}")
    print(f"[ok] wrote {len(domain_datasets)} snapshot-domain VTK files under: {out_domain}")
    print(f"[ok] wrote PVD: {melt_pvd}")
    print(f"[ok] wrote PVD: {domain_pvd}")
    print("[hint] Open both PVD files in ParaView and press Play.")


if __name__ == "__main__":
    main()
