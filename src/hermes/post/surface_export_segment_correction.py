#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export segment_correction snapshots to VTK time series for ParaView animation.

Inputs
------
- <run_dir>/snapshots_par/layer_XX_ss_YYYY.npy
- <run_dir>/snapshots_par/layer_XX_step_ZZZZZZZZZ.npy
  (or legacy snapshots_par/ss_YYYY.npy)

Each .npy is expected to be a 3D cropped temperature field in ND units.

Outputs
-------
- <run_dir>/VTK_segment/T/*.vti
- <run_dir>/VTK_segment/temperature_series.pvd
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tvtk.api import tvtk, write_data
from vtk.util import numpy_support

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
class CompRange:
    comp_id: int
    start_ss: int
    end_ss: int
    start_step: int
    end_step: int


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


def _list_snapshots(snapshots_par: Path):
    items = []
    for p in sorted(snapshots_par.glob("layer_*_step_*.npy")):
        m = _LAYER_STEP_RE.match(p.name)
        if not m:
            continue
        layer = int(m.group(1))
        step = int(m.group(2))
        items.append((layer, "step", step, p))

    if items:
        return sorted(items, key=lambda x: (x[0], x[2]))

    for p in sorted(snapshots_par.glob("layer_*_ss_*.npy")):
        m = _LAYER_SS_RE.match(p.name)
        if not m:
            continue
        layer = int(m.group(1))
        ss = int(m.group(2))
        items.append((layer, "ss", ss, p))

    if items:
        return sorted(items, key=lambda x: (x[0], x[2]))

    for p in sorted(snapshots_par.glob("ss_*.npy")):
        m = _SS_RE.match(p.name)
        if not m:
            continue
        ss = int(m.group(1))
        items.append((0, "ss", ss, p))

    return sorted(items, key=lambda x: (x[0], x[2]))


def _make_phys(config_path: Path):
    rc = load_config(config_path)
    mat_override = rc.material.to_override_dict()
    t_spot_on = 2.0 * rc.laser.x_span_m / rc.laser.v
    phys = phys_parameter(rc.laser.Q, rc.laser.x_span_m, t_spot_on, mat_ch=mat_override)
    return rc, phys


def _position_after_steps(path_def, n_steps: int) -> tuple[float, float]:
    steps_left = max(0, min(int(n_steps), int(path_def.total_steps)))
    x = float(path_def.x_start)
    y = float(path_def.y_start)
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


def _find_component_for_ss(global_ss: int, ranges: list[CompRange]) -> CompRange:
    for r in ranges:
        if r.start_ss <= global_ss < r.end_ss:
            return r
    raise KeyError(f"global_ss={global_ss} not found in component ranges")


def _find_component_for_step(global_step: int, ranges: list[CompRange]) -> CompRange:
    for r in ranges:
        if r.start_step <= global_step < r.end_step:
            return r
    raise KeyError(f"global_step={global_step} not found in component ranges")


def _write_pvd(pvd_path: Path, datasets: list[tuple[float, str]]):
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for t, rel in datasets:
        lines.append(
            f'    <DataSet timestep="{t:.9f}" group="" part="0" file="{rel}"/>'
        )
    lines += ['  </Collection>', '</VTKFile>']
    pvd_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(
        description="Export segment_correction snapshots_par/*.npy to VTK time series"
    )
    ap.add_argument(
        "--output-path",
        required=True,
        help="Run dir containing snapshots_par, or snapshots_par dir itself",
    )
    ap.add_argument(
        "--config",
        default="configs/sim_ex1.ini",
        help="Simulation config used to convert ND temperature and spacing",
    )
    ap.add_argument(
        "--to-kelvin",
        action="store_true",
        help="Convert ND temperature to Kelvin before export",
    )
    ap.add_argument(
        "--moving-origin",
        action="store_true",
        help="Place each snapshot at its reconstructed physical path position.",
    )
    ap.add_argument(
        "--path-config",
        default="configs/fast_heat.ini",
        help="Path/DAG config used to reconstruct moving snapshot positions.",
    )
    ap.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Override num_layers from path config when reconstructing positions.",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap for quick checks (0 = all)",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    base_path, snapshots_par = _resolve_base(project_root / args.output_path)
    config_path = (project_root / args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    path_config_path = (project_root / args.path_config).resolve()
    if args.moving_origin and not path_config_path.is_file():
        raise FileNotFoundError(f"path config not found: {path_config_path}")

    rc, phys = _make_phys(config_path)
    h_m = float(rc.level3.h_tuple[0])
    layer_thickness_m = float(rc.layers.layer_thickness)

    items = _list_snapshots(snapshots_par)
    if not items:
        raise RuntimeError(f"No layer_*_step_*.npy / layer_*_ss_*.npy / ss_*.npy found in {snapshots_par}")
    if args.max_files > 0:
        items = items[: args.max_files]

    path_defs = None
    comp_by_id = None
    comp_ranges = None
    ss_per_layer = None
    steps_per_ss = None
    if args.moving_origin:
        if rc.time.CFL is not None:
            dt_s = (float(rc.time.CFL) * (float(rc.level1.h_tuple[0]) ** 2)) / float(phys.kappa)
        elif rc.time.dt is not None:
            dt_s = float(rc.time.dt)
        else:
            raise ValueError("Need either [time].CFL or [time].dt in sim config.")

        pipeline_cfg = PipelineConfig.from_ini(path_config_path, num_layers=args.num_layers)
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
            source_substeps=pipeline_cfg.dependency.mock_numerical_source_steps,
        )
        dag = compute_dag_and_components(pipeline_cfg, lookup_runtime=lookup_runtime)
        if dag is None:
            raise RuntimeError("DAG pipeline returned no supersegments/components.")
        steps_per_ss = int(pipeline_cfg.steps_per_segment * pipeline_cfg.segments_per_supersegment)
        ss_per_layer = int(dag.ss_per_layer)
        path_defs = build_path_defs_from_components(
            supersegments=dag.supersegments,
            components=dag.components,
            dt_s=dt_s,
            len_scale=phys.len_scale,
            coord_scale=dag.len_scale / phys.len_scale,
        )
        comp_by_id = {int(pd.component_id): pd for pd in path_defs}
        comp_ranges = _build_component_ranges(path_defs)

    out_root = base_path / "VTK_segment"
    out_t = out_root / "T"
    out_t.mkdir(parents=True, exist_ok=True)

    datasets: list[tuple[float, str]] = []

    print(f"[info] snapshots: {snapshots_par}")
    print(f"[info] count: {len(items)}")
    print(f"[info] config: {config_path}")
    print(f"[info] spacing h: {h_m:.6e} m")
    print(f"[info] export unit: {'K' if args.to_kelvin else 'ND'}")
    print(f"[info] moving origin: {'ON' if args.moving_origin else 'OFF'}")

    for idx, (layer, key_kind, key_idx, npy_path) in enumerate(items):
        u = np.load(npy_path)
        if u.ndim != 3:
            raise ValueError(f"Expected 3D snapshot, got shape {u.shape} in {npy_path}")

        if args.to_kelvin:
            field = u.astype(np.float64, copy=False) * float(phys.deltaT) + float(phys.Ts)
            scalar_name = "Temperature_K"
        else:
            field = u.astype(np.float64, copy=False)
            scalar_name = "Temperature_ND"

        nx, ny, nz = map(int, field.shape)

        if args.moving_origin:
            if key_kind == "step":
                assert steps_per_ss is not None
                assert ss_per_layer is not None
                global_step = int(layer) * int(ss_per_layer) * int(steps_per_ss) + int(key_idx)
                cr = _find_component_for_step(global_step, comp_ranges)
                pd = comp_by_id[cr.comp_id]
                local_step = global_step - cr.start_step
                cx_nd, cy_nd = _position_after_steps(pd, local_step)
            else:
                assert steps_per_ss is not None
                assert ss_per_layer is not None
                global_ss = int(layer) * int(ss_per_layer) + int(key_idx)
                cr = _find_component_for_ss(global_ss, comp_ranges)
                pd = comp_by_id[cr.comp_id]
                local_ss = global_ss - cr.start_ss
                cx_nd, cy_nd = _position_after_steps(pd, local_ss * int(steps_per_ss))

            cx_m = float(cx_nd) * float(phys.len_scale)
            cy_m = float(cy_nd) * float(phys.len_scale)
            cz_m = float(layer) * layer_thickness_m
            origin = (
                cx_m - 0.5 * (nx - 1) * h_m,
                cy_m - 0.5 * (ny - 1) * h_m,
                cz_m - (nz - 1) * h_m,
            )
        else:
            # Cropped ROI is centered in x/y in solver output; place origin accordingly.
            origin = (
                -0.5 * (nx - 1) * h_m,
                -0.5 * (ny - 1) * h_m,
                -(nz - 1) * h_m,
            )

        img = tvtk.ImageData(spacing=(h_m, h_m, h_m), origin=origin, dimensions=(nx, ny, nz))
        arr = np.ascontiguousarray(field.ravel(order="F"))
        vtk_arr = numpy_support.numpy_to_vtk(arr, deep=True)
        vtk_arr.SetName(scalar_name)
        img.point_data.scalars = vtk_arr

        if key_kind == "step":
            out_name = f"T_layer_{layer:02d}_step_{key_idx:09d}.vti"
        else:
            out_name = f"T_layer_{layer:02d}_ss_{key_idx:04d}.vti"
        out_file = out_t / out_name
        write_data(img, str(out_file))

        rel = f"T/{out_name}"
        datasets.append((float(idx), rel))

    pvd_path = out_root / "temperature_series.pvd"
    _write_pvd(pvd_path, datasets)

    print(f"[ok] wrote {len(datasets)} VTI files under: {out_t}")
    print(f"[ok] wrote PVD: {pvd_path}")
    print("[hint] Open temperature_series.pvd in ParaView and press Play.")


if __name__ == "__main__":
    main()
