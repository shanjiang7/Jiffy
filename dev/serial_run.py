"""
Serial (single-GPU, no MPI) reference run.

Builds the same runtime plan as the parallel run, then executes the planned
runtime components sequentially as a serial reference. This keeps the saved
snapshot steps aligned with the parallel execution for later L2 comparison.

By default, snapshots are saved every 100 timesteps across the full run.

Usage:
    python src/hermes/scripts/segment_correction/serial_run.py \\
        --config configs/examples/sim_ex1.ini \\
        --path-config configs/examples/fast_heat.ini \\
        --dt-us 10 \\
        --world-size 8 \\
        --planner-mode exact_dp \\
        --out-dir outputs/serial_fast_heat
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cupy as cp
import numpy as np

from hermes.utils.mpi_utils import bind_local_gpu
from hermes.utils.path_utils import resolve_path
from hermes.physics.material import phys_parameter
from hermes.runtime.config import load_config
from hermes.scheduling.planning import build_runtime_plan
from hermes.scripts.segment_correction.outer_serial import run_sequential_outer
from hermes.scripts.segment_correction.output import (
    build_component_start_snapshot_steps,
    build_global_stride_snapshot_steps,
)
from hermes.scripts.outer_solver import build_outer_context
from hermes.utils.snapshot_utils import crop_snapshot


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Serial reference run (single GPU, no MPI)")
    p.add_argument("--config", default="configs/examples/sim_ex1.ini", help="Base simulation config")
    p.add_argument("--path-config", required=True, help="DAG laser path config")
    p.add_argument("--out-dir", default="outputs/serial", help="Output directory")
    p.add_argument("--dt-us", type=float, help="Override dt in microseconds")
    p.add_argument("--snap-every-steps", type=int, default=100,
                   help="Save a snapshot every N timesteps (default: 100).")
    p.add_argument(
        "--component-start-snapshot-mode",
        action="store_true",
        help="Save snapshots near each runtime-component start instead of using a uniform global stride.",
    )
    p.add_argument(
        "--component-start-snapshot-interval-steps",
        type=int,
        default=10,
        help="Stride inside each runtime-component snapshot window (default: 100).",
    )
    p.add_argument(
        "--component-start-snapshot-count",
        type=int,
        default=50,
        help="Maximum number of snapshots to save from each runtime-component start window (default: 10).",
    )
    p.add_argument("--num-layers", type=int, default=None,
                   help="Override num_layers from path config INI (default: use INI value)")
    p.add_argument(
        "--solver-mode",
        choices=("fused", "legacy"),
        default="fused",
        help="Level-3 outer solver mode (default: fused).",
    )
    p.add_argument(
        "--world-size",
        type=int,
        default=4,
        help="Number of ranks to emulate during planning (default: 4).",
    )
    p.add_argument(
        "--planner-mode",
        choices=("uniform", "exact_dp", "dp_monotonicity"),
        default="exact_dp",
        help="Partition planner mode used to construct runtime components (default: exact_dp).",
    )
    p.add_argument(
        "--correction-weight",
        type=float,
        default=0.75,
        help="Boundary-correction weight used in the predicted workload model (default: 0.75).",
    )
    return p.parse_args(argv)


def main(argv=None):
    bind_local_gpu()

    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[4]

    config_path = resolve_path(project_root, args.config, "configs/examples/sim_ex1.ini")
    path_config_path = resolve_path(project_root, args.path_config, "")
    out_dir = (project_root / args.out_dir).resolve()

    rc = load_config(config_path)
    float_type = cp.float64 if rc.float_type_str.lower() == "float64" else cp.float32

    mat_override = rc.material.to_override_dict()
    t_spot_on = 2.0 * rc.laser.x_span_m / rc.laser.v
    phys = phys_parameter(rc.laser.Q, rc.laser.x_span_m, t_spot_on, mat_ch=mat_override)

    if args.dt_us is not None:
        dt_s = float(args.dt_us) * 1e-6
    elif rc.time.CFL is not None:
        dt_s = (rc.time.CFL * (rc.level1.h_tuple[0] ** 2)) / phys.kappa
    elif rc.time.dt is not None:
        dt_s = rc.time.dt
    else:
        raise ValueError("Need either [time].CFL or [time].dt in sim config.")

    dt_nd = dt_s / phys.time_scale
    ctx = build_outer_context(rc, phys, float_type, dt_nd, solver_mode=args.solver_mode)
    n_all = ctx.nx * ctx.ny * ctx.nz

    print("=== Serial Reference Run ===")
    runtime_plan = build_runtime_plan(
        args=args,
        world_size=int(args.world_size),
        path_config_path=path_config_path,
        out_dir=out_dir,
        dt_s=dt_s,
        rc=rc,
        phys=phys,
        float_type=float_type,
        solver_velocity_mps=rc.laser.v,
        export_outputs=False,
    )
    if runtime_plan is None or not runtime_plan["path_defs"]:
        print("[done] no components to process.")
        sys.exit(0)

    all_path_defs = runtime_plan["path_defs"]
    steps_per_ss = int(runtime_plan["steps_per_ss"])
    num_layers = int(runtime_plan["num_layers"])
    ss_per_layer = int(runtime_plan["ss_per_layer"])
    correction_horizon_ss_map = runtime_plan["correction_horizon_ss_map"]
    runtime_components = runtime_plan["runtime_components"]

    # Group components + path_defs by layer.
    layer_groups: dict[int, list[tuple]] = defaultdict(list)
    acc_ss = 0
    for comp, pd in zip(runtime_components, all_path_defs):
        layer_idx = acc_ss // ss_per_layer
        layer_groups[layer_idx].append((comp, pd))
        acc_ss += pd.weight

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"config:       {config_path}")
    print(f"path-config:  {path_config_path}")
    print(f"dt:           {dt_s:.6e} s  ({dt_s * 1e6:.6f} us)")
    print(f"solver mode:  {args.solver_mode}")
    print(f"phys.len_scale: {phys.len_scale:.4e} m   dx_step = {rc.laser.v * dt_s / phys.len_scale:.4e} (HERMES ND = {rc.laser.v * dt_s * 1e6:.2f} um/step)")
    print(f"planner mode: {args.planner_mode}")
    print(f"global max cut depth: {int(runtime_plan.get('global_max_cut_depth', 0))}")
    print(f"Components:   {len(all_path_defs)}  ({len(all_path_defs) // max(num_layers,1)}/layer)")
    print(f"Layers:       {num_layers}  ({ss_per_layer} SS/layer)")

    ambient_gpu = cp.full((n_all,), ctx.u0, dtype=float_type)
    h_m = float(rc.level3.h_tuple[0])
    snaps_dir = out_dir / "snapshots_ser"
    snaps_dir.mkdir(parents=True, exist_ok=True)
    snapshot_stride_steps = int(args.snap_every_steps)
    snapshot_steps_by_component = None
    if args.component_start_snapshot_mode:
        snapshot_steps_by_component = build_component_start_snapshot_steps(
            all_path_defs,
            interval_steps=int(args.component_start_snapshot_interval_steps),
            max_snapshots_per_component=int(args.component_start_snapshot_count),
        )
        print(
            f"snapshot mode: component-start every {args.component_start_snapshot_interval_steps} steps, "
            f"up to {args.component_start_snapshot_count} per component"
        )
    else:
        snapshot_steps_by_component = build_global_stride_snapshot_steps(
            all_path_defs,
            ss_per_layer=ss_per_layer,
            steps_per_ss=steps_per_ss,
            snap_every_steps=snapshot_stride_steps,
        )
        print(f"snapshot mode: global every {snapshot_stride_steps} steps")
    steps_per_layer = int(ss_per_layer) * int(steps_per_ss)

    total_elapsed = 0.0
    for layer_idx in range(num_layers):
        group = layer_groups[layer_idx]
        layer_comps = [g[0] for g in group]
        layer_pds = [g[1] for g in group]

        print(f"\n[layer {layer_idx + 1}/{num_layers}] starting serial run ({len(layer_pds)} components)...")
        t0 = time.perf_counter()
        all_comp_snaps = run_sequential_outer(
            ctx=ctx,
            u_init=ambient_gpu,
            x_start=float(layer_pds[0].x_start),
            y_start=float(layer_pds[0].y_start),
            path_defs=layer_pds,
            steps_per_ss=steps_per_ss,
            snapshot_stride_steps=snapshot_stride_steps,
            snapshot_steps_by_component=[
                snapshot_steps_by_component.get(int(pd.component_id), []) for pd in layer_pds
            ] if snapshot_steps_by_component is not None else None,
        )
        elapsed = time.perf_counter() - t0
        total_elapsed += elapsed
        print(f"[layer {layer_idx + 1}/{num_layers}] done in {elapsed:.1f} s")

        layer_step_offset = 0
        for comp, pd, comp_snaps in zip(layer_comps, layer_pds, all_comp_snaps):
            for k, arr in enumerate(comp_snaps):
                rel_snapshot_steps = snapshot_steps_by_component.get(int(pd.component_id), [])
                if k >= len(rel_snapshot_steps):
                    break
                within_layer_step = layer_step_offset + int(rel_snapshot_steps[k])
                if within_layer_step >= steps_per_layer:
                    break
                fname = f"layer_{layer_idx:02d}_step_{within_layer_step:09d}.npy"
                cropped = crop_snapshot(arr, ctx.nx, ctx.ny, ctx.nz, h_m)
                np.save(snaps_dir / fname, cropped)
            layer_step_offset += comp.size * steps_per_ss

    print(f"\n[timing] serial total: {total_elapsed:.3f} s  ({num_layers} layer(s))")
    print(f"Saved snapshots to {snaps_dir}")

    with open(out_dir / "serial_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {"serial_total_seconds": total_elapsed,
             "planner_mode": str(args.planner_mode),
             "num_components": len(all_path_defs),
             "num_layers": num_layers,
             "ss_per_layer": ss_per_layer},
            f, indent=2,
        )
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
