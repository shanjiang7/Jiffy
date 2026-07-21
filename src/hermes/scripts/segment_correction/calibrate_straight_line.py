"""
Straight-line calibration experiment (paper Sec. II-C, Table I / Fig. 3).

For each candidate segment length Lseg, the scan path is three consecutive
collinear segments s1, s2, s3. By linearity, the error committed at the start
of s3 when the s1 -> s3 dependency is truncated equals the thermal residue of
s1 at time t2:

    u_full(t2)  : s1 + s2 simulated source-on from ambient (path length 2L)
    u_local(t2) : s2 only simulated source-on from ambient (starts at x = L)
    delta       = u_full(t2) - u_local(t2)     (s1's inherited contribution)

Both runs end with the laser at x = 2L, so their moving domains coincide and
delta is a pointwise field difference. Reported per Lseg over an evaluation
region centred on the laser position (crop_snapshot ROI):

    epsilon_K : max |delta| in Kelvin        (Table I column 2)
    rel_l2    : ||delta|| / ||u_full||       (Table I column 3)

The published calibration (paper Table I) was generated with the h = 30 um grid
(configs/examples/sim_calibration.ini, the default --config) and the default
1 mm evaluation ROI; this script reproduces every Table I row to 3-4
significant digits with those settings. Finer grids (e.g. sim_ex1.ini,
h = 18 um) yield smaller residues, so the published table is conservative for
production runs.

Usage:
    python src/hermes/scripts/segment_correction/calibrate_straight_line.py \
        --config configs/examples/sim_calibration.ini \
        --dt-us 10 \
        --lseg-mm 0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4 \
        --out-dir outputs/calibration
"""
from __future__ import annotations

import argparse
import json

import cupy as cp
import numpy as np

from hermes.DAG.dependency import (
    CALIBRATION_EPSILON_K,
    CALIBRATION_LSEG_MM,
    CALIBRATION_REL_L2,
)
from hermes.motion.types import PathLeg
from hermes.runtime.setup import load_sim_setup, select_float_type
from hermes.scripts.outer_solver import build_outer_context, run_ss_outer
from hermes.utils.mpi_utils import bind_local_gpu
from hermes.utils.snapshot_utils import crop_snapshot


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Straight-line calibration (Table I / Fig. 3)")
    p.add_argument(
        "--config",
        default="configs/examples/sim_calibration.ini",
        help="Base simulation config (default: the h=30um grid used for paper Table I).",
    )
    p.add_argument("--dt-us", type=float, default=10.0, help="Timestep in microseconds (default: 10)")
    p.add_argument(
        "--lseg-mm",
        default="0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4",
        help="Comma-separated candidate segment lengths in mm (default: Table I sweep).",
    )
    p.add_argument(
        "--roi-xy-mm",
        type=float,
        default=1.0,
        help="Side length (mm) of the square evaluation region centred on the laser (default: 1.0).",
    )
    p.add_argument(
        "--roi-z-mm",
        type=float,
        default=1.0,
        help="Retained z extent (mm) near the top surface for the evaluation region (default: 1.0).",
    )
    p.add_argument("--out-dir", default="outputs/calibration", help="Output directory")
    return p.parse_args(argv)


def _simulate_final_state(
    ctx,
    ambient_gpu: cp.ndarray,
    *,
    x_start_nd: float,
    dx_step_nd: float,
    n_steps: int,
) -> cp.ndarray:
    legs = [PathLeg(dx_step=float(dx_step_nd), dy_step=0.0, steps=int(n_steps), source_on=True)]
    _, final_u = run_ss_outer(
        ctx,
        ambient_gpu,
        float(x_start_nd),
        0.0,
        legs,
        int(n_steps),
        source_on=True,
        snapshot_steps=[],
    )
    return final_u


def main(argv=None):
    bind_local_gpu()
    args = parse_args(argv)
    setup = load_sim_setup(
        args.config, dt_us=args.dt_us, default_config="configs/examples/sim_calibration.ini"
    )
    project_root, config_path = setup.project_root, setup.config_path
    rc, phys, dt_s = setup.rc, setup.phys, setup.dt_s
    float_type = select_float_type(rc)
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lseg_list_mm = [float(tok) for tok in str(args.lseg_mm).split(",") if tok.strip()]
    if not lseg_list_mm:
        raise ValueError("--lseg-mm produced an empty sweep.")

    dt_nd = setup.dt_nd
    ctx = build_outer_context(rc, phys, float_type, dt_nd, solver_mode="fused")
    n_all = ctx.nx * ctx.ny * ctx.nz
    h_m = float(rc.level3.h_tuple[0])
    step_m = float(rc.laser.v) * dt_s
    step_nd = step_m / float(phys.len_scale)

    print("=== Straight-Line Calibration (Table I / Fig. 3) ===")
    print(f"config:  {config_path}")
    print(f"dt:      {dt_s*1e6:.3f} us   step: {step_m*1e6:.2f} um")
    print(f"grid:    {ctx.nx}x{ctx.ny}x{ctx.nz}  h={h_m*1e6:.0f} um")
    print(f"eval ROI: {args.roi_xy_mm:.2f} mm (xy) x {args.roi_z_mm:.2f} mm (z, from top surface)")
    print(f"Lseg sweep (mm): {lseg_list_mm}")

    ambient_gpu = cp.full((n_all,), ctx.u0, dtype=float_type)
    rows = []
    for lseg_mm in lseg_list_mm:
        lseg_m = float(lseg_mm) * 1e-3
        n_steps = int(round(lseg_m / step_m))
        if n_steps < 1:
            raise ValueError(f"Lseg={lseg_mm} mm yields zero steps at dt={dt_s} s.")
        lseg_actual_mm = n_steps * step_m * 1e3
        lseg_nd = n_steps * step_nd

        # s1 + s2 from ambient (laser ends at x = 2L)
        u_full = _simulate_final_state(
            ctx, ambient_gpu, x_start_nd=0.0, dx_step_nd=step_nd, n_steps=2 * n_steps
        )
        # s2 only from ambient (starts at x = L, ends at x = 2L)
        u_local = _simulate_final_state(
            ctx, ambient_gpu, x_start_nd=lseg_nd, dx_step_nd=step_nd, n_steps=n_steps
        )
        cp.cuda.Stream.null.synchronize()

        u_full_h = cp.asnumpy(u_full).astype(np.float64)
        u_local_h = cp.asnumpy(u_local).astype(np.float64)
        del u_full, u_local
        cp.get_default_memory_pool().free_all_blocks()

        roi_kwargs = dict(
            nx=ctx.nx, ny=ctx.ny, nz=ctx.nz, h_m=h_m,
            roi_xy_m=float(args.roi_xy_mm) * 1e-3,
            roi_z_m=float(args.roi_z_mm) * 1e-3,
        )
        full_roi = crop_snapshot(u_full_h, **roi_kwargs)
        local_roi = crop_snapshot(u_local_h, **roi_kwargs)
        delta_roi = full_roi - local_roi

        eps_K = float(np.max(np.abs(delta_roi))) * float(phys.deltaT)
        denom = float(np.linalg.norm(full_roi))
        rel_l2 = float(np.linalg.norm(delta_roi)) / max(denom, 1e-300)
        # Alternative normalisation: temperature rise above ambient.
        denom_rise = float(np.linalg.norm(full_roi - float(ctx.u0)))
        rel_l2_rise = float(np.linalg.norm(delta_roi)) / max(denom_rise, 1e-300)

        rows.append({
            "lseg_mm": float(lseg_mm),
            "lseg_actual_mm": float(lseg_actual_mm),
            "n_steps_per_segment": int(n_steps),
            "epsilon_K": eps_K,
            "rel_l2": rel_l2,
            "rel_l2_vs_rise": rel_l2_rise,
        })
        print(
            f"  Lseg={lseg_mm:5.2f} mm ({n_steps:4d} steps): "
            f"eps={eps_K:.3e} K   rel_l2={rel_l2:.3e}   rel_l2_vs_rise={rel_l2_rise:.3e}",
            flush=True,
        )

    # Console comparison against the built-in calibration table (paper Table I).
    builtin = {
        float(l): (float(e), float(r))
        for l, e, r in zip(CALIBRATION_LSEG_MM, CALIBRATION_EPSILON_K, CALIBRATION_REL_L2)
    }
    print("\n  Lseg(mm)   eps_K measured   eps_K Table I   rel_l2 measured   rel_l2 Table I")
    for row in rows:
        ref = builtin.get(float(row["lseg_mm"]))
        ref_eps = f"{ref[0]:.3e}" if ref else "     -    "
        ref_rel = f"{ref[1]:.3e}" if ref else "     -    "
        print(
            f"  {row['lseg_mm']:7.2f}   {row['epsilon_K']:.3e}        {ref_eps}       "
            f"{row['rel_l2']:.3e}         {ref_rel}"
        )

    csv_path = out_dir / "calibration.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("lseg_mm,lseg_actual_mm,n_steps_per_segment,epsilon_K,rel_l2,rel_l2_vs_rise\n")
        for row in rows:
            f.write(
                f"{row['lseg_mm']},{row['lseg_actual_mm']},{row['n_steps_per_segment']},"
                f"{row['epsilon_K']:.6e},{row['rel_l2']:.6e},{row['rel_l2_vs_rise']:.6e}\n"
            )

    summary = {
        "config": str(config_path),
        "dt_us": float(args.dt_us),
        "roi_xy_mm": float(args.roi_xy_mm),
        "roi_z_mm": float(args.roi_z_mm),
        "solver_mode": "fused",
        "rows": rows,
        "builtin_table": {
            "lseg_mm": [float(x) for x in CALIBRATION_LSEG_MM],
            "epsilon_K": [float(x) for x in CALIBRATION_EPSILON_K],
            "rel_l2": [float(x) for x in CALIBRATION_REL_L2],
        },
    }
    json_path = out_dir / "calibration_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
