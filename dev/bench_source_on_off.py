#!/usr/bin/env python3
"""
Measure the source-off / source-on per-step cost ratio.

This ratio is exactly the `correction_weight` of the partitioner's predicted
workload model: `_boundary_cut_correction_from_depth` charges

    correction_cost = correction_weight * (segments worth of correction steps)

against a base cost counted in the same units (segments worth of source-on
steps). So the physically correct weight is

    weight = (time of N source-off steps) / (time of N source-on steps)

Both passes traverse an identical straight path of the same length and step
count; only the source term differs.

Usage:
    python dev/bench_source_on_off.py --config configs/examples/sim_ex1.ini
    python dev/bench_source_on_off.py --config configs/examples/sim_calibration.ini --steps 2000
"""
from __future__ import annotations

import argparse
import time

import cupy as cp

from hermes.motion.types import PathLeg
from hermes.runtime.setup import load_sim_setup, select_float_type
from hermes.scripts.outer_solver import build_outer_context, run_ss_outer
from hermes.utils.mpi_utils import bind_local_gpu


def timed_run(ctx, ambient, *, step_nd: float, n_steps: int, source_on: bool) -> float:
    legs = [PathLeg(dx_step=float(step_nd), dy_step=0.0, steps=int(n_steps), source_on=bool(source_on))]
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    run_ss_outer(
        ctx,
        ambient,
        0.0,
        0.0,
        legs,
        int(n_steps),
        source_on=bool(source_on),
        max_steps=int(n_steps),
    )
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--config", default="configs/examples/sim_ex1.ini", help="Simulation config (grid)")
    p.add_argument("--dt-us", type=float, default=10.0, help="Timestep in microseconds (default: 10)")
    p.add_argument("--steps", type=int, default=1000, help="Steps per timed pass (default: 1000)")
    p.add_argument("--repeats", type=int, default=3, help="Timed repetitions per mode (default: 3)")
    p.add_argument("--warmup", type=int, default=100, help="Warm-up steps before timing (default: 100)")
    p.add_argument("--solver-mode", choices=("fused", "legacy"), default="fused")
    args = p.parse_args()

    bind_local_gpu()
    setup = load_sim_setup(args.config, dt_us=args.dt_us)
    rc, phys = setup.rc, setup.phys
    float_type = select_float_type(rc)
    ctx = build_outer_context(rc, phys, float_type, setup.dt_nd, solver_mode=args.solver_mode)

    n_all = ctx.nx * ctx.ny * ctx.nz
    ambient = cp.full((n_all,), ctx.u0, dtype=float_type)
    step_nd = float(rc.laser.v) * setup.dt_s / float(phys.len_scale)

    print("=== source-on / source-off cost benchmark ===")
    print(f"config     : {setup.config_path}")
    print(f"grid       : {ctx.nx}x{ctx.ny}x{ctx.nz}  (h = {rc.level3.h_tuple[0]*1e6:.0f} um)")
    print(f"dt         : {setup.dt_s*1e6:.2f} us   step = {rc.laser.v*setup.dt_s*1e6:.1f} um")
    print(f"steps/pass : {args.steps}   repeats = {args.repeats}")

    # warm-up (JIT, allocator, autotuning)
    timed_run(ctx, ambient, step_nd=step_nd, n_steps=args.warmup, source_on=True)
    timed_run(ctx, ambient, step_nd=step_nd, n_steps=args.warmup, source_on=False)

    on_times, off_times = [], []
    for i in range(args.repeats):
        on = timed_run(ctx, ambient, step_nd=step_nd, n_steps=args.steps, source_on=True)
        off = timed_run(ctx, ambient, step_nd=step_nd, n_steps=args.steps, source_on=False)
        on_times.append(on)
        off_times.append(off)
        print(
            f"  rep {i+1}: source-on {on:8.3f} s ({on/args.steps*1e3:6.3f} ms/step)   "
            f"source-off {off:8.3f} s ({off/args.steps*1e3:6.3f} ms/step)   "
            f"ratio {off/on:5.3f}"
        )

    best_on, best_off = min(on_times), min(off_times)
    print("")
    print(f"best-of-{args.repeats}: source-on {best_on:.3f} s, source-off {best_off:.3f} s")
    print(f"correction_weight (source-off / source-on) = {best_off/best_on:.3f}")


if __name__ == "__main__":
    main()
