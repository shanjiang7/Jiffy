#!/usr/bin/env python3
"""
Dump the exact_dp partition for a multi-layer config WITHOUT running the solve.

Replicates rank 0's planning path from segment_correction/main.py
(load_sim_setup -> build_runtime_plan) for a given world size, then writes:
  plan_summary.json   full planning summary (build_planning_summary)
  plan_bounds.json    [[start_frac, end_frac], ...] per rank over total SS —
                      the input format of experiments/plot_multilayer_hero.py

The planner is deterministic, so this reproduces the cuts the timing-only
scaling runs used. Single GPU, no MPI.

Usage:
  python experiments/dump_partition_plan.py --path-config configs/experiments/bull_ml15.ini \
      --world-size 64 --out-dir outputs/cr_strong_scaling_ml15/bull/plan_dump_64r
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from hermes.runtime.setup import load_sim_setup, select_float_type
from hermes.scheduling.planning import build_planning_summary, build_runtime_plan
from hermes.utils.mpi_utils import bind_local_gpu
from hermes.utils.path_utils import resolve_path
from hermes.scripts.segment_correction.main import parse_args


def main() -> None:
    # Reuse main.py's arg parser for identical defaults, then peel off the
    # dump-only options we accept on top of it.
    argv = sys.argv[1:]
    world_size = 64
    if "--world-size" in argv:
        i = argv.index("--world-size")
        world_size = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    args = parse_args(argv + ["--timing-only", "--no-export-dag"])
    args.path_complexity_report = False
    args.dependency_level_K_override = None
    args.self_check_gamma_effective = None

    bind_local_gpu()
    setup = load_sim_setup(args.config, dt_us=args.dt_us)
    project_root = setup.project_root
    path_config_path = resolve_path(project_root, args.path_config, "")
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    float_type = select_float_type(setup.rc)

    print(f"[plan-dump] {path_config_path.name} world_size={world_size} "
          f"planner={args.planner_mode} w={args.correction_weight}")
    runtime_plan = build_runtime_plan(
        args=args,
        world_size=world_size,
        path_config_path=path_config_path,
        out_dir=out_dir,
        dt_s=setup.dt_s,
        rc=setup.rc,
        phys=setup.phys,
        float_type=float_type,
        solver_velocity_mps=setup.rc.laser.v,
        export_outputs=False,
    )
    if runtime_plan is None:
        raise SystemExit("no components to plan")

    summary = build_planning_summary(
        runtime_plan=runtime_plan, args=args, world_size=world_size)
    with open(out_dir / "plan_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    ra = summary["partition"]["rank_assignments"]
    total_ss = summary["num_layers"] * summary["ss_per_layer"]
    bounds = []
    for r in sorted(ra, key=int):
        ss = ra[r]
        bounds.append([min(ss) / total_ss, (max(ss) + 1) / total_ss])
        print(f"  rank {r}: SS {min(ss)}..{max(ss)} ({len(ss)} SS)")
    with open(out_dir / "plan_bounds.json", "w") as f:
        json.dump(bounds, f)
    print(f"[ok] {out_dir}/plan_summary.json + plan_bounds.json "
          f"({total_ss} SS, {summary['num_layers']} layers, "
          f"pred skew {summary['predicted_skew']:.3f})")


if __name__ == "__main__":
    main()
