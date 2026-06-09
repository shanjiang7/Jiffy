from __future__ import annotations

"""
PROJECT_DIR=/scratch/10226/shawnraul/Parallel_Hermes
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

python src/hermes/scripts/segment_correction/plan_only.py \
  --config configs/sim_ex1.ini \
  --path-config configs/fast_heat.ini \
  --dt-us 10 \
  --world-size 4 \
  --planner-mode exact_dp \
  --out-dir outputs/plan_nosplit
"""

import argparse
import json
import sys
from pathlib import Path

import cupy as cp

from hermes.physics.material import phys_parameter
from hermes.runtime.config import load_config
from hermes.scheduling.planning import (
    build_runtime_plan,
    build_planning_summary,
    compute_dt_s,
    print_run_summary,
    print_split_records,
)
from hermes.utils.path_utils import resolve_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Planning-only preview for segment correction")
    p.add_argument("--config", default="configs/sim_ex1.ini", help="Base simulation config")
    p.add_argument("--path-config", required=True, help="DAG laser path config")
    p.add_argument("--out-dir", default="outputs/segment_plan", help="Output directory")
    p.add_argument("--dt-us", type=float, help="Override dt in microseconds")
    p.add_argument(
        "--world-size",
        type=int,
        default=4,
        help="Number of ranks to emulate during planning (default: 4).",
    )
    p.add_argument(
        "--solver-mode",
        choices=("fused", "legacy"),
        default="fused",
        help="Solver mode used by numerical lookup.",
    )
    p.add_argument(
        "--planner-mode",
        choices=("uniform", "exact_dp", "dp_monotonicity"),
        default="exact_dp",
        help="Partition planner mode (default: exact_dp).",
    )
    p.add_argument(
        "--correction-weight",
        type=float,
        default=0.75,
        help="Boundary-correction weight used in the predicted workload model (default: 0.75).",
    )
    p.add_argument(
        "--snap-every-steps",
        type=int,
        default=None,
        help="Unused in planning-only mode; kept for summary compatibility.",
    )
    p.add_argument(
        "--export-dag",
        action="store_true",
        help="Export DAG plots/CSVs during planning-only runs.",
    )
    p.add_argument(
        "--verify-dp-monotonicity",
        action="store_true",
        help=(
            "For exact_dp or dp_monotonicity, check whether optimal cut positions satisfy "
            "opt[p][j] <= opt[p][j+1]."
        ),
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if int(args.world_size) < 1:
        raise ValueError("--world-size must be >= 1")

    project_root = Path(__file__).resolve().parents[4]
    config_path = resolve_path(project_root, args.config, "configs/sim_ex1.ini")
    path_config_path = resolve_path(project_root, args.path_config, "")
    out_dir = (project_root / args.out_dir).resolve()

    rc = load_config(config_path)
    float_type = cp.float64 if rc.float_type_str.lower() == "float64" else cp.float32

    mat_override = rc.material.to_override_dict()
    t_spot_on = 2.0 * rc.laser.x_span_m / rc.laser.v
    phys = phys_parameter(rc.laser.Q, rc.laser.x_span_m, t_spot_on, mat_ch=mat_override)
    dt_s = compute_dt_s(args, rc, phys)

    print("=== Planning-Only Segment Correction Preview ===")
    print(f"world_size (emulated ranks): {args.world_size}")
    search_summary = None
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
        export_outputs=bool(args.export_dag),
    )
    if runtime_plan is None:
        print("[done] no components to process.")
        sys.exit(0)

    print_split_records(
        runtime_plan["split_records"],
        runtime_plan["runtime_components"],
        runtime_plan["source_components"],
    )
    print_run_summary(
        args=args,
        config_path=config_path,
        path_config_path=path_config_path,
        dt_s=dt_s,
        phys=phys,
        rc=rc,
        num_layers=runtime_plan["num_layers"],
        ss_per_layer=runtime_plan["ss_per_layer"],
        path_defs=runtime_plan["path_defs"],
        rank_assignments=runtime_plan["rank_assignments"],
        rank_pred_loads=runtime_plan["rank_pred_loads"],
        global_max_cut_depth=int(runtime_plan.get("global_max_cut_depth", 0)),
    )
    print(f"segments_per_supersegment: {runtime_plan['segments_per_supersegment']}")
    dp_algorithm = runtime_plan["partition_summary"].get("dp_algorithm")
    if dp_algorithm is not None:
        print(
            "[dp] "
            f"algorithm={dp_algorithm} "
            f"transition_evaluations="
            f"{int(runtime_plan['partition_summary'].get('dp_transition_evaluations', 0))}"
        )
    dp_monotonicity = runtime_plan["partition_summary"].get("dp_monotonicity")
    if dp_monotonicity is not None:
        status = "PASS" if bool(dp_monotonicity.get("holds", False)) else "FAIL"
        print(
            "[dp-monotonicity] "
            f"{status}: violations={int(dp_monotonicity.get('num_violations', 0))} "
            f"states_checked={int(dp_monotonicity.get('num_states_checked', 0))}"
        )

    summary = build_planning_summary(
        runtime_plan=runtime_plan,
        args=args,
        world_size=int(args.world_size),
        search_summary=search_summary,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "planning_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote planning summary to {summary_path}")


if __name__ == "__main__":
    main()
