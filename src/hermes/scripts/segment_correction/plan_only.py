from __future__ import annotations

"""
Planning-only preview: build the dependency DAG and rank partition without
solving. Usage (from the repo root, after `source env_vista.sh`):

python src/hermes/scripts/segment_correction/plan_only.py \
  --config configs/examples/sim_ex1.ini \
  --path-config configs/examples/fast_heat.ini \
  --dt-us 10 --world-size 4 --planner-mode exact_dp
"""

import argparse
import json
import sys


from hermes.runtime.setup import load_sim_setup, select_float_type
from hermes.scheduling.planning import (
    build_runtime_plan,
    build_planning_summary,
    print_run_summary,
    print_split_records,
)
from hermes.scripts.segment_correction.diagnostic_config import load_path_complexity_options
from hermes.utils.path_utils import resolve_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Planning-only preview for segment correction")
    p.add_argument("--config", default="configs/examples/sim_ex1.ini", help="Base simulation config")
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
    p.add_argument(
        "--path-complexity",
        action="store_true",
        help="Enable path-complexity threshold adjustment using --path-complexity-config.",
    )
    p.add_argument(
        "--path-complexity-config",
        default="configs/path_complexity.ini",
        help="INI file with [path_complexity] settings.",
    )
    p.add_argument(
        "--path-complexity-report",
        action="store_true",
        help="Compute and print A_path without changing the configured level_K.",
    )
    p.add_argument(
        "--path-complexity-target-rel-l2",
        type=float,
        default=None,
        help=(
            "Target global relative L2 error. If set, compute A_path, use "
            "local_target=target/A_path with the built-in calibration table, "
            "update level_K, and rebuild the DAG."
        ),
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if int(args.world_size) < 1:
        raise ValueError("--world-size must be >= 1")

    setup = load_sim_setup(args.config, dt_us=args.dt_us)
    project_root, config_path = setup.project_root, setup.config_path
    rc, phys, dt_s = setup.rc, setup.phys, setup.dt_s
    float_type = select_float_type(rc)

    path_config_path = resolve_path(project_root, args.path_config, "")
    out_dir = (project_root / args.out_dir).resolve()
    if bool(args.path_complexity):
        pc_config_path = resolve_path(project_root, args.path_complexity_config, "configs/path_complexity.ini")
        pc_options = load_path_complexity_options(pc_config_path)
        args.path_complexity_report = True
        args.path_complexity_target_rel_l2 = float(pc_options.target_rel_l2)
        args.path_complexity_config_path = str(pc_config_path)
        print(f"path-complexity config: {pc_config_path}")

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
    path_complexity = runtime_plan.get("path_complexity")
    if path_complexity is not None:
        print(
            "[path-complexity] "
            f"A_path={int(path_complexity.get('A_path', 0))} "
            f"mean={float(path_complexity.get('mean_predecessors_within_radius', 0.0)):.2f} "
            f"segment_length_mm={float(path_complexity.get('segment_length_mm', 0.0)):.6g} "
            f"initial_level_K={float(path_complexity.get('initial_level_K', 0.0)):.6g}"
        )
        initial_calibration = path_complexity.get("initial_calibration", {})
        if initial_calibration:
            print(
                "[path-complexity] "
                f"initial_calibrated_rel_l2={float(initial_calibration.get('rel_l2', 0.0)):.6g} "
                f"estimated_A_times_rel_l2="
                f"{float(path_complexity.get('estimated_amplified_rel_l2', 0.0)):.6g}"
            )
        if "target_rel_l2" in path_complexity:
            initial_dag = path_complexity.get("initial_dag", {})
            final_dag = path_complexity.get("final_dag", {})
            if initial_dag:
                print(
                    "[path-complexity] "
                    f"initial_max_cut_depth={int(initial_dag.get('global_max_cut_depth', 0))} "
                    f"initial_edges={int(initial_dag.get('num_edges', 0))}"
                )
            print(
                "[path-complexity] "
                f"target_rel_l2={float(path_complexity['target_rel_l2']):.6g} "
                f"local_rel_l2={float(path_complexity['adjusted_local_rel_l2']):.6g} "
                f"adjusted_level_K={float(path_complexity['adjusted_level_K']):.6g} "
                f"adjusted_A_path={int(path_complexity.get('adjusted_A_path', 0))}"
            )
            if final_dag:
                print(
                    "[path-complexity] "
                    f"final_max_cut_depth={int(final_dag.get('global_max_cut_depth', 0))} "
                    f"final_edges={int(final_dag.get('num_edges', 0))}"
                )
    partition_seconds = runtime_plan["partition_summary"].get("partition_seconds")
    if partition_seconds is not None:
        print(
            "[partition] "
            f"mode={runtime_plan['partition_summary'].get('partition_mode', args.planner_mode)} "
            f"seconds={float(partition_seconds):.6f}"
        )
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
