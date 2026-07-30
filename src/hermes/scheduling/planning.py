from __future__ import annotations

from pathlib import Path

from hermes.scheduling._grouping import CORRECTION_FIXED_COST_SS
from hermes.scheduling._partitioned import build_partitioned_runtime_plan
from hermes.runtime.setup import compute_dt_s as _compute_dt_s


def compute_dt_s(args, rc, phys) -> float:
    return _compute_dt_s(rc, phys, dt_us=args.dt_us)


def predicted_skew(rank_pred_loads) -> float:
    loads = [float(rank_pred_loads[r]) for r in sorted(rank_pred_loads)]
    mean_load = sum(loads) / max(len(loads), 1)
    max_load = max(loads) if loads else 0.0
    return (max_load / mean_load) if mean_load > 0 else 0.0


def build_planning_summary(
    *,
    runtime_plan,
    args,
    world_size: int,
    search_summary=None,
) -> dict:
    summary = {
        "world_size": int(world_size),
        "planner_mode": str(runtime_plan["planner_mode"]),
        "global_max_cut_depth": int(runtime_plan.get("global_max_cut_depth", 0)),
        "correction_weight": float(runtime_plan["correction_weight"]),
        "correction_fixed_cost_ss": float(CORRECTION_FIXED_COST_SS),
        "auto_select_ss_length": False,
        "segments_per_supersegment": int(runtime_plan["segments_per_supersegment"]),
        "num_source_components": int(len(runtime_plan["source_components"])),
        "num_runtime_components": int(len(runtime_plan["runtime_components"])),
        "split_records": runtime_plan["split_records"],
        "rank_assignments": runtime_plan["rank_assignments"],
        "rank_pred_loads": {int(k): float(v) for k, v in runtime_plan["rank_pred_loads"].items()},
        "predicted_skew": float(predicted_skew(runtime_plan["rank_pred_loads"])),
        "steps_per_ss": int(runtime_plan["steps_per_ss"]),
        "dependency_level_K": float(runtime_plan.get("dependency_level_K", 0.0)),
        "num_layers": int(runtime_plan["num_layers"]),
        "ss_per_layer": int(runtime_plan["ss_per_layer"]),
        "component_predecessors": {
            int(k): [int(vv) for vv in v]
            for k, v in runtime_plan.get("component_predecessors", {}).items()
        },
        "component_successors": {
            int(k): [int(vv) for vv in v]
            for k, v in runtime_plan.get("component_successors", {}).items()
        },
        "component_dependency_edges": list(runtime_plan.get("component_dependency_edges", [])),
        "correction_horizon_ss_map": {
            int(k): int(v)
            for k, v in runtime_plan.get("correction_horizon_ss_map", {}).items()
        },
        "correction_horizon_by_edge": {
            f"{int(src)}->{int(dst)}": int(v)
            for (src, dst), v in runtime_plan.get("correction_horizon_by_edge", {}).items()
        },
        "partition": runtime_plan["partition_summary"],
    }
    if search_summary is not None:
        summary["auto_search"] = search_summary
    if getattr(args, "dependency_level_K_override", None) is not None:
        summary["dependency_level_K_override"] = float(args.dependency_level_K_override)
    if runtime_plan.get("path_complexity") is not None:
        summary["path_complexity"] = runtime_plan["path_complexity"]
    return summary


def build_runtime_plan(
    *,
    args,
    world_size: int,
    path_config_path: Path,
    out_dir: Path,
    dt_s: float,
    rc,
    phys,
    float_type,
    solver_velocity_mps: float,
    export_outputs: bool = True,
):
    return build_partitioned_runtime_plan(
        planner_mode=str(args.planner_mode),
        correction_weight=float(args.correction_weight),
        solver_mode="fused",
        world_size=int(world_size),
        path_config_path=path_config_path,
        out_dir=out_dir,
        num_layers_override=(None if not hasattr(args, "num_layers") else args.num_layers),
        dt_s=dt_s,
        rc=rc,
        phys=phys,
        float_type=float_type,
        solver_velocity_mps=float(solver_velocity_mps),
        export_outputs=bool(export_outputs),
        verify_dp_monotonicity=bool(getattr(args, "verify_dp_monotonicity", False)),
        path_complexity_report=bool(getattr(args, "path_complexity_report", False)),
        dependency_level_K_override=getattr(args, "dependency_level_K_override", None),
        self_check_gamma=getattr(args, "self_check_gamma_effective", None),
        self_check_iterations=int(getattr(args, "self_check_iters", 1) or 1),
        self_check_mode=str(getattr(args, "self_check_mode", "horizon")),
        self_check_horizon_step_ss=int(getattr(args, "self_check_horizon_step", 2) or 2),
    )


def print_split_records(split_records, runtime_components, source_components) -> None:
    _ = runtime_components, source_components
    if not split_records:
        return
    print(f"[split-components] accepted {len(split_records)} split(s)")


def print_run_summary(
    *,
    args,
    config_path: Path,
    path_config_path: Path,
    dt_s: float,
    phys,
    rc,
    num_layers: int,
    ss_per_layer: int,
    path_defs,
    rank_assignments,
    rank_pred_loads,
    global_max_cut_depth: int | None = None,
) -> None:
    n_comps_per_layer = len(path_defs) // max(num_layers, 1)
    dx_step = rc.laser.v * dt_s / phys.len_scale
    print(f"config: {config_path}")
    print(f"path-config: {path_config_path}")
    print(f"dt: {dt_s:.6e} s ({dt_s * 1e6:.6f} us)")
    print(f"planner mode: {args.planner_mode}")
    if global_max_cut_depth is not None:
        print(f"global max cut depth: {int(global_max_cut_depth)}")
    print(f"correction weight: {args.correction_weight:.2f}")
    if getattr(args, "snap_every_steps", None) is not None:
        print(f"snapshot mode: every {args.snap_every_steps} steps")
    print(
        f"phys.len_scale: {phys.len_scale:.4e} m   "
        f"dx_step = {dx_step:.4e} (HERMES ND = {rc.laser.v * dt_s * 1e6:.2f} um/step)"
    )
    print(
        f"Components/layer: {n_comps_per_layer},  Layers: {num_layers},  "
        f"SS/layer: {ss_per_layer},  Total comps: {len(path_defs)},  Ranks: {len(rank_assignments)}"
    )
    print("Rank distribution:", rank_assignments)
    loads = [float(rank_pred_loads[r]) for r in sorted(rank_pred_loads)]
    mean_load = sum(loads) / max(len(loads), 1)
    max_load = max(loads) if loads else 0.0
    skew = (max_load / mean_load) if mean_load > 0 else 0.0
    print("Rank predicted loads:", {r: round(float(rank_pred_loads[r]), 2) for r in sorted(rank_pred_loads)})
    print(f"Predicted load skew (max/mean): {skew:.3f}")


__all__ = [
    "build_planning_summary",
    "build_runtime_plan",
    "compute_dt_s",
    "predicted_skew",
    "print_run_summary",
    "print_split_records",
]
