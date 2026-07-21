"""
Single-GPU emulated multi-rank segment-correction run.

Builds the same runtime plan as the parallel run, but executes the planned
rank-local base/correction pipeline serially inside one process. This is useful
for debugging multi-rank runtime behavior when only one GPU is available.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

from hermes.physics.material import phys_parameter
from hermes.runtime.config import load_config
from hermes.scheduling.planning import build_runtime_plan
from hermes.scripts.outer_solver import build_outer_context
from emulated_runtime import run_emulated_parallel_tracer
from hermes.scripts.segment_correction.output import (
    build_component_start_snapshot_steps,
    build_global_stride_snapshot_steps,
    comp_start_step,
    save_parallel_snapshots,
)
from hermes.utils.mpi_utils import bind_local_gpu
from hermes.utils.path_utils import resolve_path


def _find_rank_boundary_component(
    *,
    rank_assignments: dict[int, list[int]],
    component_predecessors: dict[int, list[int]],
    source_rank: int,
    target_rank: int,
) -> tuple[int, int]:
    comp_to_rank = {
        int(comp_id): int(rank)
        for rank, comps in rank_assignments.items()
        for comp_id in comps
    }
    for comp_id in sorted(int(c) for c in rank_assignments.get(int(target_rank), [])):
        for pred_id in sorted(int(p) for p in component_predecessors.get(int(comp_id), [])):
            if int(comp_to_rank.get(int(pred_id), -1)) == int(source_rank):
                return int(pred_id), int(comp_id)
    raise ValueError(
        f"No same-layer correction boundary found from rank {int(source_rank)} "
        f"to rank {int(target_rank)}."
    )


def _build_rank_dependency_summary(
    *,
    rank_assignments: dict[int, list[int]],
    component_predecessors: dict[int, list[int]],
) -> dict[str, object]:
    comp_to_rank = {
        int(comp_id): int(rank)
        for rank, comps in rank_assignments.items()
        for comp_id in comps
    }
    rank_pred_sets: dict[int, set[int]] = {
        int(rank): set() for rank in rank_assignments
    }
    rank_succ_sets: dict[int, set[int]] = {
        int(rank): set() for rank in rank_assignments
    }
    rank_edges: list[dict[str, int]] = []
    seen_edges: set[tuple[int, int, int, int]] = set()

    for dst_comp, pred_comps in component_predecessors.items():
        dst_rank = comp_to_rank.get(int(dst_comp))
        if dst_rank is None:
            continue
        for src_comp in pred_comps:
            src_rank = comp_to_rank.get(int(src_comp))
            if src_rank is None:
                continue
            edge_key = (int(src_rank), int(dst_rank), int(src_comp), int(dst_comp))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            rank_edges.append(
                {
                    "src_rank": int(src_rank),
                    "dst_rank": int(dst_rank),
                    "src_component": int(src_comp),
                    "dst_component": int(dst_comp),
                }
            )
            if int(src_rank) == int(dst_rank):
                continue
            rank_pred_sets[int(dst_rank)].add(int(src_rank))
            rank_succ_sets[int(src_rank)].add(int(dst_rank))

    return {
        "rank_predecessors": {
            int(rank): sorted(int(pred_rank) for pred_rank in pred_ranks)
            for rank, pred_ranks in rank_pred_sets.items()
        },
        "rank_successors": {
            int(rank): sorted(int(succ_rank) for succ_rank in succ_ranks)
            for rank, succ_ranks in rank_succ_sets.items()
        },
        "rank_dependency_edges": sorted(
            rank_edges,
            key=lambda e: (
                int(e["dst_rank"]),
                int(e["src_rank"]),
                int(e["dst_component"]),
                int(e["src_component"]),
            ),
        ),
    }


def _component_layer_context(*, path_defs, component_id: int, ss_per_layer: int) -> dict[str, int]:
    acc_ss = 0
    layer_step = 0
    for pd in path_defs:
        if (int(acc_ss) % int(ss_per_layer)) == 0:
            layer_step = 0
        comp_id = int(pd.component_id)
        layer_idx = int(acc_ss) // int(ss_per_layer)
        total_steps = int(getattr(pd, "total_steps", int(pd.weight)))
        if comp_id == int(component_id):
            return {
                "layer": int(layer_idx),
                "start_ss": int(acc_ss),
                "end_ss": int(acc_ss) + int(pd.weight) - 1,
                "layer_start_step": int(layer_step),
                "layer_end_step": int(layer_step) + int(total_steps),
                "total_steps": int(total_steps),
                "weight_ss": int(pd.weight),
            }
        layer_step += int(total_steps)
        acc_ss += int(pd.weight)
    raise KeyError(f"component_id={int(component_id)} not found")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Single-GPU emulated segment-correction run")
    p.add_argument("--config", default="configs/examples/sim_ex1.ini", help="Base simulation config")
    p.add_argument("--path-config", required=True, help="DAG laser path config")
    p.add_argument("--out-dir", default="outputs/serial_emulated", help="Output directory")
    p.add_argument("--dt-us", type=float, help="Override dt in microseconds")
    p.add_argument("--snap-every-steps", type=int, default=100,
                   help="Save a snapshot every N timesteps (default: 100).")
    p.add_argument(
        "--component-start-snapshot-mode",
        action="store_true",
        help="Save snapshots near each runtime-component start instead of corrected-only/global stride.",
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
        help="Number of ranks to emulate during planning/runtime (default: 4).",
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
    p.add_argument(
        "--boundary-correction-snapshot-mode",
        action="store_true",
        help=(
            "Save only the source-off correction snapshots at the first same-layer "
            "boundary from --boundary-source-rank to --boundary-target-rank."
        ),
    )
    p.add_argument(
        "--boundary-source-rank",
        type=int,
        default=0,
        help="Source rank for --boundary-correction-snapshot-mode (default: 0).",
    )
    p.add_argument(
        "--boundary-target-rank",
        type=int,
        default=1,
        help="Target rank for --boundary-correction-snapshot-mode (default: 1).",
    )
    p.add_argument(
        "--rank-base-snapshot-mode",
        action="store_true",
        help="Save only source-on base snapshots for one emulated rank and exit early.",
    )
    p.add_argument(
        "--rank-base-snapshot-rank",
        type=int,
        default=0,
        help="Rank to export in --rank-base-snapshot-mode (default: 0).",
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

    print("=== Serial Emulated Segment-Correction Run ===")
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
    rank_assignments = {
        int(rank): [int(comp) for comp in comps]
        for rank, comps in runtime_plan["rank_assignments"].items()
    }
    steps_per_ss = int(runtime_plan["steps_per_ss"])
    num_layers = int(runtime_plan["num_layers"])
    ss_per_layer = int(runtime_plan["ss_per_layer"])
    correction_horizon_ss_map = runtime_plan["correction_horizon_ss_map"]
    correction_horizon_by_edge = runtime_plan.get("correction_horizon_by_edge", {})
    component_predecessors = {
        int(comp): [int(pred) for pred in preds]
        for comp, preds in runtime_plan["component_predecessors"].items()
    }
    component_successors = {
        int(comp): [int(succ) for succ in succs]
        for comp, succs in runtime_plan["component_successors"].items()
    }
    rank_dependency_summary = _build_rank_dependency_summary(
        rank_assignments=rank_assignments,
        component_predecessors=component_predecessors,
    )
    path_def_by_id = {int(pd.component_id): pd for pd in all_path_defs}

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"config:       {config_path}")
    print(f"path-config:  {path_config_path}")
    print(f"dt:           {dt_s:.6e} s  ({dt_s * 1e6:.6f} us)")
    print(f"solver mode:  {args.solver_mode}")
    print(f"planner mode: {args.planner_mode}")
    print(f"global max cut depth: {int(runtime_plan.get('global_max_cut_depth', 0))}")
    print(f"Components:   {len(all_path_defs)}  ({len(all_path_defs) // max(num_layers,1)}/layer)")
    print(f"Layers:       {num_layers}  ({ss_per_layer} SS/layer)")
    print(f"emulated ranks: {int(args.world_size)}")
    print("rank distribution:", rank_assignments)
    print("rank predecessors:", rank_dependency_summary["rank_predecessors"])
    if rank_dependency_summary["rank_dependency_edges"]:
        print("rank dependency edges:")
        for edge in rank_dependency_summary["rank_dependency_edges"]:
            if int(edge["src_rank"]) == int(edge["dst_rank"]):
                continue
            print(
                f"  rank {int(edge['dst_rank'])} <- rank {int(edge['src_rank'])} "
                f"(component {int(edge['dst_component'])} <- {int(edge['src_component'])})"
            )

    ambient_gpu = cp.full((n_all,), ctx.u0, dtype=float_type)
    h_m = float(rc.level3.h_tuple[0])
    snaps_dir = out_dir / "snapshots_emu"
    meta_dir = out_dir / "snapshots_emu_meta"
    snaps_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    snapshot_stride_steps = int(args.snap_every_steps)
    snapshot_steps_by_component = None
    boundary_export: dict[str, object] | None = None
    rank_base_export: dict[str, object] | None = None
    if bool(args.boundary_correction_snapshot_mode) and bool(args.rank_base_snapshot_mode):
        raise ValueError(
            "--boundary-correction-snapshot-mode and --rank-base-snapshot-mode "
            "are mutually exclusive."
        )
    if args.rank_base_snapshot_mode:
        rank_base = int(args.rank_base_snapshot_rank)
        base_component_ids = [
            int(comp_id) for comp_id in sorted(rank_assignments.get(int(rank_base), []))
        ]
        if not base_component_ids:
            raise ValueError(f"Rank {rank_base} has no assigned components.")
        snapshot_steps_by_component = build_global_stride_snapshot_steps(
            all_path_defs,
            ss_per_layer=ss_per_layer,
            steps_per_ss=steps_per_ss,
            snap_every_steps=snapshot_stride_steps,
        )
        rank_base_export = {
            "rank": int(rank_base),
            "component_ids": base_component_ids,
        }
        print(
            "snapshot mode: rank-base source-on only "
            f"rank {rank_base}, components {base_component_ids}, "
            f"every {snapshot_stride_steps} steps"
        )
    elif args.boundary_correction_snapshot_mode:
        src_comp_id, target_comp_id = _find_rank_boundary_component(
            rank_assignments=rank_assignments,
            component_predecessors=component_predecessors,
            source_rank=int(args.boundary_source_rank),
            target_rank=int(args.boundary_target_rank),
        )
        target_pd = path_def_by_id[int(target_comp_id)]
        horizon_ss = max(1, int(correction_horizon_ss_map.get(int(target_comp_id), 1)))
        horizon_steps = min(int(target_pd.total_steps), int(horizon_ss) * int(steps_per_ss))
        rel_steps = list(range(0, max(1, int(horizon_steps)), int(snapshot_stride_steps)))
        if not rel_steps:
            rel_steps = [0]
        target_context = _component_layer_context(
            path_defs=all_path_defs,
            component_id=int(target_comp_id),
            ss_per_layer=int(ss_per_layer),
        )
        boundary_export = {
            "source_rank": int(args.boundary_source_rank),
            "target_rank": int(args.boundary_target_rank),
            "source_component_id": int(src_comp_id),
            "target_component_id": int(target_comp_id),
            "target_context": target_context,
            "correction_horizon_ss": int(horizon_ss),
            "correction_horizon_steps": int(horizon_steps),
            "snapshot_relative_steps": [int(s) for s in rel_steps],
            "snapshot_within_layer_steps": [
                int(target_context["layer_start_step"]) + int(s) for s in rel_steps
            ],
        }
        snapshot_steps_by_component = {int(target_comp_id): rel_steps}
        print(
            "snapshot mode: rank-boundary source-off correction + target base "
            f"rank {args.boundary_source_rank}->{args.boundary_target_rank}, "
            f"component {src_comp_id}->{target_comp_id}, every {snapshot_stride_steps} steps"
        )
    elif args.component_start_snapshot_mode:
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

    t0 = time.perf_counter()
    correction_payloads_for_export: dict[int, list[np.ndarray]] = {}
    base_states_for_export: dict[int, list[np.ndarray]] = {}
    boundary_base_states_for_export: dict[int, list[np.ndarray]] = {}
    process_components = None
    if boundary_export is not None:
        process_components = {int(boundary_export["source_component_id"])}
    elif rank_base_export is not None:
        process_components = {int(c) for c in rank_base_export["component_ids"]}
    final_states_host, rank_timing_stats = run_emulated_parallel_tracer(
        ctx=ctx,
        ambient_gpu=ambient_gpu,
        path_defs=all_path_defs,
        rank_assignments=rank_assignments,
        steps_per_ss=steps_per_ss,
        ss_per_layer=ss_per_layer,
        snapshot_stride_steps=snapshot_stride_steps,
        snapshot_steps_by_component=snapshot_steps_by_component,
        correction_horizon_ss_map=correction_horizon_ss_map,
        correction_horizon_by_edge=correction_horizon_by_edge,
        component_predecessors=component_predecessors,
        component_successors=component_successors,
        h_m=h_m,
        deltaT_K=float(phys.deltaT),
        correction_payloads_out=(
            correction_payloads_for_export if boundary_export is not None else None
        ),
        capture_correction_components=(
            {int(boundary_export["target_component_id"])} if boundary_export is not None else None
        ),
        base_states_out=(base_states_for_export if rank_base_export is not None else None),
        process_components=process_components,
        skip_outgoing_corrections=rank_base_export is not None,
        stop_after_base_components=rank_base_export is not None,
        stop_after_captured_corrections=boundary_export is not None,
    )
    if boundary_export is not None:
        _, boundary_base_timing_stats = run_emulated_parallel_tracer(
            ctx=ctx,
            ambient_gpu=ambient_gpu,
            path_defs=all_path_defs,
            rank_assignments=rank_assignments,
            steps_per_ss=steps_per_ss,
            ss_per_layer=ss_per_layer,
            snapshot_stride_steps=snapshot_stride_steps,
            snapshot_steps_by_component=snapshot_steps_by_component,
            correction_horizon_ss_map=correction_horizon_ss_map,
            component_predecessors=component_predecessors,
            component_successors=component_successors,
            h_m=h_m,
            deltaT_K=float(phys.deltaT),
            base_states_out=boundary_base_states_for_export,
            process_components={int(boundary_export["target_component_id"])},
            skip_outgoing_corrections=True,
            stop_after_base_components=True,
        )
        target_rank = int(boundary_export["target_rank"])
        if target_rank in boundary_base_timing_stats:
            rank_timing_stats[target_rank] = dict(boundary_base_timing_stats[target_rank])
    total_elapsed = time.perf_counter() - t0

    start_step_map = comp_start_step(all_path_defs, steps_per_ss)
    if rank_base_export is not None:
        rank_base = int(rank_base_export["rank"])
        rank_base_snaps_dir = out_dir / "snapshots_rank_base"
        rank_base_meta_dir = out_dir / "snapshots_rank_base_meta"
        rank_base_snaps_dir.mkdir(parents=True, exist_ok=True)
        rank_base_meta_dir.mkdir(parents=True, exist_ok=True)
        rank_states = {
            int(comp_id): base_states_for_export[int(comp_id)]
            for comp_id in rank_base_export["component_ids"]
            if int(comp_id) in base_states_for_export
        }
        save_parallel_snapshots(
            rank=int(rank_base),
            snaps_dir=rank_base_snaps_dir,
            meta_dir=rank_base_meta_dir,
            final_states_host=rank_states,
            path_defs=all_path_defs,
            path_def_by_id=path_def_by_id,
            start_step_map=start_step_map,
            ss_per_layer=ss_per_layer,
            steps_per_ss=steps_per_ss,
            ctx=ctx,
            h_m=h_m,
            snapshot_steps_by_component=snapshot_steps_by_component,
            snap_every_steps=snapshot_stride_steps,
        )
        print(f"Saved rank-base snapshots to {rank_base_snaps_dir}")
    elif boundary_export is None:
        for rank in sorted(rank_assignments):
            rank_states = {
                int(comp_id): final_states_host[int(comp_id)]
                for comp_id in rank_assignments[int(rank)]
                if int(comp_id) in final_states_host
            }
            save_parallel_snapshots(
                rank=int(rank),
                snaps_dir=snaps_dir,
                meta_dir=meta_dir,
                final_states_host=rank_states,
                path_defs=all_path_defs,
                path_def_by_id=path_def_by_id,
                start_step_map=start_step_map,
                ss_per_layer=ss_per_layer,
                steps_per_ss=steps_per_ss,
                ctx=ctx,
                h_m=h_m,
                snapshot_steps_by_component=snapshot_steps_by_component,
                snap_every_steps=snapshot_stride_steps,
            )
    else:
        print("[snapshots_emu] skipped in boundary correction snapshot mode")

    boundary_meta_path = None
    if boundary_export is not None:
        target_comp_id = int(boundary_export["target_component_id"])
        target_context = dict(boundary_export["target_context"])
        source_off_dir = out_dir / "snapshots_source_off_T_K"
        source_off_dir.mkdir(parents=True, exist_ok=True)
        source_off_files = []
        rel_steps = [int(s) for s in boundary_export["snapshot_relative_steps"]]
        for idx, arr in enumerate(correction_payloads_for_export.get(target_comp_id, [])):
            if idx >= len(rel_steps):
                break
            within_layer_step = int(target_context["layer_start_step"]) + int(rel_steps[idx])
            fname = f"layer_{int(target_context['layer']):02d}_step_{within_layer_step:09d}.npy"
            np.save(
                source_off_dir / fname,
                np.asarray(arr, dtype=np.float64) * float(phys.deltaT) + float(phys.T0),
            )
            source_off_files.append(fname)
        target_base_dir = out_dir / "snapshots_target_base_T_K"
        target_base_dir.mkdir(parents=True, exist_ok=True)
        target_base_files = []
        for idx, arr in enumerate(boundary_base_states_for_export.get(target_comp_id, [])):
            if idx >= len(rel_steps):
                break
            within_layer_step = int(target_context["layer_start_step"]) + int(rel_steps[idx])
            fname = f"layer_{int(target_context['layer']):02d}_step_{within_layer_step:09d}.npy"
            np.save(
                target_base_dir / fname,
                np.asarray(arr, dtype=np.float64) * float(phys.deltaT) + float(phys.Ts),
            )
            target_base_files.append(fname)
        boundary_export["units"] = {
            "source_off_field": "T_K",
            "source_off_delta_definition": "source-off delta field was shifted by ambient temperature T0",
            "target_base_field": "T_K",
            "target_base_definition": "target-rank source-on base field without incoming correction",
            "reference_field": "T_K",
        }
        boundary_export["source_off_T_K_dir"] = str(source_off_dir)
        boundary_export["source_off_T_K_files"] = source_off_files
        boundary_export["target_base_T_K_dir"] = str(target_base_dir)
        boundary_export["target_base_T_K_files"] = target_base_files
        boundary_meta_path = out_dir / "boundary_visualization.json"
        with open(boundary_meta_path, "w", encoding="utf-8") as f:
            json.dump(boundary_export, f, indent=2)
        print(f"Saved source-off correction snapshots to {source_off_dir}")
        print(f"Saved target-rank base snapshots to {target_base_dir}")
        print(f"Saved boundary visualization metadata to {boundary_meta_path}")

    print(f"\n[timing] emulated total: {total_elapsed:.3f} s  ({num_layers} layer(s))")
    print("[timing] emulated rank breakdown:")
    for emu_rank in sorted(rank_timing_stats):
        stats = rank_timing_stats[int(emu_rank)]
        print(
            f"  rank {int(emu_rank):2d}: total={float(stats.get('rank_total_seconds', 0.0)):.3f}s "
            f"base={float(stats.get('base_solve_seconds', 0.0)):.3f}s "
            f"tracer={float(stats.get('tracer_solve_seconds', 0.0)):.3f}s "
            f"recv_wait={float(stats.get('recv_wait_seconds', 0.0)):.3f}s "
            f"send_wait={float(stats.get('send_wait_seconds', 0.0)):.3f}s "
            f"superpose={float(stats.get('local_superpose_seconds', 0.0)):.3f}s",
            flush=True,
        )
    if boundary_export is None and rank_base_export is None:
        print(f"Saved snapshots to {snaps_dir}")

    with open(out_dir / "emulated_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "emulated_total_seconds": total_elapsed,
                "planner_mode": str(args.planner_mode),
                "num_components": len(all_path_defs),
                "num_layers": num_layers,
                "ss_per_layer": ss_per_layer,
                "world_size": int(args.world_size),
                "rank_predecessors": {
                    str(int(k)): [int(vv) for vv in v]
                    for k, v in rank_dependency_summary["rank_predecessors"].items()
                },
                "rank_successors": {
                    str(int(k)): [int(vv) for vv in v]
                    for k, v in rank_dependency_summary["rank_successors"].items()
                },
                "rank_dependency_edges": list(rank_dependency_summary["rank_dependency_edges"]),
                "component_predecessors": {
                    str(int(k)): [int(vv) for vv in v]
                    for k, v in component_predecessors.items()
                },
                "component_successors": {
                    str(int(k)): [int(vv) for vv in v]
                    for k, v in component_successors.items()
                },
                "early_exit_after_boundary_snapshots": boundary_export is not None,
                "early_exit_after_rank_base_snapshots": rank_base_export is not None,
                "rank_base_snapshot_rank": (
                    None if rank_base_export is None else int(rank_base_export["rank"])
                ),
                "rank_base_snapshot_components": (
                    None
                    if rank_base_export is None
                    else [int(c) for c in rank_base_export["component_ids"]]
                ),
                "rank_base_snapshot_dir": (
                    None
                    if rank_base_export is None
                    else str(out_dir / "snapshots_rank_base")
                ),
                "boundary_visualization_json": None if boundary_meta_path is None else str(boundary_meta_path),
                "boundary_target_base_snapshot_dir": (
                    None
                    if boundary_export is None
                    else str(out_dir / "snapshots_target_base_T_K")
                ),
                "rank_timing_breakdown": {
                    str(int(emu_rank)): {
                        str(key): float(value)
                        for key, value in stats.items()
                    }
                    for emu_rank, stats in rank_timing_stats.items()
                },
            },
            f,
            indent=2,
        )
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
