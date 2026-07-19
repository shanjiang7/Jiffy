from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import cupy as cp

from hermes.utils.mpi_utils import mpi_context, bind_local_gpu
from hermes.utils.path_utils import resolve_path
from hermes.physics.material import phys_parameter
from hermes.runtime.config import load_config
from hermes.scheduling.planning import (
    build_planning_summary,
    build_runtime_plan,
    compute_dt_s,
    print_run_summary,
    print_split_records,
)
from hermes.scripts.outer_solver import build_outer_context
from hermes.scripts.segment_correction.compare_runs import compare_snapshot_dirs
from hermes.scripts.segment_correction.core import run_parallel_tracer
from hermes.scripts.segment_correction.diagnostic_config import (
    load_diagnostic_check_options,
    load_path_complexity_options,
)
from hermes.scripts.segment_correction.output import (
    build_component_start_snapshot_steps,
    build_global_stride_snapshot_steps,
    comp_start_step,
    save_parallel_snapshots,
)


def _rank_device_info(rank: int) -> str:
    local_rank = int(
        os.environ.get(
            "SLURM_LOCALID",
            os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("PMI_LOCAL_RANK", "0")),
        )
    )
    host = socket.gethostname()
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    dev_idx = int(cp.cuda.runtime.getDevice())
    props = cp.cuda.runtime.getDeviceProperties(dev_idx)
    gpu_name = props["name"].decode("utf-8", errors="ignore") if isinstance(props["name"], (bytes, bytearray)) else str(props["name"])
    gpu_uuid = "unknown"
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid",
                "--format=csv,noheader",
                "--id",
                str(dev_idx),
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            gpu_uuid = out.splitlines()[0].strip()
    except Exception:
        pass
    return (
        f"[rank {rank}] host={host} local_rank={local_rank} "
        f"cuda_visible={cuda_visible} device={dev_idx} gpu=\"{gpu_name}\" uuid={gpu_uuid}"
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Component-Level Parallel Segment Correction")
    p.add_argument("--config", default="configs/examples/sim_ex1.ini", help="Base simulation config")
    p.add_argument("--path-config", required=True, help="DAG laser path config")
    p.add_argument("--out-dir", default="outputs/segment_correction", help="Output directory")
    p.add_argument("--dt-us", type=float, help="Override dt in microseconds")
    p.add_argument(
        "--no-export-dag",
        action="store_true",
        help="Skip exporting DAG lookup/edge/component CSVs and DAG plots during planning.",
    )
    p.add_argument("--snap-every-steps", type=int, default=100,
                   help="Save a snapshot every N timesteps (default: 100).")
    p.add_argument(
        "--timing-only",
        action="store_true",
        help="Run the base/correction pipeline for timing only and skip saving final snapshots.",
    )
    p.add_argument(
        "--component-start-snapshot-mode",
        action="store_true",
        help="Save snapshots near each runtime-component start instead of using a uniform global stride.",
    )
    p.add_argument(
        "--component-start-snapshot-interval-steps",
        type=int,
        default=100,
        help="Stride inside each runtime-component snapshot window (default: 100).",
    )
    p.add_argument(
        "--component-start-snapshot-count",
        type=int,
        default=10,
        help="Maximum number of snapshots to save from each runtime-component start window (default: 10).",
    )
    p.add_argument(
        "--solver-mode",
        choices=("fused", "legacy"),
        default="fused",
        help="Level-3 outer solver mode (default: fused).",
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
        "--diagnostic-check",
        action="store_true",
        help="Run an additional buffered DAG pass and compare snapshots.",
    )
    p.add_argument(
        "--diagnostic-config",
        default="configs/diagnostic_check.ini",
        help="INI file with [diagnostic_check] settings.",
    )
    return p.parse_args(argv)


def _run_parallel_pass(
    *,
    pass_name: str,
    args,
    comm,
    rank: int,
    world_size: int,
    config_path: Path,
    path_config_path: Path,
    out_dir: Path,
    rc,
    phys,
    float_type,
    dt_s: float,
    ctx,
    n_all: int,
):
    if rank == 0:
        print(f"=== Segment Correction Pass: {pass_name} ===")
        print("Building DAG Components (multi-layer via pipeline)...")
        search_summary = None
        runtime_plan = build_runtime_plan(
            args=args,
            world_size=world_size,
            path_config_path=path_config_path,
            out_dir=out_dir,
            dt_s=dt_s,
            rc=rc,
            phys=phys,
            float_type=float_type,
            solver_velocity_mps=rc.laser.v,
            export_outputs=not bool(args.no_export_dag),
        )
        if runtime_plan is None:
            print("[done] no components to process.")
            comm.bcast(None, root=0)
            return None

        print_split_records(
            runtime_plan["split_records"],
            runtime_plan["runtime_components"],
            runtime_plan["source_components"],
        )
        bcast_data = (
            runtime_plan["path_defs"],
            runtime_plan["rank_assignments"],
            runtime_plan["steps_per_ss"],
            runtime_plan["num_layers"],
            runtime_plan["ss_per_layer"],
            runtime_plan["rank_pred_loads"],
            runtime_plan["correction_horizon_ss_map"],
            runtime_plan["component_predecessors"],
            runtime_plan["component_successors"],
            int(runtime_plan.get("global_max_cut_depth", 0)),
            float(runtime_plan.get("dependency_level_K", 0.0)),
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        planning_summary = build_planning_summary(
            runtime_plan=runtime_plan,
            args=args,
            world_size=world_size,
            search_summary=search_summary,
        )
        planning_summary["pass_name"] = str(pass_name)
        with open(out_dir / "planning_summary.json", "w", encoding="utf-8") as f:
            json.dump(planning_summary, f, indent=2)
        print(f"Saved planning summary to {out_dir / 'planning_summary.json'}")
    else:
        bcast_data = None

    bcast_data = comm.bcast(bcast_data, root=0)
    if bcast_data is None:
        return None
    (
        all_path_defs,
        rank_assignments,
        steps_per_ss,
        num_layers,
        ss_per_layer,
        rank_pred_loads,
        correction_horizon_ss_map,
        component_predecessors,
        component_successors,
        global_max_cut_depth,
        dependency_level_K,
    ) = bcast_data

    snap_every_steps = int(args.snap_every_steps)
    if args.component_start_snapshot_mode:
        snapshot_steps_by_component = build_component_start_snapshot_steps(
            all_path_defs,
            interval_steps=int(args.component_start_snapshot_interval_steps),
            max_snapshots_per_component=int(args.component_start_snapshot_count),
        )
    else:
        snapshot_steps_by_component = build_global_stride_snapshot_steps(
            all_path_defs,
            ss_per_layer=ss_per_layer,
            steps_per_ss=steps_per_ss,
            snap_every_steps=snap_every_steps,
        )

    assigned_comps = rank_assignments.get(rank, [])
    start_step_map = comp_start_step(all_path_defs, steps_per_ss)
    path_def_by_id = {int(pd.component_id): pd for pd in all_path_defs}

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print_run_summary(
            args=args,
            config_path=config_path,
            path_config_path=path_config_path,
            dt_s=dt_s,
            phys=phys,
            rc=rc,
            num_layers=num_layers,
            ss_per_layer=ss_per_layer,
            path_defs=all_path_defs,
            rank_assignments=rank_assignments,
            rank_pred_loads=rank_pred_loads,
            global_max_cut_depth=int(global_max_cut_depth),
        )
        print(f"dependency level_K: {float(dependency_level_K):.6g} K")
    comm.Barrier()

    ambient_gpu = cp.full((n_all,), ctx.u0, dtype=float_type)
    comm.Barrier()
    par_t0 = time.perf_counter()

    final_states_host, rank_timing_stats = run_parallel_tracer(
        ctx=ctx,
        ambient_gpu=ambient_gpu,
        path_defs=all_path_defs,
        assigned_comps=assigned_comps,
        comm=comm,
        rank=rank,
        world_size=world_size,
        rank_assignments=rank_assignments,
        steps_per_ss=steps_per_ss,
        ss_per_layer=ss_per_layer,
        snapshot_stride_steps=snap_every_steps,
        snapshot_steps_by_component=snapshot_steps_by_component,
        correction_horizon_ss_map=correction_horizon_ss_map,
        component_predecessors=component_predecessors,
        component_successors=component_successors,
        deltaT_K=float(phys.deltaT),
        collect_output_snapshots=not bool(args.timing_only),
        h_m=float(rc.level3.h_tuple[0]),
    )

    par_total_s = time.perf_counter() - par_t0
    gathered = comm.gather(par_total_s, root=0)
    gathered_rank_timing = comm.gather(rank_timing_stats, root=0)
    par_dt = max(gathered) if rank == 0 else 0.0

    if not args.timing_only:
        snaps_dir = out_dir / "snapshots_par"
        meta_dir = out_dir / "snapshots_par_meta"
        # Every rank creates the dirs (exist_ok): rank-0-only mkdir + barrier is
        # not enough on NFS, where other nodes' attribute caches may not see the
        # new directory yet and np.save fails with FileNotFoundError.
        snaps_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

        h_m = float(rc.level3.h_tuple[0])
        save_parallel_snapshots(
            rank=rank,
            snaps_dir=snaps_dir,
            meta_dir=meta_dir,
            final_states_host=final_states_host,
            path_defs=all_path_defs,
            path_def_by_id=path_def_by_id,
            start_step_map=start_step_map,
            ss_per_layer=ss_per_layer,
            steps_per_ss=steps_per_ss,
            ctx=ctx,
            h_m=h_m,
            snapshot_steps_by_component=snapshot_steps_by_component,
            snap_every_steps=snap_every_steps,
        )
        comm.Barrier()

    if rank == 0:
        if args.timing_only:
            print("[snapshots] skipped (--timing-only)")
        elif args.component_start_snapshot_mode:
            print(
                f"[snapshots] saved to {snaps_dir}  "
                f"(component-start mode: every {args.component_start_snapshot_interval_steps} steps,"
                f" up to {args.component_start_snapshot_count} per component, 2 mm ROI)"
            )
        else:
            print(f"[snapshots] saved to {snaps_dir}  (global every {snap_every_steps} steps, 2 mm ROI)")
        print(f"[timing] {pass_name} total: {par_dt:.3f} s  ({num_layers} layer(s))")
        print("[timing] rank breakdown:")
        rank_timing_summary = {}
        for rank_idx, rank_total_s, timing_stats in zip(
            range(world_size),
            gathered,
            gathered_rank_timing,
        ):
            merged_stats = {
                "rank_total_seconds": float(rank_total_s),
                "base_solve_seconds": float(timing_stats.get("base_solve_seconds", 0.0)),
                "tracer_solve_seconds": float(timing_stats.get("tracer_solve_seconds", 0.0)),
                "recv_wait_seconds": float(timing_stats.get("recv_wait_seconds", 0.0)),
                "send_wait_seconds": float(timing_stats.get("send_wait_seconds", 0.0)),
                "local_superpose_seconds": float(timing_stats.get("local_superpose_seconds", 0.0)),
                "pipeline_loop_seconds": float(timing_stats.get("pipeline_loop_seconds", 0.0)),
                "post_pipeline_seconds": float(timing_stats.get("post_pipeline_seconds", 0.0)),
                "num_assigned_components": float(timing_stats.get("num_assigned_components", 0.0)),
                "num_remote_sends": float(timing_stats.get("num_remote_sends", 0.0)),
                "num_remote_recvs": float(timing_stats.get("num_remote_recvs", 0.0)),
                "num_local_corrections": float(timing_stats.get("num_local_corrections", 0.0)),
                "num_correction_edges": float(timing_stats.get("num_correction_edges", 0.0)),
                "max_component_predecessors": float(timing_stats.get("max_component_predecessors", 0.0)),
            }
            rank_timing_summary[str(rank_idx)] = merged_stats
            print(
                f"  rank {rank_idx:2d}: total={merged_stats['rank_total_seconds']:.3f}s "
                f"base={merged_stats['base_solve_seconds']:.3f}s "
                f"tracer={merged_stats['tracer_solve_seconds']:.3f}s "
                f"recv_wait={merged_stats['recv_wait_seconds']:.3f}s "
                f"send_wait={merged_stats['send_wait_seconds']:.3f}s "
                f"superpose={merged_stats['local_superpose_seconds']:.3f}s",
                flush=True,
            )
        with open(out_dir / "timing_summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "pass_name": str(pass_name),
                "config": str(config_path),
                "num_components": len(all_path_defs),
                "num_layers": int(num_layers),
                "ss_per_layer": int(ss_per_layer),
                "num_ranks": int(world_size),
                "dependency_level_K": float(dependency_level_K),
                "parallel_total_seconds": float(par_dt),
                "component_predecessors": {
                    str(int(k)): [int(vv) for vv in v]
                    for k, v in component_predecessors.items()
                },
                "component_successors": {
                    str(int(k)): [int(vv) for vv in v]
                    for k, v in component_successors.items()
                },
                "rank_timing_breakdown": rank_timing_summary,
            }, f, indent=2)
        print(f"Saved timing to {out_dir / 'timing_summary.json'}")

    del final_states_host
    del ambient_gpu
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    comm.Barrier()
    return {
        "out_dir": str(out_dir),
        "dependency_level_K": float(dependency_level_K),
        "parallel_total_seconds": float(par_dt),
    }



def main(argv=None):
    comm, rank, world_size = mpi_context()
    bind_local_gpu()

    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[4]

    config_path = resolve_path(project_root, args.config, "configs/examples/sim_ex1.ini")
    path_config_path = resolve_path(project_root, args.path_config, "")
    out_dir = (project_root / args.out_dir).resolve()

    args.path_complexity_report = False
    args.path_complexity_target_rel_l2 = None
    args.dependency_level_K_override = None
    if bool(args.path_complexity):
        pc_config_path = resolve_path(project_root, args.path_complexity_config, "configs/path_complexity.ini")
        pc_options = load_path_complexity_options(pc_config_path)
        args.path_complexity_report = True
        args.path_complexity_target_rel_l2 = float(pc_options.target_rel_l2)
        args.path_complexity_config_path = str(pc_config_path)
        if rank == 0:
            print(f"path-complexity config: {pc_config_path}")

    diagnostic_options = None
    if bool(args.diagnostic_check):
        if bool(args.timing_only):
            raise ValueError("--diagnostic-check requires snapshots and cannot be used with --timing-only.")
        if bool(args.component_start_snapshot_mode):
            raise ValueError("--diagnostic-check requires global stride snapshots; remove --component-start-snapshot-mode.")
        diag_config_path = resolve_path(project_root, args.diagnostic_config, "configs/diagnostic_check.ini")
        diagnostic_options = load_diagnostic_check_options(diag_config_path)
        args.diagnostic_config_path = str(diag_config_path)
        args.snap_every_steps = int(diagnostic_options.snap_every_steps)
        if rank == 0:
            print(f"diagnostic config: {diag_config_path}")
            print(
                "diagnostic settings: "
                f"gamma={diagnostic_options.gamma:.6g}, "
                f"tol={diagnostic_options.tol:.6g}, "
                f"snap_every_steps={diagnostic_options.snap_every_steps}"
            )

    rc = load_config(config_path)
    print(_rank_device_info(rank), flush=True)
    float_type = cp.float64 if rc.float_type_str.lower() == "float64" else cp.float32

    mat_override = rc.material.to_override_dict()
    t_spot_on = 2.0 * rc.laser.x_span_m / rc.laser.v
    phys = phys_parameter(rc.laser.Q, rc.laser.x_span_m, t_spot_on, mat_ch=mat_override)
    dt_s = compute_dt_s(args, rc, phys)

    dt_nd = dt_s / phys.time_scale
    ctx = build_outer_context(rc, phys, float_type, dt_nd, solver_mode=args.solver_mode)
    n_all = ctx.nx * ctx.ny * ctx.nz
    production_out_dir = out_dir / "diagnostic_normal" if diagnostic_options is not None else out_dir
    production_result = _run_parallel_pass(
        pass_name="production",
        args=args,
        comm=comm,
        rank=rank,
        world_size=world_size,
        config_path=config_path,
        path_config_path=path_config_path,
        out_dir=production_out_dir,
        rc=rc,
        phys=phys,
        float_type=float_type,
        dt_s=dt_s,
        ctx=ctx,
        n_all=n_all,
    )
    if production_result is None:
        sys.exit(0)

    if diagnostic_options is None:
        return

    production_level_K = float(production_result["dependency_level_K"])
    buffer_level_K = float(production_level_K) / float(diagnostic_options.gamma)
    buffer_args = argparse.Namespace(**vars(args))
    buffer_args.path_complexity = False
    buffer_args.path_complexity_report = False
    buffer_args.path_complexity_target_rel_l2 = None
    buffer_args.dependency_level_K_override = float(buffer_level_K)

    if rank == 0:
        print(
            "=== Diagnostic buffered validation ===\n"
            f"production level_K={production_level_K:.6g} K, "
            f"buffer level_K={buffer_level_K:.6g} K"
        )
    buffer_out_dir = out_dir / "diagnostic_buffer"
    buffer_result = _run_parallel_pass(
        pass_name="diagnostic_buffer",
        args=buffer_args,
        comm=comm,
        rank=rank,
        world_size=world_size,
        config_path=config_path,
        path_config_path=path_config_path,
        out_dir=buffer_out_dir,
        rc=rc,
        phys=phys,
        float_type=float_type,
        dt_s=dt_s,
        ctx=ctx,
        n_all=n_all,
    )
    if buffer_result is None:
        sys.exit(0)

    if rank == 0:
        compare_dir = out_dir / "diagnostic_compare"
        comparison = compare_snapshot_dirs(
            test_snap_dir=buffer_out_dir / "snapshots_par",
            reference_snap_dir=production_out_dir / "snapshots_par",
            out_dir=compare_dir,
            tol=float(diagnostic_options.tol),
            test_label="buffer",
            reference_label="production",
        )
        diagnostic_summary = {
            "production_dir": str(production_out_dir),
            "buffer_dir": str(buffer_out_dir),
            "compare_dir": str(compare_dir),
            "production_level_K": float(production_level_K),
            "buffer_level_K": float(buffer_level_K),
            "gamma": float(diagnostic_options.gamma),
            "tol": float(diagnostic_options.tol),
            "snap_every_steps": int(diagnostic_options.snap_every_steps),
            "max_eta": float(comparison["max_rel_l2"]),
            "mean_eta": float(comparison["mean_rel_l2"]),
            "num_compared": int(comparison["num_compared"]),
            "passed": bool(comparison.get("passed", False)),
            "production_total_seconds": float(production_result["parallel_total_seconds"]),
            "buffer_total_seconds": float(buffer_result["parallel_total_seconds"]),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "diagnostic_summary.json", "w", encoding="utf-8") as f:
            json.dump(diagnostic_summary, f, indent=2)
        status = "PASS" if diagnostic_summary["passed"] else "FAIL"
        print(
            f"[diagnostic] {status}: max_eta={diagnostic_summary['max_eta']:.4e}, "
            f"tol={diagnostic_summary['tol']:.4e}"
        )
        print(f"Saved diagnostic summary to {out_dir / 'diagnostic_summary.json'}")
    comm.Barrier()

if __name__ == "__main__":
    main()
