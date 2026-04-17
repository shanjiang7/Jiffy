from __future__ import annotations

import time
from typing import Dict, List

import cupy as cp
import numpy as np

from hermes.motion.types import PathDef
from hermes.scripts.outer_solver import OuterContext, run_ss_outer
from hermes.scripts.segment_correction.core import (
    _build_predecessor_map,
    _build_successor_map,
    _correction_stats,
    _snapshot_steps_for_component,
    _superpose_snapshots,
)
from hermes.utils.snapshot_utils import crop_snapshot


def _append_host_snapshot(
    out: List[np.ndarray],
    snap_u: cp.ndarray,
    *,
    nx: int,
    ny: int,
    nz: int,
    h_m: float | None,
) -> None:
    if h_m is None:
        out.append(cp.asnumpy(snap_u))
    else:
        out.append(crop_snapshot(cp.asnumpy(snap_u), nx, ny, nz, h_m))


def _append_host_delta_snapshot(
    out: List[np.ndarray],
    snap_u: cp.ndarray,
    *,
    ambient_gpu: cp.ndarray,
    nx: int,
    ny: int,
    nz: int,
    h_m: float | None,
) -> None:
    if h_m is None:
        out.append(cp.asnumpy(snap_u - ambient_gpu))
    else:
        out.append(crop_snapshot(cp.asnumpy(snap_u - ambient_gpu), nx, ny, nz, h_m))


def _release_cupy_temporaries() -> None:
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def run_emulated_parallel_tracer(
    ctx: OuterContext,
    ambient_gpu: cp.ndarray,
    path_defs: List[PathDef],
    rank_assignments: Dict[int, List[int]],
    steps_per_ss: int,
    ss_per_layer: int,
    snapshot_stride_steps: int | None = None,
    snapshot_steps_by_component: Dict[int, List[int]] | None = None,
    correction_horizon_ss_map: Dict[int, int] | None = None,
    h_m: float | None = None,
    deltaT_K: float = 1.0,
    correction_payloads_out: Dict[int, List[np.ndarray]] | None = None,
    capture_correction_components: set[int] | None = None,
    base_states_out: Dict[int, List[np.ndarray]] | None = None,
    process_components: set[int] | None = None,
    skip_outgoing_corrections: bool = False,
    stop_after_base_components: bool = False,
    stop_after_captured_corrections: bool = False,
) -> tuple[Dict[int, List[np.ndarray]], dict[int, dict[str, float]]]:
    """
    Single-process emulator of the source-push segment-correction runtime.

    This follows the same logical execution model as the MPI path but replaces
    communication with in-memory payload handoff so multi-rank behavior can be
    debugged on a single GPU.
    """
    pred_map = _build_predecessor_map(path_defs, ss_per_layer)
    succ_map = _build_successor_map(pred_map)
    path_def_by_id = {int(pd.component_id): pd for pd in path_defs}
    comp_to_rank = {
        int(comp_id): int(owner_rank)
        for owner_rank, comps in rank_assignments.items()
        for comp_id in comps
    }
    capture_correction_components_norm = (
        None
        if capture_correction_components is None
        else {int(comp_id) for comp_id in capture_correction_components}
    )
    process_components_norm = (
        None if process_components is None else {int(comp_id) for comp_id in process_components}
    )

    base_states_host: Dict[int, List[np.ndarray]] = {}
    final_states_host: Dict[int, List[np.ndarray]] = {}
    correction_payloads: Dict[int, List[np.ndarray]] = {}
    processed_base_components: set[int] = set()
    rank_timing_stats: dict[int, dict[str, float]] = {
        int(rank): {
            "rank_total_seconds": 0.0,
            "base_solve_seconds": 0.0,
            "tracer_solve_seconds": 0.0,
            "recv_wait_seconds": 0.0,
            "send_wait_seconds": 0.0,
            "local_superpose_seconds": 0.0,
            "pipeline_loop_seconds": 0.0,
            "post_pipeline_seconds": 0.0,
            "num_assigned_components": float(len(rank_assignments.get(int(rank), []))),
            "num_remote_sends": 0.0,
            "num_remote_recvs": 0.0,
            "num_local_corrections": 0.0,
        }
        for rank in rank_assignments
    }

    for rank in sorted(int(r) for r in rank_assignments):
        assigned_comps = sorted(int(j) for j in rank_assignments.get(int(rank), []))
        if not assigned_comps:
            continue
        print(f"[emulated rank {rank}] Base + outgoing correction pipeline.")
        loop_t0 = time.perf_counter()
        for j in assigned_comps:
            if process_components_norm is not None and int(j) not in process_components_norm:
                continue
            pd = path_def_by_id[int(j)]
            comp_snapshot_steps = _snapshot_steps_for_component(int(j), snapshot_steps_by_component)
            base_snaps_host: List[np.ndarray] = []
            base_t0 = time.perf_counter()
            _, final_u = run_ss_outer(
                ctx, ambient_gpu, pd.x_start, pd.y_start, pd.legs, steps_per_ss,
                source_on=True,
                snapshot_stride_steps=snapshot_stride_steps,
                snapshot_steps=comp_snapshot_steps,
                snapshot_callback=(
                    lambda snap_u,
                    out=base_snaps_host,
                    nx=int(ctx.nx),
                    ny=int(ctx.ny),
                    nz=int(ctx.nz),
                    h_m=h_m: _append_host_snapshot(
                        out,
                        snap_u,
                        nx=nx,
                        ny=ny,
                        nz=nz,
                        h_m=h_m,
                    )
                ),
            )
            cp.cuda.Stream.null.synchronize()
            rank_timing_stats[int(rank)]["base_solve_seconds"] += time.perf_counter() - base_t0
            base_states_host[int(j)] = base_snaps_host
            processed_base_components.add(int(j))
            if base_states_out is not None:
                base_states_out[int(j)] = list(base_snaps_host)

            if bool(stop_after_base_components) and (
                process_components_norm is None
                or process_components_norm.issubset(processed_base_components)
            ):
                del final_u
                _release_cupy_temporaries()
                rank_timing_stats[int(rank)]["pipeline_loop_seconds"] = time.perf_counter() - loop_t0
                rank_timing_stats[int(rank)]["rank_total_seconds"] = (
                    rank_timing_stats[int(rank)]["pipeline_loop_seconds"]
                )
                print(
                    f"[emulated rank {rank}] Early exit after base snapshots in "
                    f"{rank_timing_stats[int(rank)]['pipeline_loop_seconds']:.3f} s"
                )
                print("[emulated] Early exit after requested base snapshots.")
                return final_states_host, rank_timing_stats

            if bool(skip_outgoing_corrections):
                del final_u
                _release_cupy_temporaries()
                continue

            succ_j = succ_map.get(int(j))
            if succ_j is None:
                del final_u
                _release_cupy_temporaries()
                continue
            succ_pd = path_def_by_id[int(succ_j)]
            succ_snapshot_steps = _snapshot_steps_for_component(int(succ_j), snapshot_steps_by_component)
            horizon_ss = 1
            if correction_horizon_ss_map is not None:
                horizon_ss = max(1, int(correction_horizon_ss_map.get(int(succ_j), 1)))
            horizon_ss = min(horizon_ss, int(succ_pd.weight))
            max_tracer_steps = int(steps_per_ss) * int(horizon_ss)
            delta_snaps_host: List[np.ndarray] = []
            tracer_t0 = time.perf_counter()
            _, _ = run_ss_outer(
                ctx, final_u, succ_pd.x_start, succ_pd.y_start, succ_pd.legs, steps_per_ss,
                source_on=False,
                max_steps=max_tracer_steps,
                snapshot_stride_steps=snapshot_stride_steps,
                snapshot_steps=succ_snapshot_steps,
                snapshot_callback=(
                    lambda snap_u,
                    out=delta_snaps_host,
                    ambient_gpu=ambient_gpu,
                    nx=int(ctx.nx),
                    ny=int(ctx.ny),
                    nz=int(ctx.nz),
                    h_m=h_m: _append_host_delta_snapshot(
                        out,
                        snap_u,
                        ambient_gpu=ambient_gpu,
                        nx=nx,
                        ny=ny,
                        nz=nz,
                        h_m=h_m,
                    )
                ),
            )
            cp.cuda.Stream.null.synchronize()
            rank_timing_stats[int(rank)]["tracer_solve_seconds"] += time.perf_counter() - tracer_t0
            correction_payloads[int(succ_j)] = delta_snaps_host
            captured_for_export = correction_payloads_out is not None and (
                capture_correction_components_norm is None
                or int(succ_j) in capture_correction_components_norm
            )
            if captured_for_export:
                correction_payloads_out[int(succ_j)] = list(delta_snaps_host)
            dst_rank = int(comp_to_rank[int(succ_j)])
            if int(dst_rank) == int(rank):
                rank_timing_stats[int(rank)]["num_local_corrections"] += 1.0
            else:
                rank_timing_stats[int(rank)]["num_remote_sends"] += 1.0
            del final_u
            _release_cupy_temporaries()
            if bool(stop_after_captured_corrections) and captured_for_export:
                rank_timing_stats[int(rank)]["pipeline_loop_seconds"] = time.perf_counter() - loop_t0
                rank_timing_stats[int(rank)]["rank_total_seconds"] = (
                    rank_timing_stats[int(rank)]["pipeline_loop_seconds"]
                )
                print(
                    f"[emulated rank {rank}] Early exit after capturing correction "
                    f"for component {int(succ_j)} in "
                    f"{rank_timing_stats[int(rank)]['pipeline_loop_seconds']:.3f} s"
                )
                _release_cupy_temporaries()
                print("[emulated] Early exit after requested correction snapshots.")
                return final_states_host, rank_timing_stats

        _release_cupy_temporaries()
        rank_timing_stats[int(rank)]["pipeline_loop_seconds"] = time.perf_counter() - loop_t0
        print(
            f"[emulated rank {rank}] Base/correction pipeline completed in "
            f"{rank_timing_stats[int(rank)]['pipeline_loop_seconds']:.3f} s"
        )

    for rank in sorted(int(r) for r in rank_assignments):
        post_t0 = time.perf_counter()
        assigned_comps = sorted(int(j) for j in rank_assignments.get(int(rank), []))
        for j in assigned_comps:
            pred_j = pred_map.get(int(j))
            if pred_j is None:
                final_states_host[int(j)] = list(base_states_host[int(j)])
                continue
            src_rank = int(comp_to_rank[int(pred_j)])
            delta_snaps = correction_payloads.get(int(j), [])
            if int(src_rank) != int(rank):
                rank_timing_stats[int(rank)]["num_remote_recvs"] += 1.0
            superpose_t0 = time.perf_counter()
            final_states_host[int(j)] = _superpose_snapshots(base_states_host[int(j)], delta_snaps)
            rank_timing_stats[int(rank)]["local_superpose_seconds"] += time.perf_counter() - superpose_t0
        rank_timing_stats[int(rank)]["post_pipeline_seconds"] = time.perf_counter() - post_t0
        rank_timing_stats[int(rank)]["rank_total_seconds"] = (
            rank_timing_stats[int(rank)]["pipeline_loop_seconds"]
            + rank_timing_stats[int(rank)]["post_pipeline_seconds"]
        )

    _release_cupy_temporaries()
    print("[emulated] Correction superposition completed.")
    return final_states_host, rank_timing_stats


__all__ = ["run_emulated_parallel_tracer"]
