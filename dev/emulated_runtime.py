from __future__ import annotations

import time
from typing import Dict, List

import cupy as cp
import numpy as np

from hermes.motion.types import PathDef
from hermes.scripts.outer_solver import OuterContext, run_ss_outer
from hermes.scripts.segment_correction.core import (
    _build_component_index,
    _build_fused_bridge_run,
    _normalise_dependency_maps,
    _snapshot_steps_for_component,
    _superpose_correction_lists,
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
    component_predecessors: Dict[int, List[int]] | None = None,
    component_successors: Dict[int, List[int]] | None = None,
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
    pred_map, succ_map = _normalise_dependency_maps(
        path_defs,
        ss_per_layer,
        component_predecessors,
        component_successors,
    )
    path_def_by_id = {int(pd.component_id): pd for pd in path_defs}
    ordered_component_ids, component_index = _build_component_index(path_defs)
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
    correction_payloads: Dict[int, Dict[int, List[np.ndarray]]] = {}
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
            "num_correction_edges": float(
                sum(len(succ_map.get(int(comp_id), [])) for comp_id in rank_assignments.get(int(rank), []))
            ),
            "max_component_predecessors": float(
                max((len(preds) for preds in pred_map.values()), default=0)
            ),
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

            captured_any_for_export = False
            succ_list = list(succ_map.get(int(j), []))
            if not succ_list:
                del final_u
                _release_cupy_temporaries()
                continue

            # One tracer serves every successor; see _build_fused_bridge_run.
            (
                bridge_x_start,
                bridge_y_start,
                bridge_legs,
                max_tracer_steps,
                union_snapshot_steps,
                per_succ_snapshot_steps,
            ) = _build_fused_bridge_run(
                src_component=int(j),
                dst_components=[int(s) for s in succ_list],
                ordered_component_ids=ordered_component_ids,
                component_index=component_index,
                path_def_by_id=path_def_by_id,
                steps_per_ss=steps_per_ss,
                snapshot_stride_steps=snapshot_stride_steps,
                snapshot_steps_by_component=snapshot_steps_by_component,
                correction_horizon_ss_map=correction_horizon_ss_map,
            )
            captured_snaps_host: List[np.ndarray] = []
            tracer_t0 = time.perf_counter()
            _, _ = run_ss_outer(
                ctx,
                final_u,
                bridge_x_start,
                bridge_y_start,
                bridge_legs,
                steps_per_ss,
                source_on=False,
                max_steps=max_tracer_steps,
                snapshot_stride_steps=snapshot_stride_steps,
                snapshot_steps=union_snapshot_steps,
                snapshot_callback=(
                    lambda snap_u,
                    out=captured_snaps_host,
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
            if len(captured_snaps_host) != len(union_snapshot_steps):
                raise RuntimeError(
                    f"Fused bridge tracer for component {int(j)} captured "
                    f"{len(captured_snaps_host)} snapshots, expected "
                    f"{len(union_snapshot_steps)}."
                )
            snap_index_by_step = {
                int(step): idx for idx, step in enumerate(union_snapshot_steps)
            }

            for succ_j in succ_list:
                delta_snaps_host = [
                    captured_snaps_host[snap_index_by_step[int(step)]]
                    for step in per_succ_snapshot_steps[int(succ_j)]
                ]
                correction_payloads.setdefault(int(succ_j), {})[int(j)] = delta_snaps_host
                captured_for_export = correction_payloads_out is not None and (
                    capture_correction_components_norm is None
                    or int(succ_j) in capture_correction_components_norm
                )
                if captured_for_export:
                    correction_payloads_out[int(succ_j)] = list(delta_snaps_host)
                    captured_any_for_export = True
                dst_rank = int(comp_to_rank[int(succ_j)])
                if int(dst_rank) == int(rank):
                    rank_timing_stats[int(rank)]["num_local_corrections"] += 1.0
                else:
                    rank_timing_stats[int(rank)]["num_remote_sends"] += 1.0
                if bool(stop_after_captured_corrections) and captured_for_export:
                    break
            del final_u
            _release_cupy_temporaries()
            if bool(stop_after_captured_corrections) and captured_any_for_export:
                rank_timing_stats[int(rank)]["pipeline_loop_seconds"] = time.perf_counter() - loop_t0
                rank_timing_stats[int(rank)]["rank_total_seconds"] = (
                    rank_timing_stats[int(rank)]["pipeline_loop_seconds"]
                )
                print(
                    f"[emulated rank {rank}] Early exit after capturing correction "
                    f"in {rank_timing_stats[int(rank)]['pipeline_loop_seconds']:.3f} s"
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
            pred_js = pred_map.get(int(j), [])
            if not pred_js:
                final_states_host[int(j)] = list(base_states_host[int(j)])
                continue
            correction_lists: List[List[np.ndarray]] = []
            for pred_j in pred_js:
                src_rank = int(comp_to_rank[int(pred_j)])
                delta_snaps = correction_payloads.get(int(j), {}).get(int(pred_j), [])
                if int(src_rank) != int(rank):
                    rank_timing_stats[int(rank)]["num_remote_recvs"] += 1.0
                correction_lists.append(delta_snaps)
            superpose_t0 = time.perf_counter()
            final_states_host[int(j)] = _superpose_correction_lists(
                base_states_host[int(j)],
                correction_lists,
            )
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
