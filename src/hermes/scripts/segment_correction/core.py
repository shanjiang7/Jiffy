import time
import cupy as cp
import numpy as np
from typing import Dict, List
from hermes.motion.types import PathDef
from hermes.scripts.outer_solver import OuterContext, run_ss_outer
from hermes.utils.snapshot_utils import crop_snapshot


_CORRECTION_CHUNK_MAX_BYTES = 64 * 1024 * 1024


def _build_predecessor_map(
    path_defs: List[PathDef],
    ss_per_layer: int,
) -> Dict[int, int | None]:
    """Map each component_id to its same-layer predecessor (None if first-of-layer)."""
    acc = 0
    comp_layer: Dict[int, int] = {}
    for pd in path_defs:
        comp_layer[pd.component_id] = acc // ss_per_layer
        acc += pd.weight
    ordered = [pd.component_id for pd in path_defs]
    pred: Dict[int, int | None] = {ordered[0]: None}
    for i in range(1, len(ordered)):
        cur, prev = ordered[i], ordered[i - 1]
        pred[cur] = prev if comp_layer[cur] == comp_layer[prev] else None
    return pred


def _legacy_component_predecessors(
    path_defs: List[PathDef],
    ss_per_layer: int,
) -> Dict[int, list[int]]:
    pred_map = _build_predecessor_map(path_defs, ss_per_layer)
    return {
        int(comp_id): ([] if pred is None else [int(pred)])
        for comp_id, pred in pred_map.items()
    }


def _normalise_dependency_maps(
    path_defs: List[PathDef],
    ss_per_layer: int,
    component_predecessors: Dict[int, List[int]] | None,
    component_successors: Dict[int, List[int]] | None,
) -> tuple[Dict[int, list[int]], Dict[int, list[int]]]:
    component_ids = sorted(int(pd.component_id) for pd in path_defs)
    if component_predecessors is None:
        pred_map = _legacy_component_predecessors(path_defs, ss_per_layer)
    else:
        pred_map = {
            int(comp_id): sorted({int(pred) for pred in preds})
            for comp_id, preds in component_predecessors.items()
        }
    for comp_id in component_ids:
        pred_map.setdefault(int(comp_id), [])

    if component_successors is None:
        succ_sets: Dict[int, set[int]] = {int(comp_id): set() for comp_id in component_ids}
        for dst, preds in pred_map.items():
            for src in preds:
                succ_sets.setdefault(int(src), set()).add(int(dst))
        succ_map = {
            int(comp_id): sorted(int(succ) for succ in succs)
            for comp_id, succs in succ_sets.items()
        }
    else:
        succ_map = {
            int(comp_id): sorted({int(succ) for succ in succs})
            for comp_id, succs in component_successors.items()
        }
    for comp_id in component_ids:
        succ_map.setdefault(int(comp_id), [])

    return pred_map, succ_map


def _build_component_index(path_defs: List[PathDef]) -> tuple[list[int], dict[int, int]]:
    ordered_component_ids = [int(pd.component_id) for pd in sorted(path_defs, key=lambda p: int(p.component_id))]
    return ordered_component_ids, {
        int(comp_id): int(idx) for idx, comp_id in enumerate(ordered_component_ids)
    }


def _build_bridge_run(
    *,
    src_component: int,
    dst_component: int,
    ordered_component_ids: list[int],
    component_index: dict[int, int],
    path_def_by_id: Dict[int, PathDef],
    steps_per_ss: int,
    snapshot_stride_steps: int | None,
    snapshot_steps_by_component: Dict[int, List[int]] | None,
    correction_horizon_ss_map: Dict[int, int] | None,
) -> tuple[float, float, list, int, list[int]]:
    src_idx = int(component_index[int(src_component)])
    dst_idx = int(component_index[int(dst_component)])
    if int(dst_idx) <= int(src_idx):
        raise ValueError(
            f"Component dependency must point forward in path order, got "
            f"{int(src_component)} -> {int(dst_component)}."
        )

    bridge_ids = ordered_component_ids[int(src_idx) + 1 : int(dst_idx) + 1]
    bridge_defs = [path_def_by_id[int(comp_id)] for comp_id in bridge_ids]
    if not bridge_defs:
        raise RuntimeError(
            f"No bridge path found for component dependency "
            f"{int(src_component)} -> {int(dst_component)}."
        )

    gap_steps = sum(int(pd.total_steps) for pd in bridge_defs[:-1])
    target_pd = path_def_by_id[int(dst_component)]
    target_snapshot_steps = _snapshot_steps_for_component(
        int(dst_component),
        snapshot_steps_by_component,
    )
    if target_snapshot_steps is None:
        stride = int(snapshot_stride_steps) if snapshot_stride_steps is not None else int(steps_per_ss)
        if int(stride) <= 0:
            raise ValueError("snapshot_stride_steps must be >= 1")
        target_snapshot_steps = list(range(0, int(target_pd.total_steps), int(stride)))

    horizon_ss = 1
    if correction_horizon_ss_map is not None:
        horizon_ss = max(1, int(correction_horizon_ss_map.get(int(dst_component), 1)))
    horizon_ss = min(int(horizon_ss), int(target_pd.weight))
    max_tracer_steps = int(gap_steps) + int(steps_per_ss) * int(horizon_ss)
    bridge_snapshot_steps = [
        int(gap_steps) + int(step)
        for step in target_snapshot_steps
        if int(gap_steps) + int(step) < int(max_tracer_steps)
    ]

    bridge_legs = [
        leg
        for pd in bridge_defs
        for leg in pd.legs
    ]
    return (
        float(bridge_defs[0].x_start),
        float(bridge_defs[0].y_start),
        bridge_legs,
        int(max_tracer_steps),
        bridge_snapshot_steps,
        int(gap_steps),
    )


def _snapshot_steps_for_component(
    component_id: int,
    snapshot_steps_by_component: Dict[int, List[int]] | None,
) -> list[int] | None:
    if snapshot_steps_by_component is None:
        return None
    return list(snapshot_steps_by_component.get(int(component_id), []))


def _superpose_correction_lists(
    base_snaps: List[np.ndarray],
    correction_lists: List[List[np.ndarray]],
) -> List[np.ndarray]:
    if not base_snaps:
        return []
    final_snaps = [np.array(base_snap, copy=True) for base_snap in base_snaps]
    for delta_snaps in correction_lists:
        for snap_idx, delta_snap in enumerate(delta_snaps):
            if int(snap_idx) >= len(final_snaps):
                break
            final_snaps[int(snap_idx)] = final_snaps[int(snap_idx)] + delta_snap
    return final_snaps


def _delta_max_abs_series(
    delta_snaps: List[np.ndarray],
    rel_snapshot_steps: list[int] | None,
) -> list[tuple[int, float]]:
    if not delta_snaps:
        return []
    if rel_snapshot_steps is None:
        rel_snapshot_steps = list(range(len(delta_snaps)))
    series: list[tuple[int, float]] = []
    for snap_idx, delta_snap in enumerate(delta_snaps):
        if snap_idx >= len(rel_snapshot_steps):
            break
        step = int(rel_snapshot_steps[snap_idx])
        max_abs = float(np.max(np.abs(delta_snap.astype(np.float64, copy=False))))
        series.append((step, max_abs))
    return series


def _chunk_snapshots_by_bytes(
    snaps: List[np.ndarray],
    *,
    max_bytes: int = _CORRECTION_CHUNK_MAX_BYTES,
) -> list[list[np.ndarray]]:
    if max_bytes < 1:
        raise ValueError("max_bytes must be >= 1")
    chunks: list[list[np.ndarray]] = []
    cur_chunk: list[np.ndarray] = []
    cur_bytes = 0
    for arr in snaps:
        arr_bytes = int(arr.nbytes)
        if cur_chunk and cur_bytes + arr_bytes > int(max_bytes):
            chunks.append(cur_chunk)
            cur_chunk = []
            cur_bytes = 0
        cur_chunk.append(arr)
        cur_bytes += arr_bytes
    if cur_chunk:
        chunks.append(cur_chunk)
    if not chunks:
        chunks = [[]]
    return chunks


def _enqueue_correction_send(
    *,
    comm,
    dest_rank: int,
    tag: int,
    delta_snaps_host: List[np.ndarray],
    send_reqs: list[object],
) -> None:
    chunks = _chunk_snapshots_by_bytes(delta_snaps_host)
    send_reqs.append(comm.isend(("meta", len(chunks)), dest=int(dest_rank), tag=int(tag)))
    for chunk_idx, chunk in enumerate(chunks):
        send_reqs.append(comm.isend(("chunk", chunk_idx, chunk), dest=int(dest_rank), tag=int(tag)))


def _recv_correction_chunks(
    *,
    comm,
    source_rank: int,
    tag: int,
) -> list[np.ndarray]:
    kind, num_chunks = comm.recv(source=int(source_rank), tag=int(tag))
    if kind != "meta":
        raise RuntimeError(f"Expected correction meta message for tag={tag}, got kind={kind!r}")
    delta_snaps: list[np.ndarray] = []
    for expected_idx in range(int(num_chunks)):
        kind, chunk_idx, chunk = comm.recv(source=int(source_rank), tag=int(tag))
        if kind != "chunk":
            raise RuntimeError(f"Expected correction chunk message for tag={tag}, got kind={kind!r}")
        if int(chunk_idx) != int(expected_idx):
            raise RuntimeError(
                f"Out-of-order correction chunk for tag={tag}: expected {expected_idx}, got {chunk_idx}"
            )
        delta_snaps.extend(list(chunk))
    return delta_snaps


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


def run_parallel_tracer(
    ctx: OuterContext,
    ambient_gpu: cp.ndarray,
    path_defs: List[PathDef],
    assigned_comps: List[int],
    comm,
    rank: int,
    world_size: int,
    rank_assignments: Dict[int, List[int]],
    steps_per_ss: int,
    ss_per_layer: int,
    snapshot_stride_steps: int | None = None,
    snapshot_steps_by_component: Dict[int, List[int]] | None = None,
    correction_horizon_ss_map: Dict[int, int] | None = None,
    component_predecessors: Dict[int, List[int]] | None = None,
    component_successors: Dict[int, List[int]] | None = None,
    deltaT_K: float = 1.0,
    collect_output_snapshots: bool = True,
    h_m: float | None = None,
    self_check_maps: dict | None = None,
) -> tuple[Dict[int, List[np.ndarray]], dict[str, float]]:
    """
    Pipelined component execution with source-side tracer correction.

    Each rank runs the base simulation of its local components in path order.
    Immediately after finishing a component, it computes outgoing source-off
    corrections for every successor listed in the component-level DAG. Remote
    correction payloads are sent to the successor owner rank. The successor rank
    later superposes all incoming correction snapshots onto its own base
    snapshots.

    Returns `(final_states_host, timing_stats)`, where `final_states_host` maps
    component id -> list of numpy snapshot arrays when
    `collect_output_snapshots=True`; otherwise it is empty after executing the
    base/correction pipeline and communication.
    """
    _ = world_size
    assigned_comps = sorted(int(j) for j in assigned_comps)
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

    self_check_active = bool(self_check_maps) and bool(collect_output_snapshots)

    self_check_emitters: set[int] = set()
    if self_check_active:
        self_check_emitters = _self_check_emitters(
            assigned_comps=assigned_comps,
            succ_map=succ_map,
            self_check_maps=self_check_maps,
            correction_horizon_ss_map=correction_horizon_ss_map,
            path_def_by_id=path_def_by_id,
        )

    base_states_host: Dict[int, List[np.ndarray]] = {}
    final_states_host: Dict[int, List[np.ndarray]] = {}
    # End states retained (host side) only for components that must emit
    # additional self-check corrections after the main pipeline.
    end_states_host: Dict[int, np.ndarray] = {}
    local_correction_snaps: Dict[int, Dict[int, List[np.ndarray]]] = {}
    send_reqs: list[object] = []
    timing_stats = {
        "base_solve_seconds": 0.0,
        "tracer_solve_seconds": 0.0,
        "recv_wait_seconds": 0.0,
        "send_wait_seconds": 0.0,
        "local_superpose_seconds": 0.0,
        "pipeline_loop_seconds": 0.0,
        "post_pipeline_seconds": 0.0,
        "num_assigned_components": float(len(assigned_comps)),
        "num_remote_sends": 0.0,
        "num_remote_recvs": 0.0,
        "num_local_corrections": 0.0,
        "num_correction_edges": float(
            sum(len(succ_map.get(int(j), [])) for j in assigned_comps)
        ),
        "max_component_predecessors": float(
            max((len(preds) for preds in pred_map.values()), default=0)
        ),
    }

    print(f"[rank {rank}] Base + outgoing correction pipeline.")
    loop_t0 = time.perf_counter()
    for j in assigned_comps:
        pd = path_def_by_id[int(j)]
        comp_snapshot_steps = _snapshot_steps_for_component(int(j), snapshot_steps_by_component)
        if not collect_output_snapshots:
            comp_snapshot_steps = []
        base_snaps_host: List[np.ndarray] = []
        base_t0 = time.perf_counter()
        _, final_u = run_ss_outer(
            ctx, ambient_gpu, pd.x_start, pd.y_start, pd.legs, steps_per_ss,
            source_on=True,
            snapshot_stride_steps=snapshot_stride_steps,
            snapshot_steps=comp_snapshot_steps,
            snapshot_callback=(
                lambda snap_u, out=base_snaps_host: _append_host_snapshot(
                    out,
                    snap_u,
                    nx=int(ctx.nx),
                    ny=int(ctx.ny),
                    nz=int(ctx.nz),
                    h_m=h_m,
                )
            ) if collect_output_snapshots else None,
        )
        cp.cuda.Stream.null.synchronize()
        timing_stats["base_solve_seconds"] += time.perf_counter() - base_t0
        if collect_output_snapshots:
            base_states_host[j] = base_snaps_host
        if self_check_active and int(j) in self_check_emitters:
            end_states_host[int(j)] = cp.asnumpy(final_u)

        for succ_j in succ_map.get(int(j), []):
            (
                bridge_x_start,
                bridge_y_start,
                bridge_legs,
                max_tracer_steps,
                bridge_snapshot_steps,
                _gap_steps,
            ) = _build_bridge_run(
                src_component=int(j),
                dst_component=int(succ_j),
                ordered_component_ids=ordered_component_ids,
                component_index=component_index,
                path_def_by_id=path_def_by_id,
                steps_per_ss=steps_per_ss,
                snapshot_stride_steps=snapshot_stride_steps,
                snapshot_steps_by_component=snapshot_steps_by_component,
                correction_horizon_ss_map=correction_horizon_ss_map,
            )
            delta_snaps_host: List[np.ndarray] = []
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
                snapshot_steps=bridge_snapshot_steps,
                snapshot_callback=lambda snap_u, out=delta_snaps_host: _append_host_delta_snapshot(
                    out,
                    snap_u,
                    ambient_gpu=ambient_gpu,
                    nx=int(ctx.nx),
                    ny=int(ctx.ny),
                    nz=int(ctx.nz),
                    h_m=h_m,
                ),
            )
            cp.cuda.Stream.null.synchronize()
            timing_stats["tracer_solve_seconds"] += time.perf_counter() - tracer_t0
            dst_rank = int(comp_to_rank[int(succ_j)])
            if int(dst_rank) == int(rank):
                if collect_output_snapshots:
                    local_correction_snaps.setdefault(int(succ_j), {})[int(j)] = delta_snaps_host
                timing_stats["num_local_corrections"] += 1.0
            else:
                _enqueue_correction_send(
                    comm=comm,
                    dest_rank=int(dst_rank),
                    tag=int(succ_j),
                    delta_snaps_host=delta_snaps_host,
                    send_reqs=send_reqs,
                )
                timing_stats["num_remote_sends"] += 1.0

    cp.cuda.Stream.null.synchronize()
    timing_stats["pipeline_loop_seconds"] = time.perf_counter() - loop_t0
    print(f"[rank {rank}] Base/correction pipeline completed in {time.perf_counter() - loop_t0:.3f} s")

    post_t0 = time.perf_counter()
    for j in assigned_comps:
        pred_js = pred_map.get(int(j), [])
        if not pred_js:
            if collect_output_snapshots:
                final_states_host[int(j)] = list(base_states_host[int(j)])
            continue
        correction_lists: List[List[np.ndarray]] = []
        for pred_j in pred_js:
            src_rank = int(comp_to_rank[int(pred_j)])
            if int(src_rank) == int(rank):
                delta_snaps = local_correction_snaps.get(int(j), {}).get(int(pred_j), [])
            else:
                recv_t0 = time.perf_counter()
                delta_snaps = _recv_correction_chunks(
                    comm=comm,
                    source_rank=int(src_rank),
                    tag=int(j),
                )
                timing_stats["recv_wait_seconds"] += time.perf_counter() - recv_t0
                timing_stats["num_remote_recvs"] += 1.0
            correction_lists.append(delta_snaps)
        if collect_output_snapshots:
            superpose_t0 = time.perf_counter()
            final_states_host[int(j)] = _superpose_correction_lists(
                base_states_host[int(j)],
                correction_lists,
            )
            timing_stats["local_superpose_seconds"] += time.perf_counter() - superpose_t0

    for req in send_reqs:
        send_wait_t0 = time.perf_counter()
        req.wait()
        timing_stats["send_wait_seconds"] += time.perf_counter() - send_wait_t0

    cp.cuda.Stream.null.synchronize()
    timing_stats["post_pipeline_seconds"] = time.perf_counter() - post_t0
    print(f"[rank {rank}] Correction superposition completed.")

    if self_check_active:
        _run_self_check_round(
            ctx=ctx,
            ambient_gpu=ambient_gpu,
            comm=comm,
            rank=rank,
            assigned_comps=assigned_comps,
            self_check_maps=self_check_maps,
            end_states_host=end_states_host,
            final_states_host=final_states_host,
            comp_to_rank=comp_to_rank,
            ordered_component_ids=ordered_component_ids,
            component_index=component_index,
            path_def_by_id=path_def_by_id,
            steps_per_ss=steps_per_ss,
            snapshot_stride_steps=snapshot_stride_steps,
            snapshot_steps_by_component=snapshot_steps_by_component,
            production_succ_map=succ_map,
            production_pred_map=pred_map,
            production_horizon_ss_map=correction_horizon_ss_map,
            h_m=h_m,
            timing_stats=timing_stats,
        )

    return (final_states_host if collect_output_snapshots else {}), timing_stats


def _effective_horizon_ss(
    horizon_ss_map: Dict[int, int] | None,
    dst_component: int,
    dst_weight: int,
) -> int:
    """Horizon actually used by _build_bridge_run for this destination."""
    horizon_ss = 1
    if horizon_ss_map is not None:
        horizon_ss = max(1, int(horizon_ss_map.get(int(dst_component), 1)))
    return min(int(horizon_ss), int(dst_weight))


def _self_check_ext_horizon_ss(
    deep_horizon_map: Dict[int, int] | None,
    production_horizon_map: Dict[int, int] | None,
    dst_component: int,
    dst_weight: int,
) -> int:
    """
    Refined correction horizon for an existing pair: at least one supersegment
    beyond the production horizon (the first neglected tier - chain-like paths
    keep the same cut depth at the tighter threshold, but their truncated
    correction tail is exactly the error), or the deeper DAG's horizon if that
    reaches further. Capped at the target's full extent.
    """
    old_h = _effective_horizon_ss(production_horizon_map, dst_component, dst_weight)
    deep_h = _effective_horizon_ss(deep_horizon_map, dst_component, dst_weight)
    return min(int(dst_weight), max(int(deep_h), int(old_h) + 1))


def _self_check_emitters(
    *,
    assigned_comps: List[int],
    succ_map: Dict[int, List[int]],
    self_check_maps: dict,
    correction_horizon_ss_map: Dict[int, int] | None,
    path_def_by_id: Dict[int, PathDef],
) -> set[int]:
    """Owned components that must emit a phase-2 correction (new pair or
    horizon extension of an existing pair)."""
    refine_succ = self_check_maps["component_successors"]
    deep_horizon = self_check_maps["horizon_ss_map"]
    emitters: set[int] = set()
    for j in assigned_comps:
        if refine_succ.get(int(j)):
            emitters.add(int(j))
            continue
        for succ_j in succ_map.get(int(j), []):
            w = int(path_def_by_id[int(succ_j)].weight)
            old_h = _effective_horizon_ss(correction_horizon_ss_map, succ_j, w)
            if _self_check_ext_horizon_ss(deep_horizon, correction_horizon_ss_map, succ_j, w) > old_h:
                emitters.add(int(j))
                break
    return emitters


def _run_self_check_round(
    *,
    ctx: OuterContext,
    ambient_gpu: cp.ndarray,
    comm,
    rank: int,
    assigned_comps: List[int],
    self_check_maps: dict,
    end_states_host: Dict[int, np.ndarray],
    final_states_host: Dict[int, List[np.ndarray]],
    comp_to_rank: Dict[int, int],
    ordered_component_ids: list[int],
    component_index: dict[int, int],
    path_def_by_id: Dict[int, PathDef],
    steps_per_ss: int,
    snapshot_stride_steps: int | None,
    snapshot_steps_by_component: Dict[int, List[int]] | None,
    production_succ_map: Dict[int, List[int]],
    production_pred_map: Dict[int, List[int]],
    production_horizon_ss_map: Dict[int, int] | None,
    h_m: float | None,
    timing_stats: dict,
) -> None:
    """
    Self-convergence error estimate (a-posteriori, no serial reference).

    Two neglect channels are refined against the production run:
      1. NEW PAIRS - component dependencies present only in the deeper DAG
         built at level_K / gamma (full corrections, snapshot offset 0);
      2. HORIZON EXTENSIONS - existing dependencies whose deeper cut depth
         extends the correction window into the target (only the extension
         snapshots are computed, applied at their snapshot offset).
    The refined snapshots are superposed and the rel-L2 between production and
    refined states is reported - the leading (first-neglected-tier) term of the
    true parallel-vs-serial error.
    """
    t0 = time.perf_counter()
    # Separate communicator so self-check messages can reuse plain component-id
    # tags without colliding with phase-1 correction traffic (MPI only
    # guarantees tag values up to 32767).
    comm_sc = comm.Dup()
    refine_pred = {
        int(k): [int(v) for v in vs]
        for k, vs in self_check_maps["component_predecessors"].items()
    }
    refine_succ = {
        int(k): [int(v) for v in vs]
        for k, vs in self_check_maps["component_successors"].items()
    }
    deep_horizon = {int(k): int(v) for k, v in self_check_maps["horizon_ss_map"].items()}

    def _bridge(src_j: int, dst_j: int, horizon_map):
        return _build_bridge_run(
            src_component=int(src_j),
            dst_component=int(dst_j),
            ordered_component_ids=ordered_component_ids,
            component_index=component_index,
            path_def_by_id=path_def_by_id,
            steps_per_ss=steps_per_ss,
            snapshot_stride_steps=snapshot_stride_steps,
            snapshot_steps_by_component=snapshot_steps_by_component,
            correction_horizon_ss_map=horizon_map,
        )

    def _trace(src_j: int, x0, y0, legs, max_steps, snap_steps) -> List[np.ndarray]:
        deltas: List[np.ndarray] = []
        final_u = cp.asarray(end_states_host[int(src_j)])
        run_ss_outer(
            ctx,
            final_u,
            x0,
            y0,
            legs,
            steps_per_ss,
            source_on=False,
            max_steps=max_steps,
            snapshot_stride_steps=snapshot_stride_steps,
            snapshot_steps=snap_steps,
            snapshot_callback=lambda snap_u, out=deltas: _append_host_delta_snapshot(
                out,
                snap_u,
                ambient_gpu=ambient_gpu,
                nx=int(ctx.nx),
                ny=int(ctx.ny),
                nz=int(ctx.nz),
                h_m=h_m,
            ),
        )
        cp.cuda.Stream.null.synchronize()
        return deltas

    # ---- emit phase-2 corrections -------------------------------------------
    send_reqs: list[object] = []
    local_deltas: Dict[int, Dict[int, tuple[int, List[np.ndarray]]]] = {}
    n_tracers = 0

    def _dispatch(src_j: int, dst_j: int, offset: int, deltas: List[np.ndarray]) -> None:
        nonlocal n_tracers
        n_tracers += 1
        dst_rank = int(comp_to_rank[int(dst_j)])
        if int(dst_rank) == int(rank):
            local_deltas.setdefault(int(dst_j), {})[int(src_j)] = (int(offset), deltas)
        else:
            send_reqs.append(
                comm_sc.isend((int(offset),), dest=int(dst_rank), tag=int(dst_j))
            )
            _enqueue_correction_send(
                comm=comm_sc,
                dest_rank=int(dst_rank),
                tag=int(dst_j),
                delta_snaps_host=deltas,
                send_reqs=send_reqs,
            )

    for j in assigned_comps:
        # (1) new pairs: full correction over the deep horizon window
        for succ_j in refine_succ.get(int(j), []):
            x0, y0, legs, max_steps, snap_steps, _gap = _bridge(j, succ_j, deep_horizon)
            _dispatch(j, succ_j, 0, _trace(j, x0, y0, legs, max_steps, snap_steps))
        # (2) horizon extensions of existing pairs: only the extension window
        for succ_j in production_succ_map.get(int(j), []):
            w = int(path_def_by_id[int(succ_j)].weight)
            old_h = _effective_horizon_ss(production_horizon_ss_map, succ_j, w)
            new_h = _self_check_ext_horizon_ss(deep_horizon, production_horizon_ss_map, succ_j, w)
            if new_h <= old_h:
                continue
            ext_horizon = dict(deep_horizon)
            ext_horizon[int(succ_j)] = int(new_h)
            x0, y0, legs, max_steps, snap_steps, gap = _bridge(j, succ_j, ext_horizon)
            old_window = int(gap) + int(steps_per_ss) * int(old_h)
            offset = sum(1 for st in snap_steps if int(st) < int(old_window))
            ext_steps = [int(st) for st in snap_steps if int(st) >= int(old_window)]
            # Always dispatch (even an empty extension): the receiver expects a
            # message from every refined predecessor.
            _dispatch(
                j,
                succ_j,
                offset,
                _trace(j, x0, y0, legs, max_steps, ext_steps) if ext_steps else [],
            )

    # ---- receive, superpose, measure ----------------------------------------
    expected_sources: Dict[int, list[int]] = {}
    for j in assigned_comps:
        srcs = list(refine_pred.get(int(j), []))
        for pred_j in production_pred_map.get(int(j), []):
            w = int(path_def_by_id[int(j)].weight)
            old_h = _effective_horizon_ss(production_horizon_ss_map, j, w)
            new_h = _self_check_ext_horizon_ss(deep_horizon, production_horizon_ss_map, j, w)
            if new_h > old_h:
                srcs.append(int(pred_j))
        if srcs:
            expected_sources[int(j)] = sorted(set(srcs))

    local_sq_num = 0.0
    local_sq_den = 0.0
    local_max_rel = 0.0
    local_n_snaps = 0
    for j, srcs in expected_sources.items():
        production = final_states_host.get(int(j), [])
        refined = [np.array(snap, copy=True) for snap in production]
        for pred_j in srcs:
            src_rank = int(comp_to_rank[int(pred_j)])
            if int(src_rank) == int(rank):
                entry = local_deltas.get(int(j), {}).get(int(pred_j))
                if entry is None:
                    continue
                offset, deltas = entry
            else:
                (offset,) = comm_sc.recv(source=int(src_rank), tag=int(j))
                deltas = _recv_correction_chunks(
                    comm=comm_sc,
                    source_rank=int(src_rank),
                    tag=int(j),
                )
            for k, delta in enumerate(deltas):
                idx = int(offset) + int(k)
                if idx >= len(refined):
                    break
                refined[idx] = refined[idx] + delta
        for old, new in zip(production, refined):
            diff = new.astype(np.float64) - old.astype(np.float64)
            den = float(np.linalg.norm(new.astype(np.float64)))
            rel = float(np.linalg.norm(diff)) / max(den, 1e-30)
            local_max_rel = max(local_max_rel, rel)
            local_sq_num += float(np.dot(diff.ravel(), diff.ravel()))
            local_sq_den += den * den
            local_n_snaps += 1

    for req in send_reqs:
        req.wait()

    from mpi4py import MPI

    global_max_rel = comm_sc.allreduce(local_max_rel, op=MPI.MAX)
    global_sq_num = comm_sc.allreduce(local_sq_num, op=MPI.SUM)
    global_sq_den = comm_sc.allreduce(local_sq_den, op=MPI.SUM)
    global_n_snaps = comm_sc.allreduce(local_n_snaps, op=MPI.SUM)
    global_n_tracers = comm_sc.allreduce(n_tracers, op=MPI.SUM)
    comm_sc.Free()
    global_rms_rel = (
        (global_sq_num / global_sq_den) ** 0.5 if global_sq_den > 0.0 else 0.0
    )

    timing_stats["self_check_seconds"] = time.perf_counter() - t0
    timing_stats["self_check_max_rel_l2"] = float(global_max_rel)
    timing_stats["self_check_rms_rel_l2"] = float(global_rms_rel)
    timing_stats["self_check_num_snapshots"] = float(global_n_snaps)
    timing_stats["self_check_num_extra_corrections"] = float(global_n_tracers)
    if rank == 0:
        print(
            "[self-check] estimated parallel-vs-serial rel-L2 "
            f"(gamma={float(self_check_maps.get('gamma', 0.0)):.3g}, "
            f"refinement level_K={float(self_check_maps.get('level_K', 0.0)):.6g} K): "
            f"max={global_max_rel:.4e}  rms={global_rms_rel:.4e}  "
            f"over {int(global_n_snaps)} snapshot(s), "
            f"{int(global_n_tracers)} extra correction(s) "
            f"[{int(self_check_maps.get('num_extra_edges', 0))} new pair(s) + horizon extensions]",
            flush=True,
        )
