import os
import time
import cupy as cp
import numpy as np
from typing import Dict, List
from hermes.motion.types import PathDef
from hermes.scripts.outer_solver import OuterContext, run_ss_outer
from hermes.utils.snapshot_utils import crop_snapshot


_CORRECTION_CHUNK_MAX_BYTES = 64 * 1024 * 1024

# HERMES_TRACER_PROFILE=1 breaks each tracer's cost into setup / movement-cache
# build / snapshot-D2H components (reported per rank in timing_summary.json).
# Adds synchronization points inside the tracer, so leave off for timing runs.
_TRACER_PROFILE = os.environ.get("HERMES_TRACER_PROFILE", "") == "1"
_TRACER_PROFILE_TEMPLATE = {
    "setup_seconds": 0.0,
    "cache_seconds": 0.0,
    "cache_builds": 0.0,
    "snap_d2h_seconds": 0.0,
}


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


def _build_edge_window(
    *,
    src_component: int,
    dst_component: int,
    ordered_component_ids: list[int],
    component_index: dict[int, int],
    path_def_by_id: Dict[int, PathDef],
    steps_per_ss: int,
    snapshot_stride_steps: int | None,
    snapshot_steps_by_component: Dict[int, List[int]] | None,
    horizon_ss: int,
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

    # horizon_ss is how deep THIS source's retained influence reaches into the
    # destination, in supersegments. The caller decides its value (production:
    # the planner's per-edge map; self-check ladder: its per-component rung
    # map); this builder only clamps it to the destination's extent.
    horizon_ss = min(max(1, int(horizon_ss)), int(target_pd.weight))
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


def _build_fused_tracer_run(
    *,
    src_component: int,
    dst_components: List[int],
    ordered_component_ids: list[int],
    component_index: dict[int, int],
    path_def_by_id: Dict[int, PathDef],
    steps_per_ss: int,
    snapshot_stride_steps: int | None,
    snapshot_steps_by_component: Dict[int, List[int]] | None,
    horizon_ss_by_dst: Dict[int, int],
) -> tuple[float, float, list, int, list[int], Dict[int, List[int]]]:
    """One source-off tracer covering every successor of ``src_component``.

    All of a component's bridges start from the same end state and follow the
    same legs, so the bridge to a nearer successor is a strict prefix of the
    bridge to a farther one. Tracing each successor separately re-simulates
    that shared prefix once per successor; instead we trace once to the
    farthest successor and record each successor's snapshot steps as we pass
    its window. Per-successor snapshot steps are taken from
    :func:`_build_edge_window`, so the captured fields match what per-successor
    runs would produce for the same horizons.

    ``horizon_ss_by_dst`` gives, for each successor, how deep this source is
    traced into it — its own retained reach, not the destination's full
    correction window. The farthest-reaching successor determines the run
    length and the longest leg list, so every successor's window remains a
    prefix of the traced trajectory.

    Returns the fused run plus the union of snapshot steps to capture and the
    per-successor step lists needed to demultiplex the result.
    """
    if not dst_components:
        raise ValueError("dst_components must be non-empty")

    per_dst_steps: Dict[int, List[int]] = {}
    x_start = 0.0
    y_start = 0.0
    bridge_legs: list = []
    max_tracer_steps = 0
    for dst in dst_components:
        (
            dst_x_start,
            dst_y_start,
            dst_legs,
            dst_max_steps,
            dst_snapshot_steps,
            _gap_steps,
        ) = _build_edge_window(
            src_component=int(src_component),
            dst_component=int(dst),
            ordered_component_ids=ordered_component_ids,
            component_index=component_index,
            path_def_by_id=path_def_by_id,
            steps_per_ss=steps_per_ss,
            snapshot_stride_steps=snapshot_stride_steps,
            snapshot_steps_by_component=snapshot_steps_by_component,
            horizon_ss=int(horizon_ss_by_dst[int(dst)]),
        )
        per_dst_steps[int(dst)] = list(dst_snapshot_steps)
        max_tracer_steps = max(int(max_tracer_steps), int(dst_max_steps))
        # The farthest successor owns the longest leg list; every shorter
        # bridge is a prefix of it, and all share a start point.
        if len(dst_legs) > len(bridge_legs):
            bridge_legs = dst_legs
            x_start = float(dst_x_start)
            y_start = float(dst_y_start)

    union_steps = sorted({int(s) for steps in per_dst_steps.values() for s in steps})
    return (
        float(x_start),
        float(y_start),
        bridge_legs,
        int(max_tracer_steps),
        union_steps,
        per_dst_steps,
    )


def _edge_horizons_for_source(
    src_component: int,
    dst_components: List[int],
    correction_horizon_by_edge: Dict[tuple[int, int], int] | None,
) -> Dict[int, int]:
    """Per-destination trace depths for one source, from the planner's map.

    The planner records, for every retained (src, dst) component edge, how deep
    the source's retained influence reaches into the destination. That map is
    the single source of truth for production trace depths; a missing entry
    means the plan and the DAG disagree, which is an error, not a fallback.
    """
    if correction_horizon_by_edge is None:
        raise ValueError(
            "correction_horizon_by_edge is required: production corrections "
            "trace each edge to its own retained reach "
            "(runtime_plan['correction_horizon_by_edge'])."
        )
    horizons: Dict[int, int] = {}
    for dst in dst_components:
        key = (int(src_component), int(dst))
        if key not in correction_horizon_by_edge:
            raise KeyError(
                f"No correction horizon for component edge {key}: the DAG "
                f"retains this edge but the plan's correction_horizon_by_edge "
                f"has no entry for it."
            )
        horizons[int(dst)] = int(correction_horizon_by_edge[key])
    return horizons


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
    correction_horizon_by_edge: Dict[tuple[int, int], int] | None = None,
    component_predecessors: Dict[int, List[int]] | None = None,
    component_successors: Dict[int, List[int]] | None = None,
    collect_output_snapshots: bool = True,
    h_m: float | None = None,
    self_check_maps: dict | None = None,
    self_check_save_callback=None,
) -> tuple[Dict[int, List[np.ndarray]], dict[str, float]]:
    """
    Pipelined component execution with source-side tracer correction.

    Each rank runs the base simulation of its local components in path order.
    Immediately after finishing a component, it runs ONE source-off tracer that
    serves every successor listed in the component-level DAG: the tracer runs
    to the farthest successor's retained reach, and each successor's correction
    snapshots are demultiplexed out of that single run (a nearer successor's
    window is a prefix of a farther one's trajectory). Trace depths come from
    `correction_horizon_by_edge`, the planner's per-(src, dst) reach map — the
    single source of truth for how deep each correction is simulated; it is
    required. Remote correction payloads are sent to the successor owner rank,
    which later superposes all incoming corrections onto its base snapshots.

    `correction_horizon_ss_map` (per-destination horizons) is used only by the
    optional self-check ladder as the baseline it extends from.

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

        succ_list = list(succ_map.get(int(j), []))
        if succ_list:
            # One tracer serves every successor: their bridges share a start
            # state and a leg prefix, so a separate run per successor would
            # re-simulate that prefix. Each successor is traced only as deep
            # as this source's retained influence reaches into it.
            # See _build_fused_tracer_run.
            (
                bridge_x_start,
                bridge_y_start,
                bridge_legs,
                max_tracer_steps,
                union_snapshot_steps,
                per_succ_snapshot_steps,
            ) = _build_fused_tracer_run(
                src_component=int(j),
                dst_components=[int(s) for s in succ_list],
                ordered_component_ids=ordered_component_ids,
                component_index=component_index,
                path_def_by_id=path_def_by_id,
                steps_per_ss=steps_per_ss,
                snapshot_stride_steps=snapshot_stride_steps,
                snapshot_steps_by_component=snapshot_steps_by_component,
                horizon_ss_by_dst=_edge_horizons_for_source(
                    int(j), succ_list, correction_horizon_by_edge
                ),
            )
            captured_snaps_host: List[np.ndarray] = []
            tracer_profile = _TRACER_PROFILE_TEMPLATE.copy() if _TRACER_PROFILE else None

            def _tracer_snapshot_cb(snap_u, out=captured_snaps_host, prof=tracer_profile):
                if prof is None:
                    _append_host_delta_snapshot(
                        out, snap_u, ambient_gpu=ambient_gpu,
                        nx=int(ctx.nx), ny=int(ctx.ny), nz=int(ctx.nz), h_m=h_m,
                    )
                    return
                s0 = time.perf_counter()
                _append_host_delta_snapshot(
                    out, snap_u, ambient_gpu=ambient_gpu,
                    nx=int(ctx.nx), ny=int(ctx.ny), nz=int(ctx.nz), h_m=h_m,
                )
                prof["snap_d2h_seconds"] += time.perf_counter() - s0

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
                snapshot_callback=_tracer_snapshot_cb,
                profile=tracer_profile,
            )
            cp.cuda.Stream.null.synchronize()
            timing_stats["tracer_solve_seconds"] += time.perf_counter() - tracer_t0
            if tracer_profile is not None:
                timing_stats["tracer_profile_setup_seconds"] = (
                    timing_stats.get("tracer_profile_setup_seconds", 0.0)
                    + tracer_profile["setup_seconds"]
                )
                timing_stats["tracer_profile_cache_seconds"] = (
                    timing_stats.get("tracer_profile_cache_seconds", 0.0)
                    + tracer_profile["cache_seconds"]
                )
                timing_stats["tracer_profile_cache_builds"] = (
                    timing_stats.get("tracer_profile_cache_builds", 0.0)
                    + tracer_profile["cache_builds"]
                )
                timing_stats["tracer_profile_snap_d2h_seconds"] = (
                    timing_stats.get("tracer_profile_snap_d2h_seconds", 0.0)
                    + tracer_profile["snap_d2h_seconds"]
                )
                timing_stats["tracer_profile_runs"] = (
                    timing_stats.get("tracer_profile_runs", 0.0) + 1.0
                )
                timing_stats["tracer_profile_snapshots"] = (
                    timing_stats.get("tracer_profile_snapshots", 0.0)
                    + float(len(captured_snaps_host))
                )
                iters = tracer_profile.get("cg_iters", [])
                if iters:
                    per_ss = [
                        sum(iters[s:s + steps_per_ss]) / max(1, len(iters[s:s + steps_per_ss]))
                        for s in range(0, len(iters), steps_per_ss)
                    ]
                    head = "  ".join(f"{v:.1f}" for v in per_ss[:16])
                    tail = (
                        f"  ...  last5: " + "  ".join(f"{v:.1f}" for v in per_ss[-5:])
                        if len(per_ss) > 21 else ""
                    )
                    print(
                        f"[tracer-profile] rank {rank} comp {int(j)}: "
                        f"{len(per_ss)} SS, mean CG iters/step by SS: {head}{tail}",
                        flush=True,
                    )
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
                # Demultiplex: hand each successor the snapshots taken inside
                # its own window. Successors whose windows overlap share the
                # array; both consumers treat corrections as read-only.
                delta_snaps_host = [
                    captured_snaps_host[snap_index_by_step[int(step)]]
                    for step in per_succ_snapshot_steps[int(succ_j)]
                ]
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
        _run_self_check_ladder(
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
            save_callback=self_check_save_callback,
        )

    return (final_states_host if collect_output_snapshots else {}), timing_stats


def _effective_horizon_ss(
    horizon_ss_map: Dict[int, int] | None,
    dst_component: int,
    dst_weight: int,
) -> int:
    """Resolve a per-destination horizon map entry to an explicit trace depth.

    Self-check-ladder semantics: the ladder keys horizons by destination (all
    of a destination's connected pairs extend together)."""
    horizon_ss = 1
    if horizon_ss_map is not None:
        horizon_ss = max(1, int(horizon_ss_map.get(int(dst_component), 1)))
    return min(int(horizon_ss), int(dst_weight))



def _source_on_snapshot_mask(
    pd: PathDef,
    rel_snapshot_steps: list[int],
) -> list[bool]:
    """True for snapshot steps that fall inside a source-on leg of pd.

    The self-check shift is measured with the same source-on-only semantics as
    the accuracy comparisons: source-off connector snapshots are excluded."""
    intervals: list[tuple[int, int, bool]] = []
    acc = 0
    for leg in pd.legs:
        n = int(getattr(leg, "steps", 0))
        intervals.append((acc, acc + n, bool(getattr(leg, "source_on", True))))
        acc += n
    mask: list[bool] = []
    for step in rel_snapshot_steps:
        on = True
        for lo, hi, leg_on in intervals:
            if lo <= int(step) < hi:
                on = leg_on
                break
        mask.append(on)
    return mask


def _self_check_emitters(
    *,
    assigned_comps: List[int],
    succ_map: Dict[int, List[int]],
    self_check_maps: dict,
    correction_horizon_ss_map: Dict[int, int] | None,
    path_def_by_id: Dict[int, PathDef],
) -> set[int]:
    """Owned components that may emit a refinement correction on any rung:
    any component with a successor in the production DAG (horizon extensions)
    or in any rung's new pairs."""
    _ = correction_horizon_ss_map, path_def_by_id
    emitters: set[int] = set()
    for j in assigned_comps:
        if succ_map.get(int(j)):
            emitters.add(int(j))
            continue
        for rung in self_check_maps["rungs"]:
            if rung["component_successors"].get(int(j)) or rung["component_successors"].get(str(j)):
                emitters.add(int(j))
                break
    return emitters


def _run_self_check_ladder(
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
    save_callback=None,
) -> None:
    """
    Iterative self-convergence refinement (a-posteriori, no serial reference).

    Rung k applies the corrections neglected by the state after rung k-1:
      - NEW PAIRS: component dependencies first connected at level_K/gamma^k
        (full corrections over the rung's horizon window);
      - HORIZON EXTENSIONS: every already-connected pair's correction window is
        pushed to max(rung-k DAG horizon, previous horizon + 1 supersegment),
        computing only the extension snapshots at their snapshot offset.
    After superposing each rung, the inter-iteration rel-L2 shift (max/rms) is
    reported: the estimate of the error of the PREVIOUS iterate. save_callback
    (rung_index, states) receives each refined iterate for optional saving.
    """
    t0 = time.perf_counter()
    # Separate communicator: self-check traffic reuses plain component-id tags
    # without colliding with phase-1 messages; the per-rung allreduce also
    # orders the rungs so same-tag messages of different rungs cannot mix.
    comm_sc = comm.Dup()

    def _bridge(src_j: int, dst_j: int, horizon_map):
        # The ladder drives per-DESTINATION horizons (every connected pair of a
        # destination is extended in lockstep), unlike production's per-edge
        # depths; the map is resolved here and passed as an explicit depth.
        return _build_edge_window(
            src_component=int(src_j),
            dst_component=int(dst_j),
            ordered_component_ids=ordered_component_ids,
            component_index=component_index,
            path_def_by_id=path_def_by_id,
            steps_per_ss=steps_per_ss,
            snapshot_stride_steps=snapshot_stride_steps,
            snapshot_steps_by_component=snapshot_steps_by_component,
            horizon_ss=_effective_horizon_ss(
                horizon_map, int(dst_j), int(path_def_by_id[int(dst_j)].weight)
            ),
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

    from mpi4py import MPI

    all_comp_ids = [int(c) for c in ordered_component_ids]
    # Cumulative connectivity (dst -> preds) and per-dst effective horizon,
    # advanced rung by rung; every rank tracks the same global state.
    connected_pred: Dict[int, set[int]] = {
        int(c): set(int(v) for v in production_pred_map.get(int(c), [])) for c in all_comp_ids
    }
    horizon_now: Dict[int, int] = {
        int(c): _effective_horizon_ss(
            production_horizon_ss_map, int(c), int(path_def_by_id[int(c)].weight)
        )
        for c in all_comp_ids
    }

    current = final_states_host
    production_states = final_states_host  # u_0 stays intact: rungs build copies
    estimates: list[tuple[float, float]] = []
    cumulative: list[float] = []
    total_tracers = 0

    for rung_idx, rung in enumerate(self_check_maps["rungs"], start=1):
        rung_pred = {
            int(k): [int(v) for v in vs]
            for k, vs in rung["component_predecessors"].items()
        }
        rung_succ = {
            int(k): [int(v) for v in vs]
            for k, vs in rung["component_successors"].items()
        }
        rung_deep_horizon = {int(k): int(v) for k, v in rung["horizon_ss_map"].items()}
        horizon_only = float(rung.get("level_K", 0.0)) <= 0.0
        step = int(self_check_maps.get("horizon_step_ss", 4))
        horizon_next: Dict[int, int] = {}
        for c in all_comp_ids:
            w = int(path_def_by_id[int(c)].weight)
            if horizon_only:
                # Incremental ladder: each iteration applies `horizon_step_ss`
                # additional supersegments of correction to every connected
                # pair; the decaying shift sequence across iterations is the
                # self-convergence evidence. The step should span the path's
                # local revisit scale (a 1-SS step samples single slices whose
                # magnitudes follow the revisit geometry non-monotonically).
                horizon_next[int(c)] = min(int(w), int(horizon_now[int(c)]) + step)
            else:
                deep_h = _effective_horizon_ss(rung_deep_horizon, int(c), w)
                horizon_next[int(c)] = min(
                    int(w), max(int(deep_h), int(horizon_now[int(c)]) + 1)
                )

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
            # (1) pairs newly connected at this rung: full correction window
            for succ_j in rung_succ.get(int(j), []):
                x0, y0, legs, max_steps, snap_steps, _gap = _bridge(
                    j, succ_j, horizon_next
                )
                _dispatch(j, succ_j, 0, _trace(j, x0, y0, legs, max_steps, snap_steps))
            # (2) horizon extensions of previously connected pairs
            for succ_j in sorted(
                s2 for s2, preds in connected_pred.items() if int(j) in preds
            ):
                if horizon_next[int(succ_j)] <= horizon_now[int(succ_j)]:
                    continue
                x0, y0, legs, max_steps, snap_steps, gap = _bridge(
                    j, succ_j, horizon_next
                )
                old_window = int(gap) + int(steps_per_ss) * int(horizon_now[int(succ_j)])
                offset = sum(1 for st in snap_steps if int(st) < int(old_window))
                ext_steps = [int(st) for st in snap_steps if int(st) >= int(old_window)]
                _dispatch(
                    j,
                    succ_j,
                    offset,
                    _trace(j, x0, y0, legs, max_steps, ext_steps) if ext_steps else [],
                )

        # expected refinement sources for each owned target this rung;
        # channel attribution: "new" = pairs first connected this rung,
        # "ext" = horizon extensions of previously connected pairs.
        local_sq_num = 0.0
        local_sq_den = 0.0
        local_max_rel = 0.0
        local_max_rel_new = 0.0
        local_max_rel_ext = 0.0
        local_n_snaps = 0
        refined_states: Dict[int, List[np.ndarray]] = {}
        for j in assigned_comps:
            new_srcs = set(rung_pred.get(int(j), []))
            srcs = list(new_srcs)
            if horizon_next[int(j)] > horizon_now[int(j)]:
                srcs.extend(int(v) for v in connected_pred.get(int(j), set()))
            srcs = sorted(set(srcs))
            production = current.get(int(j), [])
            refined = [np.array(snap, copy=True) for snap in production]
            # per-channel accumulated deltas for attribution
            chan_new = [np.zeros_like(snap, dtype=np.float64) for snap in production]
            chan_ext = [np.zeros_like(snap, dtype=np.float64) for snap in production]
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
                chan = chan_new if int(pred_j) in new_srcs else chan_ext
                for k2, delta in enumerate(deltas):
                    idx = int(offset) + int(k2)
                    if idx >= len(refined):
                        break
                    refined[idx] = refined[idx] + delta
                    chan[idx] += delta.astype(np.float64)
            refined_states[int(j)] = refined
            rel_steps = _snapshot_steps_for_component(int(j), snapshot_steps_by_component)
            if rel_steps is None:
                rel_steps = list(range(len(production)))
            on_mask = _source_on_snapshot_mask(path_def_by_id[int(j)], rel_steps)
            for snap_idx, (old_snap, new_snap) in enumerate(zip(production, refined)):
                if snap_idx < len(on_mask) and not on_mask[snap_idx]:
                    continue  # source-off connector snapshot: excluded, matching the accuracy metric
                diff = new_snap.astype(np.float64) - old_snap.astype(np.float64)
                den = float(np.linalg.norm(new_snap.astype(np.float64)))
                den_safe = max(den, 1e-30)
                local_max_rel = max(local_max_rel, float(np.linalg.norm(diff)) / den_safe)
                local_max_rel_new = max(
                    local_max_rel_new, float(np.linalg.norm(chan_new[snap_idx])) / den_safe
                )
                local_max_rel_ext = max(
                    local_max_rel_ext, float(np.linalg.norm(chan_ext[snap_idx])) / den_safe
                )
                local_sq_num += float(np.dot(diff.ravel(), diff.ravel()))
                local_sq_den += den * den
                local_n_snaps += 1

        for req in send_reqs:
            req.wait()

        # cumulative shift vs the production state u_0 (the converged value of
        # this quantity IS the production error, by the telescoping identity)
        local_cum_max = 0.0
        for j in assigned_comps:
            prod0 = production_states.get(int(j), [])
            refined = refined_states.get(int(j), [])
            rel_steps = _snapshot_steps_for_component(int(j), snapshot_steps_by_component)
            if rel_steps is None:
                rel_steps = list(range(len(prod0)))
            on_mask = _source_on_snapshot_mask(path_def_by_id[int(j)], rel_steps)
            for snap_idx, (s0, sk) in enumerate(zip(prod0, refined)):
                if snap_idx < len(on_mask) and not on_mask[snap_idx]:
                    continue
                diff = sk.astype(np.float64) - s0.astype(np.float64)
                den = max(float(np.linalg.norm(sk.astype(np.float64))), 1e-30)
                local_cum_max = max(local_cum_max, float(np.linalg.norm(diff)) / den)

        g_max = comm_sc.allreduce(local_max_rel, op=MPI.MAX)
        g_cum_max = comm_sc.allreduce(local_cum_max, op=MPI.MAX)
        g_max_new = comm_sc.allreduce(local_max_rel_new, op=MPI.MAX)
        g_max_ext = comm_sc.allreduce(local_max_rel_ext, op=MPI.MAX)
        g_num = comm_sc.allreduce(local_sq_num, op=MPI.SUM)
        g_den = comm_sc.allreduce(local_sq_den, op=MPI.SUM)
        g_snaps = comm_sc.allreduce(local_n_snaps, op=MPI.SUM)
        g_tracers = comm_sc.allreduce(n_tracers, op=MPI.SUM)
        g_rms = (g_num / g_den) ** 0.5 if g_den > 0.0 else 0.0
        estimates.append((float(g_max), float(g_rms)))
        cumulative.append(float(g_cum_max))
        total_tracers += int(g_tracers)
        if rank == 0:
            rung_label = (
                "horizon-only"
                if float(rung.get("level_K", 0.0)) <= 0.0
                else f"refinement level_K={float(rung['level_K']):.6g} K"
            )
            print(
                f"[self-check] iter {rung_idx}: shift max={g_max:.4e} rms={g_rms:.4e}  "
                f"cumulative-estimate-of-u0 max={g_cum_max:.4e}  "
                f"({rung_label}) "
                f"[new-pairs max={g_max_new:.4e}, horizon-ext max={g_max_ext:.4e}]  "
                f"({int(g_tracers)} correction(s), {int(g_snaps)} snapshot(s))",
                flush=True,
            )

        current = refined_states
        if save_callback is not None:
            save_callback(int(rung_idx), current)

        # advance cumulative state
        for dst, preds in rung_pred.items():
            connected_pred.setdefault(int(dst), set()).update(int(v) for v in preds)
        horizon_now = horizon_next

    comm_sc.Free()
    timing_stats["self_check_seconds"] = time.perf_counter() - t0
    timing_stats["self_check_iterations"] = float(len(estimates))
    timing_stats["self_check_max_rel_l2"] = float(estimates[0][0]) if estimates else 0.0
    timing_stats["self_check_rms_rel_l2"] = float(estimates[0][1]) if estimates else 0.0
    timing_stats["self_check_final_max_rel_l2"] = float(estimates[-1][0]) if estimates else 0.0
    timing_stats["self_check_num_extra_corrections"] = float(total_tracers)
    timing_stats["self_check_cumulative_max_rel_l2"] = (
        float(cumulative[-1]) if cumulative else 0.0
    )
    if rank == 0 and estimates:
        ladder = "  ".join(f"iter{k+1}: {mx:.3e}" for k, (mx, _) in enumerate(estimates))
        print(f"[self-check] ladder (max shift per iteration): {ladder}", flush=True)
        cum = "  ".join(f"iter{k+1}: {c:.3e}" for k, c in enumerate(cumulative))
        print(f"[self-check] cumulative estimate of production error: {cum}", flush=True)
        print(
            "[self-check] ESTIMATED production max rel-L2 (converged cumulative): "
            f"{cumulative[-1]:.4e}",
            flush=True,
        )
