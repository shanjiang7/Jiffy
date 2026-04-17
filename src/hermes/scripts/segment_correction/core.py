import time
import cupy as cp
import numpy as np
from typing import Dict, List
from hermes.motion.types import PathDef
from hermes.scripts.outer_solver import OuterContext, run_ss_outer


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


def _build_successor_map(pred_map: Dict[int, int | None]) -> Dict[int, int]:
    return {int(pred): int(comp) for comp, pred in pred_map.items() if pred is not None}


def _snapshot_steps_for_component(
    component_id: int,
    snapshot_steps_by_component: Dict[int, List[int]] | None,
) -> list[int] | None:
    if snapshot_steps_by_component is None:
        return None
    return list(snapshot_steps_by_component.get(int(component_id), []))


def _superpose_snapshots(
    base_snaps: List[np.ndarray],
    delta_snaps: List[np.ndarray],
) -> List[np.ndarray]:
    if not base_snaps:
        return []
    if not delta_snaps:
        return list(base_snaps)
    final_snaps: List[np.ndarray] = []
    for snap_idx, base_snap in enumerate(base_snaps):
        if int(snap_idx) < len(delta_snaps):
            final_snaps.append(base_snap + delta_snaps[int(snap_idx)])
        else:
            final_snaps.append(base_snap)
    return final_snaps


def _correction_stats(
    base_snaps: List[np.ndarray],
    delta_snaps: List[np.ndarray],
) -> dict[str, float]:
    if not delta_snaps:
        return {
            "num_snaps": 0.0,
            "max_abs": 0.0,
            "max_delta_l2": 0.0,
            "max_rel_l2_vs_base": 0.0,
        }

    max_abs = 0.0
    max_delta_l2 = 0.0
    max_rel_l2_vs_base = 0.0
    for snap_idx, delta_snap in enumerate(delta_snaps):
        delta64 = delta_snap.astype(np.float64, copy=False)
        delta_l2 = float(np.linalg.norm(delta64))
        max_delta_l2 = max(max_delta_l2, delta_l2)
        max_abs = max(max_abs, float(np.max(np.abs(delta64))))
        if snap_idx < len(base_snaps):
            base64 = base_snaps[snap_idx].astype(np.float64, copy=False)
            base_l2 = float(np.linalg.norm(base64))
            if base_l2 > 0.0:
                max_rel_l2_vs_base = max(max_rel_l2_vs_base, delta_l2 / base_l2)

    return {
        "num_snaps": float(len(delta_snaps)),
        "max_abs": float(max_abs),
        "max_delta_l2": float(max_delta_l2),
        "max_rel_l2_vs_base": float(max_rel_l2_vs_base),
    }


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


def _print_delta_max_abs_series(
    *,
    prefix: str,
    delta_snaps: List[np.ndarray],
    rel_snapshot_steps: list[int] | None,
    deltaT_K: float = 1.0,
) -> None:
    series = _delta_max_abs_series(delta_snaps, rel_snapshot_steps)
    if not series:
        return
    formatted = ", ".join(
        f"{step}:{value * float(deltaT_K):.4e}" for step, value in series
    )
    print(f"{prefix} max|delta|_K by step [{formatted}]", flush=True)


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
    deltaT_K: float = 1.0,
    collect_output_snapshots: bool = True,
) -> tuple[Dict[int, List[np.ndarray]], dict[str, float]]:
    """
    Pipelined component execution with source-side tracer correction.

    Each rank runs the base simulation of its local components in path order.
    Immediately after finishing a component, it computes the outgoing tracer
    correction for that component's same-layer successor and sends the
    resulting correction snapshots to the successor owner rank. The successor
    rank later superposes those correction snapshots onto its own base
    snapshots. The first component of each layer has no predecessor and
    therefore needs no correction.

    Returns `(final_states_host, timing_stats)`, where `final_states_host` maps
    component id -> list of numpy snapshot arrays when
    `collect_output_snapshots=True`; otherwise it is empty after executing the
    base/correction pipeline and communication.
    """
    _ = world_size
    assigned_comps = sorted(int(j) for j in assigned_comps)
    pred_map = _build_predecessor_map(path_defs, ss_per_layer)
    succ_map = _build_successor_map(pred_map)
    path_def_by_id = {int(pd.component_id): pd for pd in path_defs}
    comp_to_rank = {
        int(comp_id): int(owner_rank)
        for owner_rank, comps in rank_assignments.items()
        for comp_id in comps
    }

    base_states_host: Dict[int, List[np.ndarray]] = {}
    final_states_host: Dict[int, List[np.ndarray]] = {}
    local_correction_snaps: Dict[int, List[np.ndarray]] = {}
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
                lambda snap_u, out=base_snaps_host: out.append(cp.asnumpy(snap_u))
            ) if collect_output_snapshots else None,
        )
        cp.cuda.Stream.null.synchronize()
        timing_stats["base_solve_seconds"] += time.perf_counter() - base_t0
        if collect_output_snapshots:
            base_states_host[j] = base_snaps_host

        succ_j = succ_map.get(int(j))
        if succ_j is None:
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
            snapshot_callback=lambda snap_u, out=delta_snaps_host: out.append(cp.asnumpy(snap_u - ambient_gpu)),
        )
        cp.cuda.Stream.null.synchronize()
        timing_stats["tracer_solve_seconds"] += time.perf_counter() - tracer_t0
        dst_rank = int(comp_to_rank[int(succ_j)])
        if int(dst_rank) == int(rank):
            if collect_output_snapshots:
                local_correction_snaps[int(succ_j)] = delta_snaps_host
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
        pred_j = pred_map.get(int(j))
        if pred_j is None:
            if collect_output_snapshots:
                final_states_host[int(j)] = list(base_states_host[int(j)])
            continue
        src_rank = int(comp_to_rank[int(pred_j)])
        if int(src_rank) == int(rank):
            delta_snaps = local_correction_snaps.get(int(j), [])
        else:
            recv_t0 = time.perf_counter()
            delta_snaps = _recv_correction_chunks(
                comm=comm,
                source_rank=int(src_rank),
                tag=int(j),
            )
            timing_stats["recv_wait_seconds"] += time.perf_counter() - recv_t0
            timing_stats["num_remote_recvs"] += 1.0
        if collect_output_snapshots:
            superpose_t0 = time.perf_counter()
            final_states_host[int(j)] = _superpose_snapshots(base_states_host[int(j)], delta_snaps)
            timing_stats["local_superpose_seconds"] += time.perf_counter() - superpose_t0

    for req in send_reqs:
        send_wait_t0 = time.perf_counter()
        req.wait()
        timing_stats["send_wait_seconds"] += time.perf_counter() - send_wait_t0

    cp.cuda.Stream.null.synchronize()
    timing_stats["post_pipeline_seconds"] = time.perf_counter() - post_t0
    print(f"[rank {rank}] Correction superposition completed.")
    return (final_states_host if collect_output_snapshots else {}), timing_stats
