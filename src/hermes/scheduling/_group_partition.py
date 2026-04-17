from __future__ import annotations

from ._grouping import _boundary_cut_correction_from_depth, _compute_cut_depths


def _build_partition_summary(
    *,
    rank_assignments: dict[int, list[int]],
    rank_loads: dict[int, float],
    rank_partition_ranges: dict[int, dict | None],
    rank_boundary_correction_costs: dict[int, float],
    rank_boundary_correction_span_ss: dict[int, float],
    rank_boundary_correction_mode: dict[int, str],
    total_base_workload: float,
    num_processors: int,
) -> dict:
    loads = [float(rank_loads[rank]) for rank in sorted(rank_loads)]
    mean_load = sum(loads) / max(len(loads), 1)
    max_load = max(loads, default=0.0)
    balance_index = (max_load / mean_load) if mean_load > 0.0 else 0.0
    return {
        "num_processors": int(num_processors),
        "rank_assignments": {int(rank): list(items) for rank, items in rank_assignments.items()},
        "rank_loads": {int(rank): float(load) for rank, load in rank_loads.items()},
        "rank_partition_ranges": rank_partition_ranges,
        "rank_boundary_correction_costs": {
            int(rank): float(cost) for rank, cost in rank_boundary_correction_costs.items()
        },
        "rank_boundary_correction_span_ss": {
            int(rank): float(span) for rank, span in rank_boundary_correction_span_ss.items()
        },
        "rank_boundary_correction_mode": {
            int(rank): str(mode) for rank, mode in rank_boundary_correction_mode.items()
        },
        "total_boundary_correction_cost": float(sum(rank_boundary_correction_costs.values())),
        "total_base_workload": float(total_base_workload),
        "max_rank_workload": float(max_load),
        "mean_rank_workload": float(mean_load),
        "load_balance_index": float(balance_index),
    }


def _prepare_exact_dp_cost_data(
    *,
    n_ss: int,
    edge_pairs: list[tuple[int, int]],
    segments_per_supersegment: int = 1,
    correction_weight: float = 0.75,
    cut_depths: list[int] | None = None,
) -> tuple[float, list[dict], list[list[float]]]:
    if int(n_ss) <= 0:
        return 0.0, [], []

    base_weight_per_ss = float(int(segments_per_supersegment))
    prefix_sums: list[float] = []
    running = 0.0
    for _ in range(int(n_ss)):
        running += float(base_weight_per_ss)
        prefix_sums.append(float(running))

    total_base_work = float(prefix_sums[-1])
    resolved_cut_depths = (
        list(cut_depths) if cut_depths is not None else _compute_cut_depths(int(n_ss), edge_pairs)
    )

    outgoing_corrections: list[dict] = []
    for end_ss in range(int(n_ss)):
        if int(end_ss) == int(n_ss) - 1:
            outgoing_corrections.append(
                _boundary_cut_correction_from_depth(
                    cut_depth=0,
                    segments_per_supersegment=int(segments_per_supersegment),
                    correction_weight=float(correction_weight),
                )
            )
            continue
        boundary_depth = (
            int(resolved_cut_depths[int(end_ss)])
            if 0 <= int(end_ss) < len(resolved_cut_depths)
            else 0
        )
        outgoing_corrections.append(
            _boundary_cut_correction_from_depth(
                cut_depth=int(boundary_depth),
                segments_per_supersegment=int(segments_per_supersegment),
                correction_weight=float(correction_weight),
            )
        )

    segment_costs = [[0.0 for _ in range(int(n_ss))] for _ in range(int(n_ss))]
    for start_ss in range(int(n_ss)):
        base_start = float(prefix_sums[int(start_ss) - 1]) if int(start_ss) > 0 else 0.0
        for end_ss in range(int(start_ss), int(n_ss)):
            partition_base = float(prefix_sums[int(end_ss)]) - float(base_start)
            outgoing_correction = outgoing_corrections[int(end_ss)]
            segment_costs[int(start_ss)][int(end_ss)] = float(partition_base) + float(
                outgoing_correction["correction_cost"]
            )

    return total_base_work, outgoing_corrections, segment_costs


def _compute_exact_dp_partitions(
    *,
    n_items: int,
    num_processors: int,
    segment_costs: list[list[float]],
) -> tuple[list[tuple[int, int]], int]:
    if int(n_items) <= 0:
        return [], 0
    num_used_processors = min(int(num_processors), int(n_items))
    dp = [[float("inf") for _ in range(int(n_items))] for _ in range(int(num_used_processors))]
    cut = [[-1 for _ in range(int(n_items))] for _ in range(int(num_used_processors))]

    for end_idx in range(int(n_items)):
        dp[0][int(end_idx)] = float(segment_costs[0][int(end_idx)])
        cut[0][int(end_idx)] = 0

    for proc_idx in range(1, int(num_used_processors)):
        for end_idx in range(int(proc_idx), int(n_items)):
            best_value = float("inf")
            best_start = -1
            for start_idx in range(int(proc_idx), int(end_idx) + 1):
                prev_value = float(dp[int(proc_idx) - 1][int(start_idx) - 1])
                current_value = float(segment_costs[int(start_idx)][int(end_idx)])
                objective = max(float(prev_value), float(current_value))
                if float(objective) < float(best_value):
                    best_value = float(objective)
                    best_start = int(start_idx)
            dp[int(proc_idx)][int(end_idx)] = float(best_value)
            cut[int(proc_idx)][int(end_idx)] = int(best_start)

    partitions: list[tuple[int, int]] = []
    end_idx = int(n_items) - 1
    proc_idx = int(num_used_processors) - 1
    while proc_idx >= 0:
        start_idx = int(cut[int(proc_idx)][int(end_idx)])
        partitions.append((int(start_idx), int(end_idx)))
        end_idx = int(start_idx) - 1
        proc_idx -= 1
    partitions.reverse()
    return partitions, int(num_used_processors)


def partition_supersegments_exact_dp(
    n_ss: int,
    edge_pairs: list[tuple[int, int]],
    *,
    num_processors: int,
    segments_per_supersegment: int = 1,
    correction_weight: float = 0.75,
    cut_depths: list[int] | None = None,
) -> dict:
    if int(n_ss) < 0:
        raise ValueError("n_ss must be >= 0")
    if int(num_processors) < 1:
        raise ValueError("num_processors must be >= 1")

    rank_assignments = {rank: [] for rank in range(int(num_processors))}
    rank_loads = {rank: 0.0 for rank in range(int(num_processors))}
    rank_partition_ranges = {rank: None for rank in range(int(num_processors))}
    rank_boundary_correction_costs = {rank: 0.0 for rank in range(int(num_processors))}
    rank_boundary_correction_span_ss = {rank: 0.0 for rank in range(int(num_processors))}
    rank_boundary_correction_mode = {rank: "none" for rank in range(int(num_processors))}

    if int(n_ss) == 0:
        return _build_partition_summary(
            rank_assignments=rank_assignments,
            rank_loads=rank_loads,
            rank_partition_ranges=rank_partition_ranges,
            rank_boundary_correction_costs=rank_boundary_correction_costs,
            rank_boundary_correction_span_ss=rank_boundary_correction_span_ss,
            rank_boundary_correction_mode=rank_boundary_correction_mode,
            total_base_workload=0.0,
            num_processors=int(num_processors),
        )

    total_base_work, outgoing_corrections, segment_costs = _prepare_exact_dp_cost_data(
        n_ss=int(n_ss),
        edge_pairs=edge_pairs,
        segments_per_supersegment=int(segments_per_supersegment),
        correction_weight=float(correction_weight),
        cut_depths=cut_depths,
    )
    partitions, _ = _compute_exact_dp_partitions(
        n_items=int(n_ss),
        num_processors=int(num_processors),
        segment_costs=segment_costs,
    )

    for rank, (start_ss, end_ss) in enumerate(partitions):
        rank_assignments[int(rank)] = list(range(int(start_ss), int(end_ss) + 1))
        outgoing_correction = outgoing_corrections[int(end_ss)]
        rank_boundary_correction_costs[int(rank)] = float(outgoing_correction["correction_cost"])
        rank_boundary_correction_span_ss[int(rank)] = float(outgoing_correction["correction_span_ss"])
        rank_boundary_correction_mode[int(rank)] = str(outgoing_correction["correction_mode"])
        rank_loads[int(rank)] = float(segment_costs[int(start_ss)][int(end_ss)])
        rank_partition_ranges[int(rank)] = {
            "start_ss": int(start_ss),
            "end_ss": int(end_ss),
        }

    return _build_partition_summary(
        rank_assignments=rank_assignments,
        rank_loads=rank_loads,
        rank_partition_ranges=rank_partition_ranges,
        rank_boundary_correction_costs=rank_boundary_correction_costs,
        rank_boundary_correction_span_ss=rank_boundary_correction_span_ss,
        rank_boundary_correction_mode=rank_boundary_correction_mode,
        total_base_workload=float(total_base_work),
        num_processors=int(num_processors),
    )


def direct_partition_dag_n1(
    n_ss: int,
    edge_pairs: list[tuple[int, int]],
    *,
    num_processors: int,
    correction_weight: float = 0.75,
    cut_depths: list[int] | None = None,
) -> dict:
    if int(n_ss) < 0:
        raise ValueError("n_ss must be >= 0")
    if int(num_processors) < 1:
        raise ValueError("num_processors must be >= 1")

    rank_assignments = {rank: [] for rank in range(int(num_processors))}
    rank_loads = {rank: 0.0 for rank in range(int(num_processors))}
    rank_partition_ranges = {rank: None for rank in range(int(num_processors))}
    rank_boundary_correction_costs = {rank: 0.0 for rank in range(int(num_processors))}
    rank_boundary_correction_span_ss = {rank: 0.0 for rank in range(int(num_processors))}
    rank_boundary_correction_mode = {rank: "none" for rank in range(int(num_processors))}

    if int(n_ss) == 0:
        summary = _build_partition_summary(
            rank_assignments=rank_assignments,
            rank_loads=rank_loads,
            rank_partition_ranges=rank_partition_ranges,
            rank_boundary_correction_costs=rank_boundary_correction_costs,
            rank_boundary_correction_span_ss=rank_boundary_correction_span_ss,
            rank_boundary_correction_mode=rank_boundary_correction_mode,
            total_base_workload=0.0,
            num_processors=int(num_processors),
        )
        summary["total_boundary_correction_span_ss"] = 0.0
        return summary

    base_size = int(n_ss) // int(num_processors)
    remainder = int(n_ss) % int(num_processors)
    resolved_cut_depths = (
        list(cut_depths) if cut_depths is not None else _compute_cut_depths(int(n_ss), edge_pairs)
    )
    partition_ranges: list[tuple[int, int] | None] = []
    cursor = 0
    for rank in range(int(num_processors)):
        part_size = int(base_size) + (1 if int(rank) < int(remainder) else 0)
        if int(part_size) <= 0:
            partition_ranges.append(None)
            continue
        start_ss = int(cursor)
        end_ss = int(cursor) + int(part_size) - 1
        partition_ranges.append((int(start_ss), int(end_ss)))
        cursor = int(end_ss) + 1

    for rank, part_range in enumerate(partition_ranges):
        if part_range is None:
            continue
        start_ss, end_ss = part_range
        base_segments = float(int(end_ss) - int(start_ss) + 1)
        rank_assignments[int(rank)] = list(range(int(start_ss), int(end_ss) + 1))
        rank_partition_ranges[int(rank)] = {
            "start_ss": int(start_ss),
            "end_ss": int(end_ss),
        }

        boundary_depth = int(resolved_cut_depths[int(end_ss)]) if int(end_ss) < int(n_ss) - 1 else 0
        boundary_correction = _boundary_cut_correction_from_depth(
            cut_depth=int(boundary_depth),
            segments_per_supersegment=1,
            correction_weight=float(correction_weight),
        )
        rank_boundary_correction_span_ss[int(rank)] = float(boundary_correction["correction_span_ss"])
        rank_boundary_correction_costs[int(rank)] = float(boundary_correction["correction_cost"])
        rank_boundary_correction_mode[int(rank)] = str(boundary_correction["correction_mode"])
        rank_loads[int(rank)] = float(base_segments) + float(boundary_correction["correction_cost"])

    summary = _build_partition_summary(
        rank_assignments=rank_assignments,
        rank_loads=rank_loads,
        rank_partition_ranges=rank_partition_ranges,
        rank_boundary_correction_costs=rank_boundary_correction_costs,
        rank_boundary_correction_span_ss=rank_boundary_correction_span_ss,
        rank_boundary_correction_mode=rank_boundary_correction_mode,
        total_base_workload=float(n_ss),
        num_processors=int(num_processors),
    )
    summary["total_boundary_correction_span_ss"] = float(sum(rank_boundary_correction_span_ss.values()))
    return summary


__all__ = [
    "partition_supersegments_exact_dp",
    "direct_partition_dag_n1",
]
