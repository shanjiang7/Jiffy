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
    total_base_work, outgoing_corrections, prefix_sums = _prepare_dp_work_model(
        n_ss=int(n_ss),
        edge_pairs=edge_pairs,
        segments_per_supersegment=int(segments_per_supersegment),
        correction_weight=float(correction_weight),
        cut_depths=cut_depths,
    )
    if int(n_ss) <= 0:
        return 0.0, [], []

    segment_costs = [[0.0 for _ in range(int(n_ss))] for _ in range(int(n_ss))]
    for start_ss in range(int(n_ss)):
        for end_ss in range(int(start_ss), int(n_ss)):
            segment_costs[int(start_ss)][int(end_ss)] = _segment_cost_from_work_model(
                start_ss=int(start_ss),
                end_ss=int(end_ss),
                prefix_sums=prefix_sums,
                outgoing_corrections=outgoing_corrections,
            )

    return total_base_work, outgoing_corrections, segment_costs


def _prepare_dp_work_model(
    *,
    n_ss: int,
    edge_pairs: list[tuple[int, int]],
    segments_per_supersegment: int = 1,
    correction_weight: float = 0.75,
    cut_depths: list[int] | None = None,
) -> tuple[float, list[dict], list[float]]:
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

    return total_base_work, outgoing_corrections, prefix_sums


def _segment_cost_from_work_model(
    *,
    start_ss: int,
    end_ss: int,
    prefix_sums: list[float],
    outgoing_corrections: list[dict],
) -> float:
    base_start = float(prefix_sums[int(start_ss) - 1]) if int(start_ss) > 0 else 0.0
    partition_base = float(prefix_sums[int(end_ss)]) - float(base_start)
    return float(partition_base) + float(
        outgoing_corrections[int(end_ss)]["correction_cost"]
    )


def _compute_exact_dp_partitions(
    *,
    n_items: int,
    num_processors: int,
    segment_costs: list[list[float]],
) -> tuple[list[tuple[int, int]], int, list[list[int]]]:
    if int(n_items) <= 0:
        return [], 0, []
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
    return partitions, int(num_used_processors), cut


def _compute_monotone_dp_partitions(
    *,
    n_items: int,
    num_processors: int,
    prefix_sums: list[float],
    outgoing_corrections: list[dict],
) -> tuple[list[tuple[int, int]], int, list[list[int]], dict]:
    if int(n_items) <= 0:
        return [], 0, [], {
            "algorithm": "monotone_divide_and_conquer",
            "transition_evaluations": 0,
        }

    num_used_processors = min(int(num_processors), int(n_items))
    cut = [[-1 for _ in range(int(n_items))] for _ in range(int(num_used_processors))]
    prev_dp = [float("inf") for _ in range(int(n_items))]
    transition_evaluations = 0

    for end_idx in range(int(n_items)):
        prev_dp[int(end_idx)] = _segment_cost_from_work_model(
            start_ss=0,
            end_ss=int(end_idx),
            prefix_sums=prefix_sums,
            outgoing_corrections=outgoing_corrections,
        )
        cut[0][int(end_idx)] = 0
        transition_evaluations += 1

    for proc_idx in range(1, int(num_used_processors)):
        curr_dp = [float("inf") for _ in range(int(n_items))]

        def compute_range(
            left_end: int,
            right_end: int,
            opt_left: int,
            opt_right: int,
        ) -> None:
            nonlocal transition_evaluations
            if int(left_end) > int(right_end):
                return

            mid_end = (int(left_end) + int(right_end)) // 2
            start_min = max(int(proc_idx), int(opt_left))
            start_max = min(int(mid_end), int(opt_right))

            best_value = float("inf")
            best_start = int(start_min)
            for start_idx in range(int(start_min), int(start_max) + 1):
                prev_value = float(prev_dp[int(start_idx) - 1])
                current_value = _segment_cost_from_work_model(
                    start_ss=int(start_idx),
                    end_ss=int(mid_end),
                    prefix_sums=prefix_sums,
                    outgoing_corrections=outgoing_corrections,
                )
                objective = max(float(prev_value), float(current_value))
                transition_evaluations += 1
                if float(objective) < float(best_value):
                    best_value = float(objective)
                    best_start = int(start_idx)

            curr_dp[int(mid_end)] = float(best_value)
            cut[int(proc_idx)][int(mid_end)] = int(best_start)

            compute_range(
                int(left_end),
                int(mid_end) - 1,
                int(opt_left),
                int(best_start),
            )
            compute_range(
                int(mid_end) + 1,
                int(right_end),
                int(best_start),
                int(opt_right),
            )

        compute_range(
            int(proc_idx),
            int(n_items) - 1,
            int(proc_idx),
            int(n_items) - 1,
        )
        prev_dp = curr_dp

    partitions: list[tuple[int, int]] = []
    end_idx = int(n_items) - 1
    proc_idx = int(num_used_processors) - 1
    while proc_idx >= 0:
        start_idx = int(cut[int(proc_idx)][int(end_idx)])
        partitions.append((int(start_idx), int(end_idx)))
        end_idx = int(start_idx) - 1
        proc_idx -= 1
    partitions.reverse()

    stats = {
        "algorithm": "monotone_divide_and_conquer",
        "transition_evaluations": int(transition_evaluations),
    }
    return partitions, int(num_used_processors), cut, stats


def _summarize_cut_monotonicity(
    *,
    cut_table: list[list[int]],
    n_items: int,
    num_used_processors: int,
    max_examples: int = 20,
) -> dict:
    total_violations = 0
    max_drop = 0
    states_checked = 0
    violations_by_row: dict[int, int] = {}
    examples: list[dict[str, int]] = []

    for proc_idx in range(1, int(num_used_processors)):
        prev_opt: int | None = None
        row_violations = 0
        for end_idx in range(int(proc_idx), int(n_items)):
            opt = int(cut_table[int(proc_idx)][int(end_idx)])
            if opt < 0:
                continue
            if prev_opt is not None:
                states_checked += 1
                if int(opt) < int(prev_opt):
                    row_violations += 1
                    total_violations += 1
                    max_drop = max(int(max_drop), int(prev_opt) - int(opt))
                    if len(examples) < int(max_examples):
                        examples.append(
                            {
                                "processor_index": int(proc_idx),
                                "previous_end_index": int(end_idx) - 1,
                                "end_index": int(end_idx),
                                "previous_opt_start": int(prev_opt),
                                "opt_start": int(opt),
                            }
                        )
            prev_opt = int(opt)
        if row_violations:
            violations_by_row[int(proc_idx)] = int(row_violations)

    return {
        "checked": True,
        "property": "opt[p][j] <= opt[p][j+1]",
        "holds": int(total_violations) == 0,
        "num_rows_checked": max(0, int(num_used_processors) - 1),
        "num_states_checked": int(states_checked),
        "num_violations": int(total_violations),
        "max_drop": int(max_drop),
        "violations_by_processor_index": {
            int(k): int(v) for k, v in violations_by_row.items()
        },
        "examples": examples,
    }


def partition_supersegments_exact_dp(
    n_ss: int,
    edge_pairs: list[tuple[int, int]],
    *,
    num_processors: int,
    segments_per_supersegment: int = 1,
    correction_weight: float = 0.75,
    cut_depths: list[int] | None = None,
    verify_monotonicity: bool = False,
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
        if bool(verify_monotonicity):
            summary["dp_monotonicity"] = {
                "checked": True,
                "property": "opt[p][j] <= opt[p][j+1]",
                "holds": True,
                "num_rows_checked": 0,
                "num_states_checked": 0,
                "num_violations": 0,
                "max_drop": 0,
                "violations_by_processor_index": {},
                "examples": [],
            }
        return summary

    total_base_work, outgoing_corrections, segment_costs = _prepare_exact_dp_cost_data(
        n_ss=int(n_ss),
        edge_pairs=edge_pairs,
        segments_per_supersegment=int(segments_per_supersegment),
        correction_weight=float(correction_weight),
        cut_depths=cut_depths,
    )
    partitions, num_used_processors, cut_table = _compute_exact_dp_partitions(
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

    summary = _build_partition_summary(
        rank_assignments=rank_assignments,
        rank_loads=rank_loads,
        rank_partition_ranges=rank_partition_ranges,
        rank_boundary_correction_costs=rank_boundary_correction_costs,
        rank_boundary_correction_span_ss=rank_boundary_correction_span_ss,
        rank_boundary_correction_mode=rank_boundary_correction_mode,
        total_base_workload=float(total_base_work),
        num_processors=int(num_processors),
    )
    if bool(verify_monotonicity):
        summary["dp_monotonicity"] = _summarize_cut_monotonicity(
            cut_table=cut_table,
            n_items=int(n_ss),
            num_used_processors=int(num_used_processors),
        )
    return summary


def partition_supersegments_monotone_dp(
    n_ss: int,
    edge_pairs: list[tuple[int, int]],
    *,
    num_processors: int,
    segments_per_supersegment: int = 1,
    correction_weight: float = 0.75,
    cut_depths: list[int] | None = None,
    verify_monotonicity: bool = False,
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
        summary["dp_algorithm"] = "monotone_divide_and_conquer"
        summary["dp_transition_evaluations"] = 0
        if bool(verify_monotonicity):
            summary["dp_monotonicity"] = {
                "checked": True,
                "property": "opt[p][j] <= opt[p][j+1]",
                "holds": True,
                "num_rows_checked": 0,
                "num_states_checked": 0,
                "num_violations": 0,
                "max_drop": 0,
                "violations_by_processor_index": {},
                "examples": [],
            }
        return summary

    total_base_work, outgoing_corrections, prefix_sums = _prepare_dp_work_model(
        n_ss=int(n_ss),
        edge_pairs=edge_pairs,
        segments_per_supersegment=int(segments_per_supersegment),
        correction_weight=float(correction_weight),
        cut_depths=cut_depths,
    )
    partitions, num_used_processors, cut_table, dp_stats = _compute_monotone_dp_partitions(
        n_items=int(n_ss),
        num_processors=int(num_processors),
        prefix_sums=prefix_sums,
        outgoing_corrections=outgoing_corrections,
    )

    for rank, (start_ss, end_ss) in enumerate(partitions):
        rank_assignments[int(rank)] = list(range(int(start_ss), int(end_ss) + 1))
        outgoing_correction = outgoing_corrections[int(end_ss)]
        rank_boundary_correction_costs[int(rank)] = float(outgoing_correction["correction_cost"])
        rank_boundary_correction_span_ss[int(rank)] = float(outgoing_correction["correction_span_ss"])
        rank_boundary_correction_mode[int(rank)] = str(outgoing_correction["correction_mode"])
        rank_loads[int(rank)] = _segment_cost_from_work_model(
            start_ss=int(start_ss),
            end_ss=int(end_ss),
            prefix_sums=prefix_sums,
            outgoing_corrections=outgoing_corrections,
        )
        rank_partition_ranges[int(rank)] = {
            "start_ss": int(start_ss),
            "end_ss": int(end_ss),
        }

    summary = _build_partition_summary(
        rank_assignments=rank_assignments,
        rank_loads=rank_loads,
        rank_partition_ranges=rank_partition_ranges,
        rank_boundary_correction_costs=rank_boundary_correction_costs,
        rank_boundary_correction_span_ss=rank_boundary_correction_span_ss,
        rank_boundary_correction_mode=rank_boundary_correction_mode,
        total_base_workload=float(total_base_work),
        num_processors=int(num_processors),
    )
    summary["dp_algorithm"] = str(dp_stats["algorithm"])
    summary["dp_transition_evaluations"] = int(dp_stats["transition_evaluations"])
    summary["dp_num_used_processors"] = int(num_used_processors)
    if bool(verify_monotonicity):
        summary["dp_monotonicity"] = _summarize_cut_monotonicity(
            cut_table=cut_table,
            n_items=int(n_ss),
            num_used_processors=int(num_used_processors),
        )
    return summary


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
    "partition_supersegments_monotone_dp",
    "direct_partition_dag_n1",
]
