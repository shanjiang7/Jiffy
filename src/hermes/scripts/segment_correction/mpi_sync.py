from __future__ import annotations

from typing import Dict, List, Tuple
from hermes.utils.dag_utils import Component


def _component_cost(
    comp: Component,
    *,
    mode: str,
    steps_per_ss: int,
    ss_per_layer: int,
    phase2_weight: float,
    apply_phase2: bool,
) -> float:
    if mode == "size":
        return float(comp.size)

    _ = steps_per_ss
    phase1 = float(comp.size)
    has_pred = (int(comp.start_ss) % int(ss_per_layer)) != 0
    phase2 = float(phase2_weight) if apply_phase2 and has_pred else 0.0
    return phase1 + phase2


def balance_components(
    components: List[Component],
    world_size: int,
    *,
    mode: str = "size",
    steps_per_ss: int = 1,
    ss_per_layer: int = 1,
    phase2_weight: float = 0.75,
    apply_phase2: bool = True,
) -> Tuple[Dict[int, List[int]], Dict[int, float]]:
    """
    Greedy makespan balancer with an additive task cost model.

    In cost mode, each task contributes:
        phase1 = number of supersegments in the component
        phase2 = phase2_weight for each non-first-of-layer component

    Components are placed largest-first to the currently lightest rank.

    Returns:
        (rank_assignments, rank_predicted_loads)
    """
    if mode not in {"size", "cost"}:
        raise ValueError(f"Unknown balance mode '{mode}', expected 'size' or 'cost'.")

    ordered_comps = sorted(
        components,
        key=lambda c: int(c.start_ss),
    )
    comp_costs = {
        int(comp.id): _component_cost(
            comp,
            mode=mode,
            steps_per_ss=steps_per_ss,
            ss_per_layer=ss_per_layer,
            phase2_weight=phase2_weight,
            apply_phase2=apply_phase2,
        )
        for comp in ordered_comps
    }
    placement_order = sorted(
        ordered_comps,
        key=lambda c: (-float(comp_costs[int(c.id)]), int(c.start_ss), int(c.id)),
    )

    rank_loads = {i: 0.0 for i in range(world_size)}
    rank_assignments = {i: [] for i in range(world_size)}

    for comp in placement_order:
        comp_id = int(comp.id)
        best_rank = min(
            rank_loads.keys(),
            key=lambda r: (float(rank_loads[r]), int(r)),
        )
        rank_assignments[best_rank].append(comp_id)
        rank_loads[best_rank] += float(comp_costs[comp_id])

    for r in rank_assignments:
        rank_assignments[r].sort()

    return rank_assignments, rank_loads

def rank_for_component(component_id: int, rank_assignments: Dict[int, List[int]]) -> int:
    for rank, comp_ids in rank_assignments.items():
        if component_id in comp_ids:
            return rank
    raise ValueError(f"Component ID {component_id} not assigned to any rank.")
