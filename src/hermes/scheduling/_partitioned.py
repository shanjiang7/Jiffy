from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from hermes.DAG.dependency import LookupRuntime
from hermes.pipelines.components import compute_dag_and_components, export_dag_results
from hermes.pipelines.config import PipelineConfig
from hermes.scheduling._group_partition import (
    partition_supersegments_exact_dp,
    partition_supersegments_monotone_dp,
    direct_partition_dag_n1,
)
from hermes.scheduling._grouping import _compute_cut_depths
from hermes.utils.dag_utils import Component, write_components_csv
from hermes.utils.path_utils import build_path_defs_from_components
from hermes.utils.viz_utils import plot_components


@dataclass(frozen=True)
class DAGStageResult:
    pipeline_cfg: PipelineConfig
    dag_result: object
    edge_pairs: list[tuple[int, int]]
    cut_depths: list[int]
    n_ss: int
    effective_step_nd: float


def _range_kind_depth(
    start_ss: int,
    end_ss: int,
    edge_pairs: list[tuple[int, int]],
) -> tuple[str, int]:
    depth = max(
        (
            int(dst) - int(src)
            for src, dst in edge_pairs
            if int(start_ss) <= int(src) and int(dst) <= int(end_ss)
        ),
        default=0,
    )
    if int(depth) > 1:
        return "coupled", int(depth)
    return "trivial", 1


def _build_correction_horizon_ss_map(
    runtime_components: list[Component],
    *,
    cut_depths: list[int],
    ss_per_layer: int,
) -> dict[int, int]:
    ordered = sorted(runtime_components, key=lambda c: int(c.start_ss))
    horizon_map: dict[int, int] = {}
    for comp in ordered:
        start_ss = int(comp.start_ss)
        if start_ss <= 0 or (int(start_ss) % int(ss_per_layer)) == 0:
            horizon_map[int(comp.id)] = 0
            continue
        boundary = int(start_ss) - 1
        cut_depth = int(cut_depths[int(boundary)]) if 0 <= int(boundary) < len(cut_depths) else 0
        remaining_layer_ss = int(ss_per_layer) - (int(start_ss) % int(ss_per_layer))
        horizon_map[int(comp.id)] = max(
            0,
            min(
                int(cut_depth),
                int(comp.size),
                int(remaining_layer_ss),
            ),
        )
    return horizon_map


def _build_component_dependency_maps(
    runtime_components: list[Component],
    edge_pairs: list[tuple[int, int]],
) -> tuple[dict[int, list[int]], dict[int, list[int]], list[dict[str, int]]]:
    """Collapse supersegment DAG edges into inter-runtime-component edges."""
    ss_to_component: dict[int, int] = {}
    for comp in runtime_components:
        for ss_idx in range(int(comp.start_ss), int(comp.end_ss) + 1):
            ss_to_component[int(ss_idx)] = int(comp.id)

    predecessor_sets: dict[int, set[int]] = {
        int(comp.id): set() for comp in runtime_components
    }
    successor_sets: dict[int, set[int]] = {
        int(comp.id): set() for comp in runtime_components
    }
    component_edge_set: set[tuple[int, int]] = set()

    for src_ss, dst_ss in edge_pairs:
        src_comp = ss_to_component.get(int(src_ss))
        dst_comp = ss_to_component.get(int(dst_ss))
        if src_comp is None or dst_comp is None or int(src_comp) == int(dst_comp):
            continue
        edge = (int(src_comp), int(dst_comp))
        if edge in component_edge_set:
            continue
        component_edge_set.add(edge)
        successor_sets[int(src_comp)].add(int(dst_comp))
        predecessor_sets[int(dst_comp)].add(int(src_comp))

    component_predecessors = {
        int(comp_id): sorted(int(pred) for pred in preds)
        for comp_id, preds in predecessor_sets.items()
    }
    component_successors = {
        int(comp_id): sorted(int(succ) for succ in succs)
        for comp_id, succs in successor_sets.items()
    }
    component_dependency_edges = [
        {"src_component": int(src), "dst_component": int(dst)}
        for src, dst in sorted(component_edge_set)
    ]
    return component_predecessors, component_successors, component_dependency_edges


def _build_dag_stage(
    *,
    solver_mode: str,
    path_config_path: Path,
    num_layers_override: int | None,
    dt_s: float,
    rc,
    phys,
    float_type,
    solver_velocity_mps: float,
    export_outputs: bool,
    out_dir: Path,
) -> DAGStageResult | None:
    pipeline_cfg = PipelineConfig.from_ini(
        path_config_path,
        num_layers=num_layers_override,
    )
    pipeline_cfg = replace(pipeline_cfg, segments_per_supersegment=1)
    effective_step_nd = pipeline_cfg.compute_effective_motion_step_nd(
        dt_s=dt_s,
        solver_velocity_mps=solver_velocity_mps,
    )
    pipeline_cfg = pipeline_cfg.with_solver_motion(
        dt_s=dt_s,
        solver_velocity_mps=solver_velocity_mps,
    )
    lookup_runtime = LookupRuntime(
        rc=rc,
        phys=phys,
        float_type=float_type,
        solver_mode=str(solver_mode),
        source_substeps=pipeline_cfg.dependency.mock_numerical_source_steps,
    )
    dag_result = compute_dag_and_components(pipeline_cfg, lookup_runtime=lookup_runtime)
    if dag_result is None:
        return None

    if export_outputs:
        export_dag_results(result=dag_result, out_dir=out_dir)

    edge_pairs = [(int(e.src), int(e.dst)) for e in dag_result.edges]
    return DAGStageResult(
        pipeline_cfg=pipeline_cfg,
        dag_result=dag_result,
        edge_pairs=edge_pairs,
        cut_depths=_compute_cut_depths(len(dag_result.supersegments), edge_pairs),
        n_ss=len(dag_result.supersegments),
        effective_step_nd=float(effective_step_nd),
    )


def _build_exact_dp_stage(
    *,
    world_size: int,
    correction_weight: float,
    dag_stage: DAGStageResult,
    verify_dp_monotonicity: bool,
) -> tuple[dict, dict[int, tuple[int, int] | None]]:
    partition_summary = partition_supersegments_exact_dp(
        int(dag_stage.n_ss),
        edge_pairs=dag_stage.edge_pairs,
        num_processors=int(world_size),
        segments_per_supersegment=1,
        correction_weight=float(correction_weight),
        cut_depths=dag_stage.cut_depths,
        verify_monotonicity=bool(verify_dp_monotonicity),
    )
    rank_ranges = _uniform_rank_ranges(partition_summary, int(world_size))
    return partition_summary, rank_ranges


def _build_monotone_dp_stage(
    *,
    world_size: int,
    correction_weight: float,
    dag_stage: DAGStageResult,
    verify_dp_monotonicity: bool,
) -> tuple[dict, dict[int, tuple[int, int] | None]]:
    partition_summary = partition_supersegments_monotone_dp(
        int(dag_stage.n_ss),
        edge_pairs=dag_stage.edge_pairs,
        num_processors=int(world_size),
        segments_per_supersegment=1,
        correction_weight=float(correction_weight),
        cut_depths=dag_stage.cut_depths,
        verify_monotonicity=bool(verify_dp_monotonicity),
    )
    rank_ranges = _uniform_rank_ranges(partition_summary, int(world_size))
    return partition_summary, rank_ranges


def _uniform_rank_ranges(partition_summary: dict, world_size: int) -> dict[int, tuple[int, int] | None]:
    ranges: dict[int, tuple[int, int] | None] = {}
    for rank in range(int(world_size)):
        info = partition_summary["rank_partition_ranges"].get(int(rank))
        if info is None:
            ranges[int(rank)] = None
            continue
        ranges[int(rank)] = (int(info["start_ss"]), int(info["end_ss"]))
    return ranges


def _build_partition_stage(
    *,
    planner_mode: str,
    world_size: int,
    correction_weight: float,
    dag_stage: DAGStageResult,
    verify_dp_monotonicity: bool,
) -> tuple[dict, dict[int, tuple[int, int] | None]]:
    if str(planner_mode) == "uniform":
        partition_summary = direct_partition_dag_n1(
            int(dag_stage.n_ss),
            dag_stage.edge_pairs,
            num_processors=int(world_size),
            correction_weight=float(correction_weight),
            cut_depths=dag_stage.cut_depths,
        )
        return partition_summary, _uniform_rank_ranges(partition_summary, int(world_size))

    if str(planner_mode) == "exact_dp":
        partition_summary, rank_ranges = _build_exact_dp_stage(
            world_size=int(world_size),
            correction_weight=float(correction_weight),
            dag_stage=dag_stage,
            verify_dp_monotonicity=bool(verify_dp_monotonicity),
        )
        return partition_summary, rank_ranges

    if str(planner_mode) == "dp_monotonicity":
        partition_summary, rank_ranges = _build_monotone_dp_stage(
            world_size=int(world_size),
            correction_weight=float(correction_weight),
            dag_stage=dag_stage,
            verify_dp_monotonicity=bool(verify_dp_monotonicity),
        )
        return partition_summary, rank_ranges

    raise ValueError(
        f"Unknown planner_mode '{planner_mode}', expected 'uniform', 'exact_dp', "
        "or 'dp_monotonicity'."
    )


def _build_runtime_components_and_assignments(
    *,
    rank_ranges: dict[int, tuple[int, int] | None],
    edge_pairs: list[tuple[int, int]],
    world_size: int,
) -> tuple[list[Component], dict[int, list[int]]]:
    owned_ranges = [
        (int(rank), int(start_ss), int(end_ss))
        for rank, part in rank_ranges.items()
        if part is not None
        for start_ss, end_ss in [part]
    ]
    owned_ranges.sort(key=lambda item: (int(item[1]), int(item[2]), int(item[0])))

    runtime_components: list[Component] = []
    rank_assignments = {int(rank): [] for rank in range(int(world_size))}
    for comp_id, (rank, start_ss, end_ss) in enumerate(owned_ranges):
        kind, depth = _range_kind_depth(int(start_ss), int(end_ss), edge_pairs)
        runtime_components.append(
            Component(
                id=int(comp_id),
                start_ss=int(start_ss),
                end_ss=int(end_ss),
                depth=int(depth),
                kind=str(kind),
            )
        )
        rank_assignments[int(rank)].append(int(comp_id))
    return runtime_components, rank_assignments


def build_partitioned_runtime_plan(
    *,
    planner_mode: str,
    correction_weight: float,
    solver_mode: str,
    world_size: int,
    path_config_path: Path,
    out_dir: Path,
    num_layers_override: int | None,
    dt_s: float,
    rc,
    phys,
    float_type,
    solver_velocity_mps: float,
    export_outputs: bool = True,
    verify_dp_monotonicity: bool = False,
):
    dag_stage = _build_dag_stage(
        solver_mode=str(solver_mode),
        path_config_path=path_config_path,
        num_layers_override=num_layers_override,
        dt_s=dt_s,
        rc=rc,
        phys=phys,
        float_type=float_type,
        solver_velocity_mps=float(solver_velocity_mps),
        export_outputs=bool(export_outputs),
        out_dir=out_dir,
    )
    if dag_stage is None:
        return None

    partition_summary, rank_ranges = _build_partition_stage(
        planner_mode=str(planner_mode),
        world_size=int(world_size),
        correction_weight=float(correction_weight),
        dag_stage=dag_stage,
        verify_dp_monotonicity=bool(verify_dp_monotonicity),
    )

    runtime_components, rank_assignments = _build_runtime_components_and_assignments(
        rank_ranges=rank_ranges,
        edge_pairs=dag_stage.edge_pairs,
        world_size=int(world_size),
    )
    if not runtime_components:
        return None

    if export_outputs:
        write_components_csv(runtime_components, out_dir / "runtime_components.csv")
        plot_components(
            runtime_components,
            dag_stage.edge_pairs,
            int(dag_stage.n_ss),
            out_path=out_dir / "runtime_components.png",
        )

    path_defs = build_path_defs_from_components(
        supersegments=dag_stage.dag_result.supersegments,
        components=runtime_components,
        dt_s=dt_s,
        len_scale=phys.len_scale,
        coord_scale=dag_stage.dag_result.len_scale / phys.len_scale,
    )
    correction_horizon_ss_map = _build_correction_horizon_ss_map(
        runtime_components,
        cut_depths=dag_stage.cut_depths,
        ss_per_layer=int(dag_stage.dag_result.ss_per_layer),
    )
    component_predecessors, component_successors, component_dependency_edges = (
        _build_component_dependency_maps(
            runtime_components=runtime_components,
            edge_pairs=dag_stage.edge_pairs,
        )
    )

    return {
        "planner_mode": str(planner_mode),
        "global_max_cut_depth": int(max(dag_stage.cut_depths, default=0)),
        "correction_weight": float(correction_weight),
        "partition_summary": partition_summary,
        "path_defs": path_defs,
        "rank_assignments": rank_assignments,
        "steps_per_ss": int(dag_stage.dag_result.unit_steps),
        "effective_step_nd": float(dag_stage.effective_step_nd),
        "num_layers": int(dag_stage.dag_result.num_layers),
        "ss_per_layer": int(dag_stage.dag_result.ss_per_layer),
        "rank_pred_loads": {
            int(rank): float(load)
            for rank, load in partition_summary["rank_loads"].items()
        },
        "correction_horizon_ss_map": correction_horizon_ss_map,
        "component_predecessors": component_predecessors,
        "component_successors": component_successors,
        "component_dependency_edges": component_dependency_edges,
        "runtime_components": runtime_components,
        "source_components": list(dag_stage.dag_result.components),
        "split_records": [],
        "segments_per_supersegment": 1,
    }


__all__ = ["build_partitioned_runtime_plan"]
