from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

from hermes.DAG.dependency import LookupRuntime
from hermes.pipelines.components import compute_dag_and_components, export_dag_results
from hermes.pipelines.config import PipelineConfig
from hermes.scheduling._group_partition import (
    partition_supersegments_exact_dp,
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


def _build_correction_horizon_by_edge(
    runtime_components: list[Component],
    *,
    edge_pairs: list[tuple[int, int]],
    ss_per_layer: int,
) -> dict[tuple[int, int], int]:
    """Per-edge correction horizon: how far into the destination each source's
    retained influence actually reaches, in supersegments.

    The per-component map (:func:`_build_correction_horizon_ss_map`) keys the
    horizon by destination alone, so it equals the reach of the destination's
    *strongest* predecessor -- the adjacent one, whose deposit is closest and so
    couples deepest. A distant predecessor's influence dies sooner, yet was
    being traced to that same full depth. Keying by the (src, dst) edge and the
    deepest retained supersegment link between them lets each source trace only
    as far as it genuinely influences the destination. Nothing above the DAG's
    epsilon threshold is dropped: beyond the deepest retained edge there is, by
    the retention rule, no edge because the influence is already below epsilon.
    """
    ss_to_component: dict[int, Component] = {}
    for comp in runtime_components:
        for ss_idx in range(int(comp.start_ss), int(comp.end_ss) + 1):
            ss_to_component[int(ss_idx)] = comp

    # Deepest destination supersegment reached by any edge of each component
    # pair. edge_pairs is the retained SS-level DAG, so this is exactly the
    # reach the threshold kept.
    edge_reach_ss: dict[tuple[int, int], int] = {}
    for src_ss, dst_ss in edge_pairs:
        src_comp = ss_to_component.get(int(src_ss))
        dst_comp = ss_to_component.get(int(dst_ss))
        if src_comp is None or dst_comp is None or int(src_comp.id) == int(dst_comp.id):
            continue
        key = (int(src_comp.id), int(dst_comp.id))
        edge_reach_ss[key] = max(int(edge_reach_ss.get(key, -1)), int(dst_ss))

    comp_by_id = {int(c.id): c for c in runtime_components}
    horizon_by_edge: dict[tuple[int, int], int] = {}
    for (src_id, dst_id), deepest_dst_ss in edge_reach_ss.items():
        dst = comp_by_id[int(dst_id)]
        start_ss = int(dst.start_ss)
        # Same layer-boundary guard as the per-component map: corrections do not
        # cross into a component that starts a layer.
        if start_ss <= 0 or (start_ss % int(ss_per_layer)) == 0:
            horizon_by_edge[(int(src_id), int(dst_id))] = 0
            continue
        reach_ss = int(deepest_dst_ss) - start_ss + 1
        remaining_layer_ss = int(ss_per_layer) - (start_ss % int(ss_per_layer))
        horizon_by_edge[(int(src_id), int(dst_id))] = max(
            0, min(int(reach_ss), int(dst.size), int(remaining_layer_ss))
        )
    return horizon_by_edge


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
    path_complexity_report: bool = False,
    dependency_level_K_override: float | None = None,
) -> DAGStageResult | None:
    pipeline_cfg = PipelineConfig.from_ini(
        path_config_path,
        num_layers=num_layers_override,
    )
    if dependency_level_K_override is not None:
        if float(dependency_level_K_override) <= 0.0:
            raise ValueError("dependency_level_K_override must be > 0.")
        pipeline_cfg = replace(
            pipeline_cfg,
            dependency=replace(
                pipeline_cfg.dependency,
                model=replace(
                    pipeline_cfg.dependency.model,
                    level_K=float(dependency_level_K_override),
                ),
            ),
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
        source_on_steps=pipeline_cfg.dependency.lookup_source_on_steps,
        source_substeps=pipeline_cfg.dependency.mock_numerical_source_steps,
    )
    dag_result = compute_dag_and_components(
        pipeline_cfg,
        lookup_runtime=lookup_runtime,
        path_complexity_report=bool(path_complexity_report),
    )
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


# DP planners share one call signature; "uniform" differs (no monotonicity check).
# exact_dp delegates internally to the crossing-point search above the dense
# limit (see _group_partition.py), so it is the only DP mode exposed.
_DP_PARTITIONERS = {
    "exact_dp": partition_supersegments_exact_dp,
}


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
    partition_t0 = time.perf_counter()
    mode = str(planner_mode)
    if mode == "uniform":
        partition_summary = direct_partition_dag_n1(
            int(dag_stage.n_ss),
            dag_stage.edge_pairs,
            num_processors=int(world_size),
            correction_weight=float(correction_weight),
            cut_depths=dag_stage.cut_depths,
            ss_per_layer=int(dag_stage.dag_result.ss_per_layer),
        )
    elif mode in _DP_PARTITIONERS:
        partition_summary = _DP_PARTITIONERS[mode](
            int(dag_stage.n_ss),
            edge_pairs=dag_stage.edge_pairs,
            num_processors=int(world_size),
            segments_per_supersegment=1,
            correction_weight=float(correction_weight),
            cut_depths=dag_stage.cut_depths,
            verify_monotonicity=bool(verify_dp_monotonicity),
            ss_per_layer=int(dag_stage.dag_result.ss_per_layer),
        )
    else:
        raise ValueError(
            f"Unknown planner_mode '{planner_mode}', expected 'uniform', 'exact_dp', "
            "or 'uniform'."
        )
    partition_summary["partition_mode"] = mode
    partition_summary["partition_seconds"] = float(time.perf_counter() - partition_t0)
    return partition_summary, _uniform_rank_ranges(partition_summary, int(world_size))


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


def _build_self_check_maps(
    *,
    gamma: float,
    iterations: int,
    mode: str,
    horizon_step_ss: int,
    production_level_K: float,
    production_pred_map: dict[int, list[int]],
    runtime_components,
    dag_stage_kwargs: dict,
) -> dict:
    """
    Refinement ladder for the self-convergence error estimate.

    Rung k (k = 1..iterations) rebuilds the dependency DAG at the tighter
    threshold level_K / gamma**k over the SAME runtime components. Each rung
    records only the component pairs newly connected relative to the previous
    rung (rung 0 = the production DAG) plus the rung's correction-horizon map.
    The runtime applies the rungs in order: corrections for each rung's new
    pairs and one-supersegment horizon extensions of all previously connected
    pairs, reporting the inter-iteration rel-L2 shift as the error estimate.
    """
    if float(gamma) <= 1.0:
        raise ValueError("self-check gamma must be > 1.")
    if int(iterations) < 1:
        raise ValueError("self-check iterations must be >= 1.")
    mode = str(mode).strip().lower()
    if mode not in {"horizon", "full"}:
        raise ValueError(f"self-check mode must be 'horizon' or 'full', got {mode!r}.")
    if mode == "horizon":
        # Horizon-only ladder: every rung extends each connected pair's
        # correction window by one supersegment. No deep DAGs, no new
        # influence-radius lookups - the cheap production estimator. Empirical
        # basis: on the straight, hybrid and Bull studies the entire measured
        # shift was the horizon-extension channel (new-pair contributions were
        # at numerical-noise level with the validated chord-lookup DAG).
        return {
            "gamma": float(gamma),
            "iterations": int(iterations),
            "mode": "horizon",
            "horizon_step_ss": int(horizon_step_ss),
            "rungs": [
                {
                    "level_K": 0.0,
                    "num_new_pairs": 0,
                    "component_predecessors": {},
                    "component_successors": {},
                    "horizon_ss_map": {},
                }
                for _ in range(int(iterations))
            ],
        }
    prev_pred = {int(k): sorted(int(v) for v in vs) for k, vs in production_pred_map.items()}
    rungs = []
    for k in range(1, int(iterations) + 1):
        deep_level_K = float(production_level_K) / (float(gamma) ** k)
        deep_stage = _build_dag_stage(
            dependency_level_K_override=deep_level_K,
            export_outputs=False,
            path_complexity_report=False,
            **dag_stage_kwargs,
        )
        if deep_stage is None:
            raise RuntimeError("self-check: deep DAG stage produced no supersegments.")
        deep_pred, _deep_succ, _ = _build_component_dependency_maps(
            runtime_components=runtime_components,
            edge_pairs=deep_stage.edge_pairs,
        )
        new_pred: dict[int, list[int]] = {}
        new_succ: dict[int, list[int]] = {}
        n_new = 0
        for comp_id, preds in deep_pred.items():
            extra = sorted(set(preds) - set(prev_pred.get(int(comp_id), [])))
            new_pred[int(comp_id)] = extra
            n_new += len(extra)
            for src in extra:
                new_succ.setdefault(int(src), []).append(int(comp_id))
        for src in new_succ:
            new_succ[src] = sorted(new_succ[src])
        horizon_map = _build_correction_horizon_ss_map(
            runtime_components,
            cut_depths=deep_stage.cut_depths,
            ss_per_layer=int(deep_stage.dag_result.ss_per_layer),
        )
        rungs.append(
            {
                "level_K": float(deep_level_K),
                "num_new_pairs": int(n_new),
                "component_predecessors": new_pred,
                "component_successors": new_succ,
                "horizon_ss_map": horizon_map,
            }
        )
        # cumulative union for the next rung's diff
        merged: dict[int, list[int]] = {}
        for comp_id in set(prev_pred) | set(deep_pred):
            merged[int(comp_id)] = sorted(
                set(prev_pred.get(int(comp_id), [])) | set(deep_pred.get(int(comp_id), []))
            )
        prev_pred = merged
    return {
        "gamma": float(gamma),
        "iterations": int(iterations),
        "mode": "full",
        "horizon_step_ss": int(horizon_step_ss),
        "rungs": rungs,
    }


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
    path_complexity_report: bool = False,
    dependency_level_K_override: float | None = None,
    self_check_gamma: float | None = None,
    self_check_iterations: int = 1,
    self_check_mode: str = "horizon",
    self_check_horizon_step_ss: int = 2,
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
        path_complexity_report=bool(path_complexity_report),
        dependency_level_K_override=dependency_level_K_override,
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
    correction_horizon_by_edge = _build_correction_horizon_by_edge(
        runtime_components,
        edge_pairs=dag_stage.edge_pairs,
        ss_per_layer=int(dag_stage.dag_result.ss_per_layer),
    )

    self_check = None
    if self_check_gamma is not None:
        self_check = _build_self_check_maps(
            gamma=float(self_check_gamma),
            iterations=int(self_check_iterations),
            mode=str(self_check_mode),
            horizon_step_ss=int(self_check_horizon_step_ss),
            production_level_K=float(dag_stage.dag_result.dependency_level_K),
            production_pred_map=component_predecessors,
            runtime_components=runtime_components,
            dag_stage_kwargs=dict(
                solver_mode=str(solver_mode),
                path_config_path=path_config_path,
                num_layers_override=num_layers_override,
                dt_s=dt_s,
                rc=rc,
                phys=phys,
                float_type=float_type,
                solver_velocity_mps=float(solver_velocity_mps),
                out_dir=out_dir,
            ),
        )
        if self_check["mode"] == "horizon":
            print(
                "[self-check] horizon-only refinement ladder: "
                f"{self_check['iterations']} iteration(s), "
                f"+{self_check['horizon_step_ss']} supersegment(s) per iteration "
                "(no deep DAGs/lookups)"
            )
        else:
            rung_desc = ", ".join(
                f"rung{k+1}: {r['level_K']:.4g}K/+{r['num_new_pairs']}p"
                for k, r in enumerate(self_check["rungs"])
            )
            print(
                "[self-check] refinement ladder "
                f"(gamma={self_check['gamma']:.3g}, {self_check['iterations']} iteration(s)): "
                f"{rung_desc}"
            )

    return {
        "planner_mode": str(planner_mode),
        "global_max_cut_depth": int(max(dag_stage.cut_depths, default=0)),
        "correction_weight": float(correction_weight),
        "partition_summary": partition_summary,
        "path_defs": path_defs,
        "rank_assignments": rank_assignments,
        "steps_per_ss": int(dag_stage.dag_result.unit_steps),
        "dependency_level_K": float(dag_stage.dag_result.dependency_level_K),
        "effective_step_nd": float(dag_stage.effective_step_nd),
        "num_layers": int(dag_stage.dag_result.num_layers),
        "ss_per_layer": int(dag_stage.dag_result.ss_per_layer),
        "rank_pred_loads": {
            int(rank): float(load)
            for rank, load in partition_summary["rank_loads"].items()
        },
        "correction_horizon_ss_map": correction_horizon_ss_map,
        "correction_horizon_by_edge": correction_horizon_by_edge,
        "component_predecessors": component_predecessors,
        "component_successors": component_successors,
        "component_dependency_edges": component_dependency_edges,
        "runtime_components": runtime_components,
        "source_components": list(dag_stage.dag_result.components),
        "split_records": [],
        "segments_per_supersegment": 1,
        "path_complexity": getattr(dag_stage.dag_result, "path_complexity_summary", None),
        "self_check": self_check,
    }


__all__ = ["build_partitioned_runtime_plan"]
