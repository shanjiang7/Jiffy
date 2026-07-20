from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass, replace

from hermes.pipelines.config import PipelineConfig
from hermes.pipelines.ss_builder import build_ss_from_cfg
from hermes.utils.dag_utils import (
    write_edges_csv,
    Component,
    write_components_csv,
    find_components,
    split_components_at_layer_boundaries,
    Edge,
)
from hermes.utils.segment_types import SuperSegment
from hermes.DAG.dependency import (
    REpsLookup,
    LookupRuntime,
    build_r_eps_lookup,
    write_r_eps_lookup_csv,
    build_supersegment_dependency_edges,
    calibration_rel_l2_for_epsilon,
    compute_edge_indegree_summary,
)
from hermes.utils.viz_utils import (
    plot_path_with_supersegments,
    plot_dag_spatial,
    plot_components,
    plot_dag_networkx,
)


def _edge_depth_summary(edges: list[Edge]) -> dict:
    n_chain = sum(1 for e in edges if int(e.dst) == int(e.src) + 1)
    n_cross = len(edges) - n_chain
    return {
        "num_edges": int(len(edges)),
        "num_chain_edges": int(n_chain),
        "num_cross_edges": int(n_cross),
        "global_max_cut_depth": int(max((int(e.dst) - int(e.src) for e in edges), default=0)),
    }


@dataclass
class DAGPipelineResult:
    supersegments: list[SuperSegment]
    len_scale: float
    r_eps_lookup: REpsLookup
    edges: list[Edge]
    components: list[Component]
    num_layers: int
    ss_per_layer: int
    unit_steps: int
    dependency_level_K: float
    path_complexity_summary: dict | None = None


def compute_dag_and_components(
    cfg: PipelineConfig,
    *,
    lookup_runtime: LookupRuntime | None = None,
    path_complexity_report: bool = False,
) -> DAGPipelineResult | None:

    print("=== Step 1: Building supersegments ===")
    ss_result = build_ss_from_cfg(cfg)
    supersegments = ss_result.supersegments
    len_scale = ss_result.len_scale
    num_layers = ss_result.num_layers
    ss_per_layer = ss_result.ss_per_layer
    n_ss = len(supersegments)
    if n_ss == 0:
        print("[done] no supersegments to process.")
        return None
    # Solver timesteps per supersegment: a segment's steps tuple holds path
    # SAMPLES (n_moves + 1); the correction horizon must count moves, not samples.
    unit_steps = sum(int(seg.n_moves) for seg in supersegments[0].segments)
    print(f"  {ss_per_layer} SS/layer × {num_layers} layer(s) = {n_ss} total  ({unit_steps} steps each)")

    print("=== Step 2: Building r_eps lookup table ===")
    first_seg = supersegments[0].segments[0]
    if lookup_runtime is None:
        raise ValueError("DAG dependency build requires numerical lookup runtime.")
    seg_steps = int(cfg.steps_per_segment)
    if seg_steps < 1:
        raise ValueError("[run].steps_per_segment must be >= 1.")
    lookup_dt_s = float(first_seg.duration_s) / float(seg_steps)
    max_steps = int(cfg.dag.back_window) * seg_steps
    if max_steps < 1:
        raise ValueError("[dag].back_window must be >= 1.")
    total_time_s = float(max_steps) * lookup_dt_s

    print("  lookup backend = numerical (forced)")
    print(f"  lookup dt = solver dt = {lookup_dt_s*1e6:.1f} us")
    print(
        f"  back window = {cfg.dag.back_window} segments x {seg_steps} steps/segment "
        f"({total_time_s*1e3:.2f} ms) -> {max_steps} lookup steps (initial estimate; "
        "lookup extends until the isotherm vanishes)"
    )
    print(f"  numerical solver mode = {lookup_runtime.solver_mode}")
    if lookup_runtime.source_substeps is not None:
        print(f"  mock numerical source steps = {lookup_runtime.source_substeps}")

    lookup = build_r_eps_lookup(
        model=cfg.dependency.model,
        dt_s=lookup_dt_s,
        V_mps=first_seg.V_mps,
        P_W=first_seg.power_W,
        max_steps=max_steps,
        backend="numerical",
        runtime=lookup_runtime,
    )
    if len(lookup.r_eps_m) < max_steps + 1:
        print(
            f"  r_eps converged at step {len(lookup.r_eps_m) - 1} "
            f"({(len(lookup.r_eps_m) - 1) * lookup_dt_s * 1e3:.2f} ms) – stopped early"
        )

    path_complexity_summary = None
    edges = None
    if path_complexity_report:
        print("=== Step 2b: Estimating path complexity (DAG max in-degree) ===")
        initial_level_K = float(cfg.dependency.model.level_K)
        actual_lseg_mm = float(first_seg.duration_s) * float(first_seg.V_mps) * 1000.0
        # A_path is the max in-degree of the regular retained DAG, so the
        # amplification factor is measured on the graph the pipeline actually
        # corrects along (same pair test / lookup source as the final build).
        edges = build_supersegment_dependency_edges(
            supersegments,
            model=cfg.dependency.model,
            lookup=lookup,
            back_window=cfg.dag.back_window,
        )
        initial_indegree = compute_edge_indegree_summary(edges, n_ss)
        initial_edge_summary = _edge_depth_summary(edges)
        initial_calibration = calibration_rel_l2_for_epsilon(initial_level_K)
        A_path = int(initial_indegree.get("A_path", 0))
        A_for_budget = max(1, int(A_path))
        estimated_amplified_rel_l2 = (
            float(A_for_budget) * float(initial_calibration["rel_l2"])
        )
        path_complexity_summary = {
            "enabled": True,
            "metric": "dag_max_indegree",
            "A_path": int(A_path),
            "A_for_budget": int(A_for_budget),
            "num_segments": int(initial_indegree.get("num_supersegments", 0)),
            "argmax_supersegment_id": initial_indegree.get("argmax_supersegment_id"),
            "mean_predecessors_within_radius": float(
                initial_indegree.get("mean_indegree", 0.0)
            ),
            "segment_length_mm": float(actual_lseg_mm),
            "initial_level_K": float(initial_level_K),
            "initial_calibration": initial_calibration,
            "estimated_amplified_rel_l2": float(estimated_amplified_rel_l2),
            "initial_dag": initial_edge_summary,
        }
        print(
            "  "
            f"initial DAG max cut depth={int(initial_edge_summary['global_max_cut_depth'])}  "
            f"edges={int(initial_edge_summary['num_edges'])} "
            f"({int(initial_edge_summary['num_chain_edges'])} chain, "
            f"{int(initial_edge_summary['num_cross_edges'])} cross)"
        )
        print(
            "  "
            f"A_path={A_path} (max in-degree @ SS {initial_indegree.get('argmax_supersegment_id')})  "
            f"mean={float(path_complexity_summary['mean_predecessors_within_radius']):.2f}  "
            f"initial level_K={initial_level_K:.6g} K  "
            f"calibrated relL2={float(initial_calibration['rel_l2']):.6g}  "
            f"estimated amplified relL2={estimated_amplified_rel_l2:.6g}"
        )

    print("=== Step 3: Building dependency DAG ===")
    if edges is None:
        edges = build_supersegment_dependency_edges(
            supersegments,
            model=cfg.dependency.model,
            lookup=lookup,
            back_window=cfg.dag.back_window,
        )
    else:
        print("  (reusing DAG built during path-complexity estimation)")
    final_edge_summary = _edge_depth_summary(edges)
    if path_complexity_summary is not None:
        path_complexity_summary["final_dag"] = final_edge_summary
    print(
        f"  {int(final_edge_summary['num_edges'])} edges  "
        f"({int(final_edge_summary['num_chain_edges'])} chain, "
        f"{int(final_edge_summary['num_cross_edges'])} cross-edges)"
    )

    print("=== Step 4: Finding DAG components ===")
    edge_pairs = [(int(e.src), int(e.dst)) for e in edges]
    components = find_components(n_ss, edge_pairs)
    components = split_components_at_layer_boundaries(components, ss_per_layer)

    n_trivial = sum(1 for c in components if c.kind == "trivial")
    n_coupled = sum(1 for c in components if c.kind == "coupled")
    print(f"  {len(components)} components: {n_trivial} trivial, {n_coupled} coupled")

    return DAGPipelineResult(
        supersegments=supersegments,
        len_scale=len_scale,
        r_eps_lookup=lookup,
        edges=edges,
        components=components,
        num_layers=num_layers,
        ss_per_layer=ss_per_layer,
        unit_steps=unit_steps,
        dependency_level_K=float(cfg.dependency.model.level_K),
        path_complexity_summary=path_complexity_summary,
    )


def export_dag_results(
    result: DAGPipelineResult,
    out_dir: Path,
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    n_ss = len(result.supersegments)
    edge_pairs = [(int(e.src), int(e.dst)) for e in result.edges]

    print("=== Step 5: Generating visualisations & CSVs ===")
    write_r_eps_lookup_csv(result.r_eps_lookup, out_dir / "r_eps_lookup.csv")
    print(f"  [ok] wrote: {out_dir / 'r_eps_lookup.csv'}")

    write_edges_csv(result.edges, out_dir / "dag_edges.csv", header="src_ss_id,dst_ss_id")
    print(f"  [ok] wrote: {out_dir / 'dag_edges.csv'}")

    write_components_csv(result.components, out_dir / "components.csv")
    print(f"  [ok] wrote: {out_dir / 'components.csv'}")

    plot_dag_networkx(
        result.edges,
        n_ss,
        supersegments=result.supersegments,
        components=result.components,
        len_scale=result.len_scale,
        out_path=out_dir / "dag_graph.png",
    )

    plot_path_with_supersegments(
        result.supersegments,
        len_scale=result.len_scale,
        out_path=out_dir / "path_supersegments.png",
    )

    plot_dag_spatial(
        result.supersegments,
        result.edges,
        len_scale=result.len_scale,
        out_path=out_dir / "dag_spatial.png",
    )

    plot_components(result.components, edge_pairs, n_ss, out_path=out_dir / "components.png")

    print("=== Done ===")
    print(f"  Output dir: {out_dir}")
