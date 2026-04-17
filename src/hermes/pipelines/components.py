from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass

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
)
from hermes.utils.viz_utils import (
    plot_path_with_supersegments,
    plot_dag_spatial,
    plot_components,
    plot_dag_networkx,
)


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


def compute_dag_and_components(
    cfg: PipelineConfig,
    *,
    lookup_runtime: LookupRuntime | None = None,
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
    unit_steps = sum(len(seg.steps) for seg in supersegments[0].segments)
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
        f"({total_time_s*1e3:.2f} ms) -> {max_steps} lookup steps"
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

    print("=== Step 3: Building dependency DAG ===")
    edges = build_supersegment_dependency_edges(
        supersegments,
        model=cfg.dependency.model,
        lookup=lookup,
        back_window=cfg.dag.back_window,
    )
    n_chain = sum(1 for e in edges if int(e.dst) == int(e.src) + 1)
    n_cross = len(edges) - n_chain
    print(f"  {len(edges)} edges  ({n_chain} chain, {n_cross} cross-edges)")

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
