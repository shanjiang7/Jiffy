from __future__ import annotations

import os

# Fixed cost of one boundary correction tracer, in supersegments of source-on
# work. Calibrated 2026-07-28 as the pooled least-squares fit over 101
# profiled tracers (Bull h=18um, eps=0.01K, ranks 8/32/64, march+capture):
# tracer_seconds ~= 3.98 + 0.1075*span_ss, i.e. a0 = 3.98/0.506 ~= 7.9 SS and
# slope 0.21 (pair with --correction-weight 0.21). Mechanism (profiled): the
# post-switch-off transient -- the adaptive CG runs ~7 iters/step for ~3 SS
# while the thermal spike is sharp, then coasts near 0. The two-parameter
# affine form deliberately averages over two smaller measured effects
# (snapshot capture, ~3.5 ms/snapshot; region-dependent solver wake-ups,
# +3-4 s on high-curvature path stretches) to stay path-agnostic.
# Override for calibration experiments via HERMES_CORRECTION_FIXED_COST_SS.
CORRECTION_FIXED_COST_SS = float(os.environ.get("HERMES_CORRECTION_FIXED_COST_SS", "7.9"))

# Calibrated slope of the affine correction-cost model (w in w*span + a0),
# fitted together with CORRECTION_FIXED_COST_SS above; every published
# result uses this value. Single source of truth for all signature and CLI
# defaults.
DEFAULT_CORRECTION_WEIGHT = 0.21


def _compute_cut_depths(n_ss: int, edge_pairs: list[tuple[int, int]]) -> list[int]:
    if int(n_ss) <= 1:
        return []

    cut_depths = [0 for _ in range(int(n_ss) - 1)]
    for src, dst in edge_pairs:
        src_i = int(src)
        dst_i = int(dst)
        if not (0 <= int(src_i) < int(dst_i) < int(n_ss)):
            continue
        for boundary in range(int(src_i), int(dst_i)):
            cut_depths[int(boundary)] = max(int(cut_depths[int(boundary)]), int(dst_i) - int(boundary))
    return cut_depths


def _compute_max_out_dst(
    n_ss: int,
    edge_pairs: list[tuple[int, int]],
    ss_per_layer: int | None = None,
) -> list[int]:
    """Per source SS, the farthest destination SS its retained influence
    reaches (the SS itself if it has no forward edge). The outgoing fused
    tracer of a range [s, e] marches to max(max_out_dst[s..e]) - e: each rank
    is charged for the corrections it launches, not for corrections from
    earlier sources that merely pass through its boundary (those are already
    charged once, at their source). When ss_per_layer is given, destinations
    are clamped to the source's layer, matching the runtime horizon guard
    (corrections never cross a layer boundary)."""
    out = list(range(int(n_ss)))
    for src, dst in edge_pairs:
        s, d = int(src), int(dst)
        if not (0 <= s < int(n_ss) and d > s):
            continue
        d = min(d, int(n_ss) - 1)
        if ss_per_layer is not None and int(ss_per_layer) > 0:
            layer_end = ((s // int(ss_per_layer)) + 1) * int(ss_per_layer) - 1
            d = min(d, layer_end)
        if d > out[s]:
            out[s] = d
    return out


def _boundary_cut_correction_from_depth(
    *,
    cut_depth: int,
    segments_per_supersegment: int = 1,
    correction_weight: float = DEFAULT_CORRECTION_WEIGHT,
) -> dict:
    correction_span_ss = float(max(0, int(cut_depth)))
    correction_span_segments = float(correction_span_ss) * float(segments_per_supersegment)
    correction_cost = float(correction_weight) * float(correction_span_segments) + (
        float(CORRECTION_FIXED_COST_SS) * float(segments_per_supersegment)
        if int(cut_depth) > 0
        else 0.0
    )
    if int(cut_depth) <= 0:
        return {
            "correction_span_ss": 0.0,
            "correction_span_segments": 0.0,
            "correction_cost": 0.0,
            "cross_edge_count": 0,
            "min_predecessor_ss": None,
            "max_corrected_ss": None,
            "correction_mode": "none",
        }
    return {
        "correction_span_ss": float(correction_span_ss),
        "correction_span_segments": float(correction_span_segments),
        "correction_cost": float(correction_cost),
        "cross_edge_count": 0,
        "min_predecessor_ss": None,
        "max_corrected_ss": None,
        "correction_mode": "outgoing_cut_depth",
    }


__all__ = [
    "_boundary_cut_correction_from_depth",
    "_compute_cut_depths",
    "_compute_max_out_dst",
]
