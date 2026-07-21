from __future__ import annotations


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


def _boundary_cut_correction_from_depth(
    *,
    cut_depth: int,
    segments_per_supersegment: int = 1,
    correction_weight: float = 0.25,
) -> dict:
    correction_span_ss = float(max(0, int(cut_depth)))
    correction_span_segments = float(correction_span_ss) * float(segments_per_supersegment)
    correction_cost = float(correction_weight) * float(correction_span_segments)
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
]
