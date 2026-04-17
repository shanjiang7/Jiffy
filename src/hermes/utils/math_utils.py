"""
Math and error calculation utilities.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from hermes.scripts.outer_solver import OuterContext


# ── Vector math ─────────────────────────────────────────────────────────────

def dir_to_unit(direction: str) -> Tuple[float, float]:
    d = direction.strip().lower()
    if d == "+x":
        return 1.0, 0.0
    if d == "-x":
        return -1.0, 0.0
    if d == "+y":
        return 0.0, 1.0
    if d == "-y":
        return 0.0, -1.0
    raise ValueError(f"Unsupported direction: {direction!r}. Use one of +x, -x, +y, -y.")


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(b))
    if den == 0.0:
        den = 1.0
    return float(np.linalg.norm(a - b) / den)


# ── Error logic (originally in mpi_utils.py) ────────────────────────────────

def rel_l2_roi_flat(
    a_flat: np.ndarray,
    b_flat: np.ndarray,
    ctx: OuterContext,
    roi: Tuple[int, int, int, int],
) -> float:
    ix0, ix1, iy0, iy1 = roi
    a3 = np.reshape(a_flat, (ctx.nx, ctx.ny, ctx.nz), order="F")
    b3 = np.reshape(b_flat, (ctx.nx, ctx.ny, ctx.nz), order="F")
    return rel_l2(a3[ix0:ix1 + 1, iy0:iy1 + 1, :], b3[ix0:ix1 + 1, iy0:iy1 + 1, :])


def local_error_map(
    seg_np_dict: dict[int, List[np.ndarray]],
    seq_ref_dict: dict[int, List[np.ndarray]],
) -> dict[int, List[float]]:
    out = {}
    for s, arr_list in seg_np_dict.items():
        ref_list = seq_ref_dict[s]
        errs = []
        for i in range(min(len(arr_list), len(ref_list))):
            errs.append(rel_l2(arr_list[i], ref_list[i]))
        out[s] = errs
    return out


def accumulate_error_parts(
    error_list: List[float],
    err_parts: List[dict],
) -> None:
    # Combine the dictionaries from all gathered ranks
    combined = {}
    for part in err_parts:
        if part:
            combined.update(part)
    
    # Sort by component ID and flatten the lists of errors
    for s in sorted(combined.keys()):
        error_list.extend(combined[s])


__all__ = [
    "dir_to_unit",
    "rel_l2",
    "rel_l2_roi_flat",
    "local_error_map",
    "accumulate_error_parts",
]
