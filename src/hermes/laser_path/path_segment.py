from __future__ import annotations

import numpy as np

from hermes.utils.segment_types import Segment, Step

def sample_traj_with_waypoints(points: np.ndarray, ds: float) -> np.ndarray:
    """
    Sample the trajectory with waypoints.
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError(f"points must have shape (N,2); got {P.shape}")
    if P.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    if P.shape[0] == 1:
        return P.copy()

    ds = float(ds)
    if not np.isfinite(ds) or ds <= 0.0:
        raise ValueError(f"ds must be finite and > 0; got {ds!r}")

    out: list[np.ndarray] = [P[0].copy()]
    tol = 1e-9 * ds
    for i in range(P.shape[0] - 1):
        a = out[-1]
        b = P[i + 1]
        d = b - a
        L = float(np.linalg.norm(d))
        if L == 0.0:
            if b[0] != out[-1][0] or b[1] != out[-1][1]:
                out.append(b.copy())
            continue

        u = d / L
        n_full = int(np.floor((L - tol) / ds))
        if n_full > 0:
            for k in range(1, n_full + 1):
                p = a + u * (k * ds)
                out.append(p.copy())

        rem = b - out[-1]
        rem_len = float(np.linalg.norm(rem))
        if rem_len <= tol:
            out[-1] = b.copy()
        else:
            out.append(b.copy())

    return np.vstack(out)


def sample_traj_with_sections(
    sections: list[tuple[np.ndarray, bool]],
    ds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a sequence of polyline sections and return:
    - sampled trajectory points, shape (N, 2)
    - per-move source flags, shape (N-1,)

    Each section carries a uniform `source_on` flag that applies to all moves
    within that section.
    """
    all_points: list[np.ndarray] = []
    move_flags: list[np.ndarray] = []

    for section_pts, source_on in sections:
        sampled = sample_traj_with_waypoints(np.asarray(section_pts, dtype=float), ds)
        if sampled.shape[0] == 0:
            continue
        if not all_points:
            all_points.append(sampled.copy())
        else:
            prev_last = all_points[-1][-1]
            if np.allclose(prev_last, sampled[0]):
                all_points.append(sampled[1:].copy())
            else:
                all_points.append(sampled.copy())
        n_moves = max(0, sampled.shape[0] - 1)
        if n_moves > 0:
            move_flags.append(np.full(n_moves, bool(source_on), dtype=bool))

    if not all_points:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=bool)

    traj = np.vstack(all_points)
    flags = np.concatenate(move_flags) if move_flags else np.zeros((0,), dtype=bool)
    if traj.shape[0] > 0 and flags.shape[0] != max(0, traj.shape[0] - 1):
        raise ValueError("Per-move source flags do not match sampled trajectory length")
    return traj, flags

def compute_dt_phi(traj_nd: np.ndarray, *, len_scale: float) -> np.ndarray:
    """
    Return laser path per-step rows: x_nd, y_nd, dt_m, phi_rad for each point.
    Each row stores the move *from* that point to the next point.
    The last row corresponds to the last point (dt=0, phi=0, no outgoing move).
    """
    T = np.asarray(traj_nd, dtype=float)
    if T.ndim != 2 or T.shape[1] != 2:
        raise ValueError(f"traj_nd must have shape (K,2); got {T.shape}")
    if T.shape[0] == 0:
        return np.zeros((0, 4), dtype=float)

    out = np.zeros((T.shape[0], 4), dtype=float)
    out[:, 0:2] = T
    # Last point has no outgoing move
    out[-1, 2] = 0.0
    out[-1, 3] = 0.0
    for i in range(T.shape[0] - 1):
        dx = float(T[i + 1, 0] - T[i, 0])
        dy = float(T[i + 1, 1] - T[i, 1])
        ds_nd = float(np.hypot(dx, dy))
        out[i, 2] = ds_nd * float(len_scale)  # meters: move from point i to point i+1
        out[i, 3] = float(np.arctan2(dy, dx)) if ds_nd > 0.0 else 0.0
    return out


def compute_dt_phi_with_source(
    traj_nd: np.ndarray,
    move_source_on: np.ndarray,
    *,
    len_scale: float,
) -> np.ndarray:
    """
    Return rows `[x_nd, y_nd, dt_m, phi_rad, source_on]` for each trajectory
    point, where `source_on` applies to the outgoing move from that point.
    """
    out = np.zeros((0, 5), dtype=float)
    base = compute_dt_phi(traj_nd, len_scale=len_scale)
    if base.shape[0] == 0:
        return out
    flags = np.asarray(move_source_on, dtype=bool)
    if flags.shape != (max(0, base.shape[0] - 1),):
        raise ValueError(
            f"move_source_on must have shape ({max(0, base.shape[0] - 1)},); got {flags.shape}"
        )
    out = np.zeros((base.shape[0], 5), dtype=float)
    out[:, :4] = base
    if flags.size > 0:
        out[:-1, 4] = flags.astype(float)
    out[-1, 4] = 0.0
    return out


def build_segments_from_program(
    program: np.ndarray,
    *,
    steps_per_segment: int,
    power_W: float,
    V_mps: float,
    t0_s: float = 0.0,
    width_roi_m: float = 0.0,
) -> list[Segment]:
    """
    Build `Segment` objects from program rows using K dt-moves per segment.
    """
    steps_per_segment = int(steps_per_segment)
    if steps_per_segment < 1:
        raise ValueError("steps_per_segment must be >= 1")
    prog = np.asarray(program, dtype=float)
    if prog.ndim != 2 or prog.shape[1] not in {4, 5}:
        raise ValueError(f"program must have shape (N,4) or (N,5); got {prog.shape}")

    n_rows = int(prog.shape[0])
    if n_rows == 0:
        return []

    segments: list[Segment] = []
    t = float(t0_s)
    has_source_col = prog.shape[1] == 5

    seg_id = 0
    start_pt = 0
    while start_pt < n_rows - 1:
        end_limit = min(n_rows - 1, start_pt + steps_per_segment)
        end_pt = end_limit
        if has_source_col:
            src_val = prog[start_pt, 4]
            for k in range(start_pt + 1, end_limit + 1):
                if prog[k, 4] != src_val:
                    end_pt = k
                    break
        rows = prog[start_pt : end_pt + 1]

        if rows.shape[0] >= 1:
            rows = rows.copy()
            rows[-1, 2] = 0.0
            rows[-1, 3] = 0.0

        steps = tuple(Step(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in rows)

        seg_power = float(power_W)
        if has_source_col:
            seg_power = float(power_W) if bool(prog[start_pt, 4]) else 0.0

        seg = Segment(
            id=seg_id,
            steps=steps,
            power_W=seg_power,
            V_mps=float(V_mps),
            t_start_s=t,
            width_m=float(width_roi_m),
        )
        segments.append(seg)
        t = seg.time_window_s[1]

        seg_id += 1
        start_pt = end_pt

    return segments

__all__ = [
    "build_segments_from_program",
    "sample_traj_with_waypoints",
    "sample_traj_with_sections",
    "compute_dt_phi",
    "compute_dt_phi_with_source",
]
