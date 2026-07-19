"""
Thermal dependency model and supersegment DAG construction.

Public API
----------
AABB, DependencyModel, REpsLookup, LookupRuntime
build_r_eps_lookup, build_r_eps_lookup_numerical
write_r_eps_lookup_csv
aabb_distance_nd, aabb_distance_m
build_supersegment_dependency_edges
build_adjacency
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import List

import numpy as np

from hermes.utils.dag_utils import Edge
from hermes.utils.segment_types import Segment, SuperSegment

AABB = tuple[float, float, float, float]


CALIBRATION_LSEG_MM = np.asarray(
    [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    dtype=float,
)
CALIBRATION_EPSILON_K = np.asarray(
    [4.07e2, 1.15e2, 2.74e1, 5.72e0, 1.09e0, 1.93e-1, 3.24e-2, 5.23e-3, 8.19e-4],
    dtype=float,
)
CALIBRATION_REL_L2 = np.asarray(
    [6.62e-3, 1.87e-3, 4.50e-4, 9.60e-5, 1.87e-5, 3.41e-6, 5.90e-7, 9.79e-8, 1.58e-8],
    dtype=float,
)


def _interp_loglog(x: float, xs: np.ndarray, ys: np.ndarray) -> tuple[float, bool]:
    if float(x) <= 0.0:
        raise ValueError(f"Interpolation input must be > 0, got {x!r}")
    order = np.argsort(xs)
    xs_sorted = np.asarray(xs, dtype=float)[order]
    ys_sorted = np.asarray(ys, dtype=float)[order]
    if len(xs_sorted) < 2:
        raise ValueError("At least two calibration points are required.")
    log_xs = np.log(xs_sorted)
    log_ys = np.log(ys_sorted)
    log_x = np.log(float(x))
    extrapolated = bool(log_x < float(log_xs[0]) or log_x > float(log_xs[-1]))
    if log_x < float(log_xs[0]):
        slope = (float(log_ys[1]) - float(log_ys[0])) / (float(log_xs[1]) - float(log_xs[0]))
        log_y = float(log_ys[0]) + slope * (float(log_x) - float(log_xs[0]))
    elif log_x > float(log_xs[-1]):
        slope = (float(log_ys[-1]) - float(log_ys[-2])) / (float(log_xs[-1]) - float(log_xs[-2]))
        log_y = float(log_ys[-1]) + slope * (float(log_x) - float(log_xs[-1]))
    else:
        log_y = np.interp(log_x, log_xs, log_ys)
    return float(np.exp(log_y)), extrapolated


def calibration_epsilon_for_rel_l2(target_rel_l2: float) -> dict:
    """Map target relative L2 error to epsilon/level_K using the built-in table."""
    epsilon_K, clipped = _interp_loglog(
        float(target_rel_l2),
        CALIBRATION_REL_L2,
        CALIBRATION_EPSILON_K,
    )
    implied_lseg_mm, lseg_clipped = _interp_loglog(
        float(target_rel_l2),
        CALIBRATION_REL_L2,
        CALIBRATION_LSEG_MM,
    )
    return {
        "target_rel_l2": float(target_rel_l2),
        "level_K": float(epsilon_K),
        "calibration_lseg_mm": float(implied_lseg_mm),
        "extrapolated": bool(clipped or lseg_clipped),
        "table": "builtin_lseg_epsilon_rel_l2",
    }


def calibration_rel_l2_for_epsilon(epsilon_K: float) -> dict:
    """Map epsilon/level_K to relative L2 error using the built-in table."""
    rel_l2, clipped = _interp_loglog(
        float(epsilon_K),
        CALIBRATION_EPSILON_K,
        CALIBRATION_REL_L2,
    )
    implied_lseg_mm, lseg_clipped = _interp_loglog(
        float(epsilon_K),
        CALIBRATION_EPSILON_K,
        CALIBRATION_LSEG_MM,
    )
    return {
        "level_K": float(epsilon_K),
        "rel_l2": float(rel_l2),
        "calibration_lseg_mm": float(implied_lseg_mm),
        "extrapolated": bool(clipped or lseg_clipped),
        "table": "builtin_lseg_epsilon_rel_l2",
    }


# ── AABB geometry ─────────────────────────────────────────────────────────────

def aabb_distance_nd(a: AABB, b: AABB) -> float:
    """Minimum Euclidean distance between two 2-D AABBs. Returns 0 if they overlap."""
    ax0, ax1, ay0, ay1 = sorted([float(a[0]), float(a[1])]) + sorted([float(a[2]), float(a[3])])
    bx0, bx1, by0, by1 = sorted([float(b[0]), float(b[1])]) + sorted([float(b[2]), float(b[3])])
    dx = max(0.0, bx0 - ax1, ax0 - bx1)
    dy = max(0.0, by0 - ay1, ay0 - by1)
    return float((dx * dx + dy * dy) ** 0.5)


def aabb_distance_m(a_nd: AABB, b_nd: AABB, *, len_scale: float) -> float:
    """AABB distance in meters (nd coords scaled by len_scale)."""
    s = float(len_scale)
    return aabb_distance_nd(
        tuple(float(v) * s for v in a_nd),
        tuple(float(v) * s for v in b_nd),
    )


def _segment_target_patch_samples_nd(
    seg: Segment,
    *,
    len_scale: float,
    step_stride: int,
) -> tuple[tuple[float, AABB], ...]:
    """Sample square target patches every step_stride steps along seg."""
    if not seg.steps:
        return ((0.0, seg.path_bounds_nd),)
    s = float(len_scale)
    if s <= 0.0:
        raise ValueError(f"len_scale must be > 0; got {s!r}")
    stride = max(1, int(step_stride))
    inv_v = 1.0 / float(seg.V_mps)
    half_w_nd = 0.5 * max(0.0, float(seg.width_m)) / s

    samples: list[tuple[float, AABB]] = []
    elapsed_s = 0.0
    last_idx = len(seg.steps) - 1
    for step_idx, step in enumerate(seg.steps):
        if step_idx % stride == 0 or step_idx == last_idx:
            x0 = float(step.x_nd)
            y0 = float(step.y_nd)
            patch = (x0 - half_w_nd, x0 + half_w_nd, y0 - half_w_nd, y0 + half_w_nd)
            samples.append((elapsed_s, patch))
        elapsed_s += float(step.dt_m) * inv_v
    return tuple(samples)


def _segment_chords_nd(
    seg: Segment,
    *,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose a segment's path into chords between n_samples evenly spaced
    path samples (first and last always included).

    Returns (chords, end_elapsed_s, start_elapsed_s):
      chords          (C, 4) nd coords [x0, y0, x1, y1], C = n_samples - 1
      end_elapsed_s   (C,) elapsed seconds from segment start to chord END
                      (deposit time of the chord's youngest heat, source side)
      start_elapsed_s (C,) elapsed seconds from segment start to chord START
                      (earliest laser arrival on the chord, target side)
    """
    if int(n_samples) < 2:
        raise ValueError("segment_samples must be >= 2")
    steps = seg.steps
    if len(steps) < 2:
        x = float(steps[0].x_nd) if steps else 0.0
        y = float(steps[0].y_nd) if steps else 0.0
        return (
            np.array([[x, y, x, y]], dtype=float),
            np.zeros(1, dtype=float),
            np.zeros(1, dtype=float),
        )
    inv_v = 1.0 / float(seg.V_mps)
    elapsed = np.concatenate(
        [[0.0], np.cumsum([float(s.dt_m) * inv_v for s in steps[:-1]])]
    )
    idxs = np.unique(np.round(np.linspace(0, len(steps) - 1, int(n_samples))).astype(int))
    xs = np.array([float(steps[k].x_nd) for k in idxs])
    ys = np.array([float(steps[k].y_nd) for k in idxs])
    chords = np.column_stack([xs[:-1], ys[:-1], xs[1:], ys[1:]])
    return chords, elapsed[idxs][1:], elapsed[idxs][:-1]


def _segment_pair_min_dist_nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Pairwise minimum distances between two chord sets.

    a: (Ca, 4) segments [x0, y0, x1, y1]; b: (Cb, 4). Returns (Ca, Cb).
    Closest-point-of-two-segments (Ericson), vectorised; handles degenerate
    (point) chords via the eps guards.
    """
    eps = 1e-30
    P0 = a[:, None, 0:2]
    d1 = a[:, None, 2:4] - P0
    Q0 = b[None, :, 0:2]
    d2 = b[None, :, 2:4] - Q0
    r = P0 - Q0
    aa = np.sum(d1 * d1, axis=-1)
    ee = np.sum(d2 * d2, axis=-1)
    ff = np.sum(d2 * r, axis=-1)
    cc = np.sum(d1 * r, axis=-1)
    bb = np.sum(d1 * d2, axis=-1)
    denom = aa * ee - bb * bb
    s = np.where(
        denom > eps,
        np.clip((bb * ff - cc * ee) / np.where(denom > eps, denom, 1.0), 0.0, 1.0),
        0.0,
    )
    t = np.where(ee > eps, (bb * s + ff) / np.where(ee > eps, ee, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    s = np.where(aa > eps, np.clip((bb * t - cc) / np.where(aa > eps, aa, 1.0), 0.0, 1.0), 0.0)
    diff = (P0 + s[..., None] * d1) - (Q0 + t[..., None] * d2)
    return np.sqrt(np.sum(diff * diff, axis=-1))


# ── Models ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DependencyModel:
    len_scale: float = 1.0
    level_K: float = 5e-2
    resolution_m: float = 20e-6
    bc: str = "flux"
    spacing_m: float = 200e-6
    window_x_um: int = 6000
    window_y_um: int = 6000
    window_z_um: int = 300
    target_patch_step_stride: int = 10
    # Pairwise proximity test used for edge retention:
    #   "aabb"   - source AABB vs square target patches, elapsed time from the
    #              source segment END (published/paper behaviour).
    #   "chords" - both segments decomposed into chords between
    #              `segment_samples` evenly spaced path samples; exact
    #              segment-to-segment distances; each source chord carries its
    #              own deposit time; the target ROI half-width is added to the
    #              retention radius instead of inflating geometry.
    pair_test: str = "aabb"
    segment_samples: int = 5


@dataclass(frozen=True)
class REpsLookup:
    dt_s: float
    V_mps: float
    P_W: float
    r_eps_m: tuple[float, ...]

    def at(self, idx: int) -> float:
        k = int(idx)
        if k < 0:
            raise ValueError("idx must be >= 0")
        if k >= len(self.r_eps_m):
            raise IndexError(f"idx={k} out of range (len={len(self.r_eps_m)})")
        return float(self.r_eps_m[k])

    def as_array(self) -> np.ndarray:
        return np.asarray(self.r_eps_m, dtype=float)


@dataclass(frozen=True)
class LookupRuntime:
    rc: object
    phys: object
    float_type: object
    solver_mode: str = "fused"
    source_on_steps: int | None = None
    source_substeps: int | None = None


# ── r_eps lookup ──────────────────────────────────────────────────────────────

def build_r_eps_lookup(
    *,
    model: DependencyModel,
    dt_s: float,
    V_mps: float,
    P_W: float,
    max_steps: int,
    backend: str = "numerical",
    runtime: LookupRuntime | None = None,
) -> REpsLookup:
    backend_norm = str(backend).strip().lower()
    if backend_norm == "numerical":
        return build_r_eps_lookup_numerical(
            model=model,
            dt_s=dt_s,
            V_mps=V_mps,
            P_W=P_W,
            max_steps=max_steps,
            runtime=runtime,
        )
    raise ValueError(f"Unsupported lookup backend {backend!r}; only 'numerical' is supported.")


def _build_lookup_rc(
    *,
    rc,
    model: DependencyModel,
    V_mps: float,
    P_W: float,
):
    h = float(model.resolution_m)
    lxd = float(model.window_x_um) * 1e-6
    lyd = float(model.window_y_um) * 1e-6
    lzd = float(model.window_z_um) * 1e-6
    return replace(
        rc,
        laser=replace(rc.laser, v=float(V_mps), Q=float(P_W)),
        level3=replace(
            rc.level3,
            lxd=lxd,
            lyd=lyd,
            lzd=lzd,
            h_tuple=(h, h, h),
            hx=h,
            hy=h,
            hz=h,
        ),
    )


def _resolve_source_substeps(
    *,
    runtime: LookupRuntime,
    model: DependencyModel,
    dt_s: float,
    V_mps: float,
) -> int:
    if runtime.source_substeps is not None:
        if int(runtime.source_substeps) < 1:
            raise ValueError("LookupRuntime.source_substeps must be >= 1.")
        return int(runtime.source_substeps)
    seg_len_m = float(V_mps) * float(dt_s)
    h_m = float(model.resolution_m)
    if not np.isfinite(seg_len_m) or not np.isfinite(h_m) or h_m <= 0.0:
        raise ValueError(f"Invalid segment length/grid spacing: seg_len_m={seg_len_m!r}, h_m={h_m!r}")
    return max(1, int(math.ceil(seg_len_m / h_m)))


def _resolve_source_on_steps(
    *,
    runtime: LookupRuntime,
) -> int:
    if runtime.source_on_steps is None:
        return 1
    if int(runtime.source_on_steps) < 1:
        raise ValueError("LookupRuntime.source_on_steps must be >= 1.")
    return int(runtime.source_on_steps)


def _extract_r_eps_m_numerical(
    *,
    ctx,
    u,
    phys,
    level_K: float,
    center_x_m: float = 0.0,
) -> float:
    import cupy as cp

    u3d = u.reshape((ctx.nx, ctx.ny, ctx.nz), order="F")
    delta_top_K = (u3d[:, :, -1] - ctx.u0) * float(phys.deltaT)
    mask = delta_top_K >= float(level_K)
    if not bool(cp.any(mask).item()):
        return 0.0

    # Radius measured from the deposited track's midpoint (center_x_m), the
    # minimax anchor of the chord: the pair test dilates the whole chord by
    # r_eps, and any anchor ON the chord is safe (distance-to-anchor >=
    # distance-to-chord); the midpoint minimises the isotropic over-estimate
    # (~L/2 instead of ~L for the end point).
    x_m = ctx.x_init_cp[:, None] * float(phys.len_scale) - float(center_x_m)
    y_m = ctx.y_init_cp[None, :] * float(phys.len_scale)
    r_m = cp.sqrt(x_m * x_m + y_m * y_m)
    return float(cp.max(r_m[mask]).item())


def build_r_eps_lookup_numerical(
    *,
    model: DependencyModel,
    dt_s: float,
    V_mps: float,
    P_W: float,
    max_steps: int,
    runtime: LookupRuntime | None,
) -> REpsLookup:
    if runtime is None:
        raise ValueError("Numerical lookup backend requires a LookupRuntime.")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError(f"dt_s must be finite and > 0; got {dt_s!r}")
    if max_steps < 0:
        raise ValueError("max_steps must be >= 0")

    solver_mode = str(runtime.solver_mode).strip().lower()
    if solver_mode not in {"legacy", "fused"}:
        raise ValueError(f"Unsupported numerical lookup solver_mode={runtime.solver_mode!r}.")
    source_substeps = _resolve_source_substeps(
        runtime=runtime,
        model=model,
        dt_s=dt_s,
        V_mps=V_mps,
    )
    source_on_steps = _resolve_source_on_steps(runtime=runtime)

    params = {
        "backend": "numerical",
        "dt_s": float(dt_s),
        "level_K": float(model.level_K),
        "resolution_m": float(model.resolution_m),
        "bc": str(model.bc),
        "spacing_m": float(model.spacing_m),
        "V_mps": float(V_mps),
        "P_W": float(P_W),
        "window_x_um": int(model.window_x_um),
        "window_y_um": int(model.window_y_um),
        "window_z_um": int(model.window_z_um),
        # The lookup diffuses until the isotherm vanishes; max_steps is only
        # the initial estimate, so the cache key carries the hard cap instead.
        "horizon_mode": "until_convergence",
        "hard_cap_steps": 10 * max(1, int(max_steps)),
        "solver_mode": solver_mode,
        "source_on_steps": int(source_on_steps),
        "source_substeps": int(source_substeps),
        # Radius anchor: midpoint of the deposited track (was: end point).
        "radius_center": "track_midpoint",
        "laser_x_span_m": float(runtime.rc.laser.x_span_m),
        "float_type": getattr(runtime.float_type, "__name__", str(runtime.float_type)),
        "phys_n1": float(runtime.phys.n1),
        "phys_n2": float(runtime.phys.n2),
        "phys_n3": float(runtime.phys.n3),
        "phys_deltaT": float(runtime.phys.deltaT),
        "phys_len_scale": float(runtime.phys.len_scale),
        "phys_time_scale": float(runtime.phys.time_scale),
        "source_on_motion_mode": "moving_domain_source_fixed",
    }
    param_str = json.dumps(params, sort_keys=True)
    h = hashlib.md5(param_str.encode("utf-8")).hexdigest()
    cache_dir = Path(".hermes_cache").expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"r_eps_{h}.npy"
    if cache_file.exists():
        r_eps = tuple(float(x) for x in np.load(cache_file))
        return REpsLookup(
            dt_s=float(dt_s),
            V_mps=float(V_mps),
            P_W=float(P_W),
            r_eps_m=r_eps,
        )

    import cupy as cp
    from hermes.motion.executor import apply_domain_movement, prepare_movement_cache, reset_cache_origin
    from hermes.scripts.outer_solver import build_outer_context

    rc_lookup = _build_lookup_rc(
        rc=runtime.rc,
        model=model,
        V_mps=V_mps,
        P_W=P_W,
    )

    dt_nd = float(dt_s) / float(runtime.phys.time_scale)
    dt_sub_s = float(dt_s) / float(source_substeps)
    dt_sub_nd = dt_sub_s / float(runtime.phys.time_scale)

    ctx = build_outer_context(rc_lookup, runtime.phys, runtime.float_type, dt_nd, solver_mode=solver_mode)
    ctx_source = build_outer_context(
        rc_lookup,
        runtime.phys,
        runtime.float_type,
        dt_sub_nd,
        solver_mode=solver_mode,
    )
    u = cp.full((ctx.nx * ctx.ny * ctx.nz,), ctx.u0, dtype=runtime.float_type)

    qs_off = cp.zeros((ctx.nx, ctx.ny), dtype=runtime.float_type)

    seg_len_nd = (float(V_mps) * float(dt_s)) / float(runtime.phys.len_scale)
    dx_sub_nd = seg_len_nd / float(source_substeps)

    x_arr = ctx_source.x_init_cp.copy()
    y_arr = ctx_source.y_init_cp.copy()
    z_arr = ctx_source.z_init_cp.copy()
    x_src = 0.0
    y_src = 0.0
    move_cache = None
    if dx_sub_nd != 0.0:
        move_cache = prepare_movement_cache(
            nx=ctx_source.nx,
            ny=ctx_source.ny,
            nz=ctx_source.nz,
            u=u,
            x_arr=x_arr,
            y_arr=y_arr,
            z_arr=z_arr,
            vx=abs(dx_sub_nd),
            vy=0.0,
        )
        reset_cache_origin(move_cache, x_arr, y_arr, z_arr)

    for _ in range(int(source_on_steps)):
        for _ in range(int(source_substeps)):
            # Source-on: move both domain and source each substep (same pattern as outer solver path stepping).
            if move_cache is not None:
                apply_domain_movement(
                    ctx=ctx_source,
                    u=u,
                    x_arr=x_arr,
                    y_arr=y_arr,
                    z_arr=z_arr,
                    dx_step=dx_sub_nd,
                    dy_step=0.0,
                    cache=move_cache,
                )
            x_src += dx_sub_nd

            y_lin, x_lin = cp.meshgrid(y_arr, x_arr)
            qs_on = (
                ctx_source.n1
                * cp.exp(-2.0 * (((x_lin - x_src) ** 2 + (y_lin - y_src) ** 2) / (ctx_source.sigma_nd ** 2)))
            ).astype(runtime.float_type)
            u = ctx_source.solve_one_step(u, qs_on)

    # Diffuse until the isotherm actually vanishes: `max_steps` (derived from
    # [dag].back_window) is only an initial horizon estimate; the physical
    # thermal-memory horizon is where r_eps reaches zero. A hard safety cap
    # (10x the estimate) guards against configurations whose isotherm never
    # converges (e.g. near-adiabatic lookup domains at very small level_K).
    hard_cap_steps = 10 * max(1, int(max_steps))
    # The moving-domain deposit pins the source at the window origin, so the
    # deposited track spans [-L, 0] along +x; its midpoint sits at -L/2.
    track_len_m = float(source_on_steps) * float(V_mps) * float(dt_s)
    center_x_m = -0.5 * track_len_m
    r_eps_vals = [
        _extract_r_eps_m_numerical(
            ctx=ctx,
            u=u,
            phys=runtime.phys,
            level_K=model.level_K,
            center_x_m=center_x_m,
        )
    ]
    k = 0
    while float(r_eps_vals[-1]) > 0.0 or k == 0:
        if k >= hard_cap_steps:
            print(
                f"  [warning] r_eps lookup did not converge within {hard_cap_steps} steps "
                f"({hard_cap_steps * dt_s * 1e3:.1f} ms); thermal memory will be truncated. "
                "Increase [dag].back_window or the lookup window/bc drainage.",
                flush=True,
            )
            break
        k += 1
        u = ctx.solve_one_step(u, qs_off)
        r_eps_vals.append(
            _extract_r_eps_m_numerical(
                ctx=ctx,
                u=u,
                phys=runtime.phys,
                level_K=model.level_K,
                center_x_m=center_x_m,
            )
        )
        if (k % 10000) == 0:
            print(
                f"  r_eps lookup: step {k} ({k * dt_s * 1e3:.1f} ms), "
                f"radius {float(r_eps_vals[-1]) * 1e3:.3f} mm",
                flush=True,
            )

    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    r_eps = tuple(float(x) for x in r_eps_vals)
    np.save(cache_file, np.asarray(r_eps, dtype=float))
    return REpsLookup(
        dt_s=float(dt_s),
        V_mps=float(V_mps),
        P_W=float(P_W),
        r_eps_m=r_eps,
    )



def write_r_eps_lookup_csv(lookup: REpsLookup, out_csv) -> Path:
    """Write lookup table to CSV: idx, time_s, r_eps_m."""
    out_path = Path(out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = lookup.as_array()
    idx = np.arange(int(r.shape[0]), dtype=int)
    mat = np.column_stack([idx.astype(float), idx.astype(float) * float(lookup.dt_s), r])
    np.savetxt(out_path, mat, delimiter=",",
               header="idx,time_s,r_eps_m", comments="")
    return out_path


# ── SuperSegment DAG builder ──────────────────────────────────────────────────

def build_supersegment_dependency_edges(
    supersegments: List[SuperSegment],
    *,
    model: DependencyModel,
    lookup: REpsLookup,
    back_window: int = 100,
) -> List[Edge]:
    """
    Build directed dependency edges (src=SS_i → dst=SS_j) between SuperSegments
    using segment-level precision.

    Flattens the supersegments to time-ordered Segments and dispatches on
    ``model.pair_test``: "chords" (exact chord-to-chord distances with
    per-chord deposit times) or "aabb" (published bounding-box test). Any
    segment pair from different SSes meeting the thermal proximity condition
    produces one deduplicated SS-level edge.

    back_window : maximum number of previous *segments* to check (not SSes);
    the chords test treats it as an initial estimate and extends it to the
    lookup's physical thermal horizon.
    """
    # Flatten to (ss_id, Segment) in time order
    seg_list: List[tuple[int, Segment]] = [
        (int(ss.id), seg)
        for ss in supersegments
        for seg in ss.segments
    ]
    n = len(seg_list)
    if n == 0:
        return []
    back_window = max(1, int(back_window))
    pair_test = str(getattr(model, "pair_test", "aabb")).strip().lower()
    builder = {"chords": _build_edges_chords, "aabb": _build_edges_aabb}.get(pair_test)
    if builder is None:
        raise ValueError(f"Unknown pair_test {pair_test!r}; expected 'aabb' or 'chords'.")
    return builder(
        seg_list,
        model=model,
        lookup=lookup,
        back_window=back_window,
    )


def _build_edges_aabb(
    seg_list: List[tuple[int, Segment]],
    *,
    model: DependencyModel,
    lookup: REpsLookup,
    back_window: int,
) -> List[Edge]:
    """
    Published pair test: source AABB vs moving square target patches, elapsed
    time measured from the source segment END.
    """
    n = len(seg_list)
    source_bounds_nd = [seg.path_bounds_nd for _, seg in seg_list]
    source_end_s = [float(seg.t_start_s + seg.duration_s) for _, seg in seg_list]
    target_samples_nd = [
        _segment_target_patch_samples_nd(
            seg,
            len_scale=float(model.len_scale),
            step_stride=int(model.target_patch_step_stride),
        )
        for _, seg in seg_list
    ]
    edges_set: set[tuple[int, int]] = set()

    for j in range(n):
        ss_j_id, seg_j = seg_list[j]
        if seg_j.power_W == 0.0:
            continue
        target_samples_j = target_samples_nd[j]
        for i in range(max(0, j - back_window), j):
            ss_i_id, seg_i = seg_list[i]
            if ss_i_id == ss_j_id:
                continue  # intra-SS pair – no inter-SS edge needed
            if seg_i.power_W == 0.0:
                continue
            # Source side uses the centerline path bounds only; target side uses a
            # moving square ROI patch sampled along seg_j so width_roi_m matches
            # the S3-start patch logic while still covering the full target path.
            src_bounds_i = source_bounds_nd[i]
            seg_i_end_s = source_end_s[i]
            for local_elapsed_s, target_patch_nd in target_samples_j:
                elapsed_s = max(0.0, seg_j.t_start_s + local_elapsed_s - seg_i_end_s)
                dt_idx = max(0, min(int(elapsed_s / lookup.dt_s), len(lookup.r_eps_m) - 1))
                d_m = aabb_distance_m(
                    target_patch_nd,
                    src_bounds_i,
                    len_scale=float(model.len_scale),
                )
                if d_m <= float(lookup.at(dt_idx)):
                    edges_set.add((ss_i_id, ss_j_id))
                    break

    return [Edge(src=s, dst=d) for s, d in sorted(edges_set)]


def _build_edges_chords(
    seg_list: List[tuple[int, Segment]],
    *,
    model: DependencyModel,
    lookup: REpsLookup,
    back_window: int,
) -> List[Edge]:
    """
    Chord-based space-time retention test (pair_test = "chords").

    Both segments are decomposed into chords between `model.segment_samples`
    evenly spaced path samples. For each (source chord k, target chord l):
        d_kl  = exact chord-to-chord distance (metres)
        tau_kl = (t_j_start + target_chord_start_l) - source_chord_end_k
    and the edge (ss_i -> ss_j) is retained iff any pair satisfies
        d_kl <= r_eps(tau_kl) + width_roi/2
    The ROI half-width dilates the retention radius (capsule semantics)
    instead of inflating the target geometry into square patches.
    """
    n = len(seg_list)
    s = float(model.len_scale)
    if s <= 0.0:
        raise ValueError(f"len_scale must be > 0; got {s!r}")
    n_samples = int(getattr(model, "segment_samples", 5))

    chords_nd: list[np.ndarray] = []
    src_end_abs_s: list[np.ndarray] = []
    tgt_start_loc_s: list[np.ndarray] = []
    for _, seg in seg_list:
        chords, end_el, start_el = _segment_chords_nd(seg, n_samples=n_samples)
        chords_nd.append(chords)
        src_end_abs_s.append(float(seg.t_start_s) + end_el)
        tgt_start_loc_s.append(start_el)

    # Normalised per-segment AABBs (nd) for the vectorised broad phase.
    # bbox distance <= chord distance, so rejecting on
    #   d_bbox > max(r_eps) + width_roi/2
    # can never drop a pair the exact narrow-phase test would keep.
    bounds = np.empty((n, 4), dtype=float)
    ss_ids = np.empty(n, dtype=np.int64)
    powered = np.empty(n, dtype=bool)
    for k, (ss_id, seg) in enumerate(seg_list):
        b = seg.path_bounds_nd
        x0, x1 = (b[0], b[1]) if b[0] <= b[1] else (b[1], b[0])
        y0, y1 = (b[2], b[3]) if b[2] <= b[3] else (b[3], b[2])
        bounds[k] = (x0, x1, y0, y1)
        ss_ids[k] = int(ss_id)
        powered[k] = float(seg.power_W) > 0.0

    r_eps = lookup.as_array()
    n_lookup = len(r_eps)
    inv_dt = 1.0 / float(lookup.dt_s)
    r_eps_max_m = float(r_eps.max()) if n_lookup else 0.0
    edges_set: set[tuple[int, int]] = set()

    # An empty isotherm (r_eps == 0: no point above the epsilon threshold)
    # means no dependency regardless of geometry, so the last nonzero lookup
    # index defines a hard thermal-time horizon. Convert it to a segment-count
    # bound on the back window: for i < j the elapsed time is at least
    # (j - i - 1) * min segment duration.
    nz = np.nonzero(r_eps > 0.0)[0]
    if nz.size == 0:
        return []
    t_horizon_s = float(nz[-1] + 1) * float(lookup.dt_s)
    min_dur_s = min(float(seg.duration_s) for _, seg in seg_list if float(seg.duration_s) > 0.0)
    # The window is the physical thermal-memory horizon, NOT the configured
    # back_window: for i < j the elapsed time is at least
    # (j - i - 1) * min segment duration, so no segment older than window_eff
    # can still carry an above-epsilon residual.
    window_eff = int(t_horizon_s / min_dur_s) + 2
    if float(r_eps[-1]) > 0.0:
        # Lookup hit its hard cap without converging: thermal memory beyond
        # the table is unknown. Fall back to unbounded history (conservative).
        print(
            "  [warning] r_eps lookup unconverged at its hard cap; "
            "using unbounded back window for edge retention.",
            flush=True,
        )
        window_eff = n
    if int(back_window) < int(window_eff):
        print(
            f"  [note] configured back_window={int(back_window)} segments is shorter than "
            f"the thermal horizon ({int(window_eff)} segments, {t_horizon_s*1e3:.1f} ms); "
            "using the physical horizon.",
            flush=True,
        )

    for j in range(n):
        ss_j_id, seg_j = seg_list[j]
        if not powered[j]:
            continue
        half_w_m = 0.5 * max(0.0, float(seg_j.width_m))
        lo = max(0, j - window_eff)
        if lo >= j:
            continue

        # Broad phase: one vectorised bbox-distance sweep over the window.
        W = bounds[lo:j]
        dx = np.maximum(0.0, np.maximum(W[:, 0] - bounds[j, 1], bounds[j, 0] - W[:, 1]))
        dy = np.maximum(0.0, np.maximum(W[:, 2] - bounds[j, 3], bounds[j, 2] - W[:, 3]))
        near = (np.hypot(dx, dy) * s) <= (r_eps_max_m + half_w_m)
        cand = np.nonzero(near & powered[lo:j] & (ss_ids[lo:j] != int(ss_j_id)))[0] + lo
        if cand.size == 0:
            continue

        # Narrow phase, batched: stack every candidate's chords into one
        # kernel call instead of one call per candidate.
        tgt_chords = chords_nd[j]
        tgt_arrival_abs = float(seg_j.t_start_s) + tgt_start_loc_s[j]   # (Ct,)
        src_chords = np.concatenate([chords_nd[i] for i in cand])       # (Crows, 4)
        src_ends = np.concatenate([src_end_abs_s[i] for i in cand])     # (Crows,)
        counts = np.array([chords_nd[i].shape[0] for i in cand])
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])

        d_m = _segment_pair_min_dist_nd(src_chords, tgt_chords) * s     # (Crows, Ct)
        tau = np.maximum(0.0, tgt_arrival_abs[None, :] - src_ends[:, None])
        idx = np.clip((tau * inv_dt).astype(np.int64), 0, n_lookup - 1)
        radius = r_eps[idx]
        row_hit = np.any((radius > 0.0) & (d_m <= radius + half_w_m), axis=1)  # (Crows,)
        cand_hit = np.logical_or.reduceat(row_hit, offsets)
        for i in cand[np.nonzero(cand_hit)[0]]:
            edges_set.add((int(ss_ids[i]), int(ss_j_id)))

    return [Edge(src=a, dst=b) for a, b in sorted(edges_set)]


def compute_edge_indegree_summary(
    edges: List[Edge],
    num_supersegments: int,
) -> dict:
    """
    In-degree statistics of the retained dependency DAG.

    A_path = max in-degree = the largest number of retained source segments
    whose influence superposes on a single target — the amplification factor
    used to split a global error budget across simultaneous sub-epsilon
    neglects. Measured on the graph the pipeline actually corrects along, so
    the factor is consistent with the configured pair test and lookup source.
    """
    n = int(num_supersegments)
    counts = np.zeros(max(1, n), dtype=np.int64)
    for e in edges:
        dst = int(e.dst)
        if 0 <= dst < n:
            counts[dst] += 1
    if n == 0:
        return {
            "A_path": 0,
            "num_supersegments": 0,
            "argmax_supersegment_id": None,
            "mean_indegree": 0.0,
        }
    argmax = int(np.argmax(counts))
    return {
        "A_path": int(counts[argmax]),
        "num_supersegments": n,
        "argmax_supersegment_id": argmax,
        "mean_indegree": float(counts[:n].mean()),
    }


def build_adjacency(
    n_nodes: int,
    edges: List[Edge],
) -> tuple[List[List[int]], List[List[int]]]:
    """Convert edge list to (in_deps, out_deps) adjacency lists."""
    in_deps: List[List[int]] = [[] for _ in range(n_nodes)]
    out_deps: List[List[int]] = [[] for _ in range(n_nodes)]
    for e in edges:
        src, dst = int(e.src), int(e.dst)
        if 0 <= src < n_nodes and 0 <= dst < n_nodes:
            in_deps[dst].append(src)
            out_deps[src].append(dst)
    return in_deps, out_deps


__all__ = [
    "AABB",
    "DependencyModel", "REpsLookup", "LookupRuntime",
    "build_r_eps_lookup", "build_r_eps_lookup_numerical",
    "write_r_eps_lookup_csv",
    "aabb_distance_nd", "aabb_distance_m",
    "calibration_epsilon_for_rel_l2", "calibration_rel_l2_for_epsilon",
    "build_supersegment_dependency_edges",
    "compute_edge_indegree_summary",
    "build_adjacency",
]
