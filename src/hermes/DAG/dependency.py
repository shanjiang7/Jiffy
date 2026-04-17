"""
Thermal dependency model and supersegment DAG construction.

Public API
----------
AABB, DependencyModel, REpsLookup, LookupRuntime
build_r_eps_lookup, build_r_eps_lookup_analytical, build_r_eps_lookup_numerical
write_r_eps_lookup_csv
aabb_distance_nd, aabb_distance_m
build_supersegment_dependency_edges
build_adjacency
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import List

import numpy as np

from hermes.DAG.solution import EagarTsai
from hermes.utils.dag_utils import Edge
from hermes.utils.segment_types import Segment, SuperSegment

AABB = tuple[float, float, float, float]


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
    backend: str = "analytical",
    runtime: LookupRuntime | None = None,
) -> REpsLookup:
    backend_norm = str(backend).strip().lower()
    if backend_norm == "analytical":
        return build_r_eps_lookup_analytical(
            model=model,
            dt_s=dt_s,
            V_mps=V_mps,
            P_W=P_W,
            max_steps=max_steps,
        )
    if backend_norm == "numerical":
        return build_r_eps_lookup_numerical(
            model=model,
            dt_s=dt_s,
            V_mps=V_mps,
            P_W=P_W,
            max_steps=max_steps,
            runtime=runtime,
        )
    raise ValueError(f"Unsupported lookup backend {backend!r}; expected 'analytical' or 'numerical'.")


def build_r_eps_lookup_analytical(
    *,
    model: DependencyModel,
    dt_s: float,
    V_mps: float,
    P_W: float,
    max_steps: int,
) -> REpsLookup:
    """
    Build the r_eps lookup table using the EagarTsai analytical solver.

    Uses a persistent disk cache mapping physical parameters to `.hermes_cache`
    to skip re-computing the multi-step diffusion simulation.
    """
    params = {
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
        "max_steps": int(max_steps),
    }
    
    param_str = json.dumps(params, sort_keys=True)
    h = hashlib.md5(param_str.encode("utf-8")).hexdigest()

    cache_dir = Path(".hermes_cache").expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"r_eps_{h}.npy"
    
    if cache_file.exists():
        r_eps = tuple(float(x) for x in np.load(cache_file))
    else:
        r_eps = _r_eps_cached(**params)
        np.save(cache_file, np.array(r_eps, dtype=float))

    return REpsLookup(
        dt_s=float(dt_s),
        V_mps=float(V_mps),
        P_W=float(P_W),
        r_eps_m=r_eps,
    )


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
) -> float:
    import cupy as cp

    u3d = u.reshape((ctx.nx, ctx.ny, ctx.nz), order="F")
    delta_top_K = (u3d[:, :, -1] - ctx.u0) * float(phys.deltaT)
    mask = delta_top_K >= float(level_K)
    if not bool(cp.any(mask).item()):
        return 0.0

    x_m = ctx.x_init_cp[:, None] * float(phys.len_scale)
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
        "max_steps": int(max_steps),
        "solver_mode": solver_mode,
        "source_on_steps": int(source_on_steps),
        "source_substeps": int(source_substeps),
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
    r_eps_arr = np.zeros(max_steps + 1, dtype=float)

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

    r_eps_arr[0] = _extract_r_eps_m_numerical(
        ctx=ctx,
        u=u,
        phys=runtime.phys,
        level_K=model.level_K,
    )
    for k in range(1, max_steps + 1):
        u = ctx.solve_one_step(u, qs_off)
        r_eps_arr[k] = _extract_r_eps_m_numerical(
            ctx=ctx,
            u=u,
            phys=runtime.phys,
            level_K=model.level_K,
        )
        if k > 0 and float(r_eps_arr[k]) == 0.0:
            break

    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    r_eps = tuple(float(x) for x in r_eps_arr.tolist())
    np.save(cache_file, np.asarray(r_eps, dtype=float))
    return REpsLookup(
        dt_s=float(dt_s),
        V_mps=float(V_mps),
        P_W=float(P_W),
        r_eps_m=r_eps,
    )



@lru_cache(maxsize=128)
def _r_eps_cached(
    dt_s: float,
    level_K: float, resolution_m: float, bc: str, spacing_m: float,
    V_mps: float, P_W: float,
    window_x_um: int, window_y_um: int, window_z_um: int,
    max_steps: int,
) -> tuple[float, ...]:
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError(f"dt_s must be finite and > 0; got {dt_s!r}")
    if max_steps < 0:
        raise ValueError("max_steps must be >= 0")

    V, P = float(V_mps), float(P_W)
    seg_len = V * dt_s
    line = EagarTsai(
        resolution=float(resolution_m), V=V, bc=str(bc), spacing=float(spacing_m),
        depth=float(window_z_um) * 1e-6,
        x_min=-(float(window_x_um) * 0.5e-6), x_max=+(float(window_x_um) * 0.5e-6),
        y_min=-(float(window_y_um) * 0.5e-6), y_max=+(float(window_y_um) * 0.5e-6),
        init_location=(-0.5 * seg_len, 0.0),
    )
    line.forward(dt_s, 0.0, V=V, P=P)

    xs, ys = np.asarray(line.xs, dtype=float), np.asarray(line.ys, dtype=float)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    R = np.sqrt(X * X + Y * Y)
    z_top = -1

    r_eps = np.zeros(max_steps + 1, dtype=float)
    for k in range(max_steps + 1):
        if k > 0:
            line.forward_diffuse_only(dt_s, 0.0, V=V, P=P)
        delta = np.asarray(line.theta[:, :, z_top], dtype=float) - 300.0
        mask = delta >= float(level_K)
        r_eps[k] = float(np.max(R[mask])) if np.any(mask) else 0.0
        if k > 0 and float(r_eps[k]) == 0.0:
            break

    return tuple(float(x) for x in r_eps.tolist())


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

    Iterates over individual Segments internally; for each segment j, checks the
    back_window previous segments using per-segment AABBs and elapsed times.
    Any segment pair from different SSes that meets the thermal proximity condition
    produces an SS-level edge.  Duplicate (SS_i, SS_j) pairs are deduplicated.

    back_window : maximum number of previous *segments* to check (not SSes).
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
    "build_r_eps_lookup", "build_r_eps_lookup_analytical", "build_r_eps_lookup_numerical",
    "write_r_eps_lookup_csv",
    "aabb_distance_nd", "aabb_distance_m",
    "build_supersegment_dependency_edges",
    "build_adjacency",
]
