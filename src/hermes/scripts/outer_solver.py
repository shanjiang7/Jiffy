from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import cupy as cp
import numpy as np

from hermes.motion.executor import (
    apply_domain_movement,
    canonicalize_leg,
    prepare_movement_cache,
    reset_cache_origin,
    speed_key,
)
from hermes.motion.types import PathLeg
from hermes.grids.sim_params import init_level3_outer
from hermes.kernels.cg_update_cuda import call_cg_update_xr_reduce_rr
from hermes.kernels.matvec_cuda import call_mv_level3_dirichlet_with_dot_update_p
from hermes.kernels.rhs_matvec_fused_cuda import call_rhs_matvec_level3_fused_init
from hermes.runtime.gpu_setup import launch_3d_xyz

_L3_TX = 32
_L3_TY = 4
_L3_TZ = 4

# Movement caches are large because each distinct (|dx|, |dy|) speed keeps
# precomputed index slices plus halo buffers. Cap the number of resident cache
# entries so curved paths with many unique directions do not exhaust GPU memory.
_MOVEMENT_CACHE_MAX_ENTRIES = 16

_FUSED_ZERO_ITER_RTOL_F64 = 1e-12
_FUSED_ZERO_ITER_RTOL_F32 = 1e-6


@dataclass
class _FusedState:
    x: cp.ndarray
    r: cp.ndarray
    p: cp.ndarray
    p_tmp: cp.ndarray
    ap: cp.ndarray
    pap_buf: cp.ndarray   # 1-element float64: p^T A p
    rr_buf: cp.ndarray    # 1-element float64: r^T r
    blocks: tuple
    threads: tuple


@dataclass
class OuterContext:
    nx: int
    ny: int
    nz: int
    h: float
    h_isq: float
    dt_nd: float
    dt05_nd: float
    u0: float
    n1: float
    n2: float
    n3: float
    sigma_nd: float
    x_init_cp: cp.ndarray
    y_init_cp: cp.ndarray
    z_init_cp: cp.ndarray
    b: cp.ndarray
    solver_mode: str
    cg_tol: float
    cg_max_iter: int
    fused: _FusedState


def build_outer_context(
    rc,
    phys,
    float_type,
    dt_nd: float,
    solver_mode: str = "fused",
) -> OuterContext:
    solver_mode = str(solver_mode).lower()
    if solver_mode != "fused":
        raise ValueError(
            f"Unknown solver_mode='{solver_mode}'; only 'fused' is supported."
        )

    outer = init_level3_outer(
        phys=phys,
        float_type=float_type,
        lxd=rc.level3.lxd,
        lyd=rc.level3.lyd,
        lzd=rc.level3.lzd,
        h_m=rc.level3.h_tuple[0],
        dt=dt_nd,
        xp=cp,
    )

    nx = outer["sp"].nx
    ny = outer["sp"].ny
    nz = outer["sp"].nz
    h = float(outer["h_lin"])
    h_isq = float(outer["h_linisq"])
    dt05_nd = float(0.5 * dt_nd)
    u0 = float(outer["u0"])

    x_lin = outer["x_lin"]
    y_lin = outer["y_lin"]
    z_lin = outer["z_lin"]

    n_all = nx * ny * nz
    b = cp.empty((n_all,), dtype=float_type)

    fused_blocks, fused_threads = launch_3d_xyz(nx, ny, nz, tx=_L3_TX, ty=_L3_TY, tz=_L3_TZ)
    fused = _FusedState(
        x=cp.empty((n_all,), dtype=float_type),
        r=cp.empty((n_all,), dtype=float_type),
        p=cp.empty((n_all,), dtype=float_type),
        p_tmp=cp.empty((n_all,), dtype=float_type),
        ap=cp.empty((n_all,), dtype=float_type),
        pap_buf=cp.empty((1,), dtype=cp.float64),
        rr_buf=cp.empty((1,), dtype=cp.float64),
        blocks=fused_blocks,
        threads=fused_threads,
    )

    return OuterContext(
        nx=nx,
        ny=ny,
        nz=nz,
        h=h,
        h_isq=h_isq,
        dt_nd=float(dt_nd),
        dt05_nd=dt05_nd,
        u0=u0,
        n1=float(phys.n1),
        n2=float(phys.n2),
        n3=float(phys.n3),
        sigma_nd=float(rc.laser.x_span_m / phys.len_scale),
        x_init_cp=x_lin.copy(),
        y_init_cp=y_lin.copy(),
        z_init_cp=z_lin.copy(),
        b=b,
        solver_mode=solver_mode,
        cg_tol=float(rc.solver.cg_tol_level3),
        cg_max_iter=int(rc.solver.cg_max_iter_level3),
        fused=fused,
    )


def _solve_one_step(
    ctx: OuterContext,
    u: cp.ndarray,
    qs: cp.ndarray,
) -> cp.ndarray:
    return _solve_one_step_fused(ctx, u, qs)



def _solve_one_step_fused(
    ctx: OuterContext,
    u: cp.ndarray,
    qs: cp.ndarray,
    *,
    return_stats: bool = False,
) -> cp.ndarray | tuple[cp.ndarray, dict]:
    f = ctx.fused
    if f is None:
        raise RuntimeError("Fused context is not initialized.")

    f.x[...] = u

    # One pass: compute b = RHS(u, qs), r = b - A*u, p = r, ‖b‖², ‖r‖²
    call_rhs_matvec_level3_fused_init(
        nx=ctx.nx, ny=ctx.ny, nz=ctx.nz,
        u=f.x, qs=qs, b=ctx.b, r=f.r, p=f.p,
        hixsq=ctx.h_isq, hiysq=ctx.h_isq, hizsq=ctx.h_isq,
        dt05=ctx.dt05_nd, n2=ctx.n2, n3=ctx.n3, u0=ctx.u0, hz=ctx.h,
        b_dot_out=f.pap_buf, rr_out=f.rr_buf,
    )

    b_dot = float(f.pap_buf[0])
    rr = float(f.rr_buf[0])
    tol_sq = ctx.cg_tol ** 2 * b_dot
    stationary_rtol = (
        _FUSED_ZERO_ITER_RTOL_F32 if np.dtype(u.dtype) == np.dtype(cp.float32)
        else _FUSED_ZERO_ITER_RTOL_F64
    )
    stationary_tol_sq = stationary_rtol ** 2 * max(b_dot, 1.0)

    p, p_tmp = f.p, f.p_tmp
    beta = 0.0
    iters = 0

    while iters < ctx.cg_max_iter:
        # Warm-started cooling steps can have a small-but-nonzero initial residual.
        # Only permit 0-iteration exit when residual is at machine precision;
        # otherwise require at least one CG update before applying tol_sq.
        if rr <= tol_sq and (iters > 0 or rr <= stationary_tol_sq):
            break

        # Update p = r + beta*p (beta=0 on first iter → p stays = r),
        # compute ap = A*p, and accumulate pap = p^T A p into f.pap_buf.
        call_mv_level3_dirichlet_with_dot_update_p(
            f.blocks, f.threads,
            ctx.nx, ctx.ny, ctx.nz,
            f.r, p, p_tmp, f.ap,
            ctx.h_isq, ctx.h_isq, ctx.h_isq,
            ctx.dt05_nd, ctx.n2, ctx.u0, ctx.h,
            beta, f.pap_buf,
        )
        p, p_tmp = p_tmp, p   # p now holds the updated direction

        p_ap = float(f.pap_buf[0])
        if p_ap == 0.0:
            if rr <= tol_sq:
                break
            raise RuntimeError("Fused CG breakdown: p^T A p == 0.")

        alpha = rr / p_ap
        call_cg_update_xr_reduce_rr(x=f.x, r=f.r, p=p, ap=f.ap, alpha=alpha, rr_out=f.rr_buf)
        rr_new = float(f.rr_buf[0])
        beta = rr_new / rr
        rr = rr_new
        iters += 1

    if rr > tol_sq:
        raise RuntimeError(f"Fused CG did not converge (iters={iters}, maxiter={ctx.cg_max_iter}).")

    if return_stats:
        return f.x, {"cg_iters": iters, "mv_calls": iters + 1}
    return f.x


OuterContext.solve_one_step = _solve_one_step


def run_ss_outer(
    ctx: OuterContext,
    u_init: cp.ndarray,
    x_start: float,
    y_start: float,
    legs: list[PathLeg],
    steps_per_ss: int,
    *,
    source_on: bool = True,
    max_steps: int | None = None,
    snapshot_stride_steps: int | None = None,
    snapshot_steps: list[int] | None = None,
    snapshot_callback: Callable[[cp.ndarray], None] | None = None,
    profile: dict | None = None,
) -> tuple[list[cp.ndarray], cp.ndarray]:
    """
    Simulate one or more supersegments' worth of dt steps.

    When ``profile`` is a dict, per-call setup and movement-cache build time
    are accumulated into it (keys ``setup_seconds``/``cache_seconds``);
    behavior is otherwise unchanged.
    """
    _prof_t0 = time.perf_counter() if profile is not None else 0.0
    u = u_init.copy()
    x = float(x_start)
    y = float(y_start)

    x_arr = ctx.x_init_cp.copy() + x
    y_arr = ctx.y_init_cp.copy() + y
    z_arr = ctx.z_init_cp.copy()
    qs = cp.zeros((ctx.nx, ctx.ny), dtype=u.dtype)
    X_lin = Y_lin = None

    cache = None
    fast_caches: OrderedDict[tuple[float, float], dict[str, object]] = OrderedDict()
    if profile is not None:
        cp.cuda.Stream.null.synchronize()
        profile["setup_seconds"] = profile.get("setup_seconds", 0.0) + (
            time.perf_counter() - _prof_t0
        )
    snapshots: list[cp.ndarray] = []
    num_snapshots = 0
    global_step = 0
    explicit_snapshot_steps = snapshot_steps is not None
    if snapshot_steps is not None:
        planned_snapshot_steps = sorted({int(s) for s in snapshot_steps if int(s) >= 0})
        next_snapshot_idx = 0
        if max_steps is not None:
            planned_snapshot_steps = [s for s in planned_snapshot_steps if s < int(max_steps)]
    else:
        stride = int(snapshot_stride_steps) if snapshot_stride_steps is not None else int(steps_per_ss)
        if stride <= 0:
            raise ValueError("snapshot_stride_steps must be >= 1")
        planned_snapshot_steps = None
        next_snapshot_idx = 0

    def _emit_snapshot(u_snap: cp.ndarray) -> None:
        nonlocal num_snapshots
        if snapshot_callback is None:
            snapshots.append(u_snap.copy())
        else:
            snapshot_callback(u_snap)
        num_snapshots += 1

    for leg in legs:
        leg_steps = leg.steps
        if max_steps is not None:
            if global_step + leg_steps > max_steps:
                leg_steps = max_steps - global_step
            if leg_steps <= 0:
                break

        dx, dy = canonicalize_leg(leg.dx_step, leg.dy_step)
        vx = abs(dx)
        vy = abs(dy)
        if vx > 0 or vy > 0:
            key = speed_key(vx, vy)
            if key not in fast_caches:
                if len(fast_caches) >= _MOVEMENT_CACHE_MAX_ENTRIES:
                    _, evicted_cache = fast_caches.popitem(last=False)
                    del evicted_cache
                _prof_c0 = time.perf_counter() if profile is not None else 0.0
                fast_caches[key] = prepare_movement_cache(
                    nx=ctx.nx, ny=ctx.ny, nz=ctx.nz,
                    u=u, x_arr=x_arr, y_arr=y_arr, z_arr=z_arr,
                    vx=vx, vy=vy,
                )
                if profile is not None:
                    cp.cuda.Stream.null.synchronize()
                    profile["cache_seconds"] = profile.get("cache_seconds", 0.0) + (
                        time.perf_counter() - _prof_c0
                    )
                    profile["cache_builds"] = profile.get("cache_builds", 0.0) + 1.0
            else:
                fast_caches.move_to_end(key)
            cache = fast_caches[key]
            reset_cache_origin(cache, x_arr, y_arr, z_arr)

        leg_src = bool(leg.source_on) and bool(source_on)
        if leg_src:
            Y_lin, X_lin = cp.meshgrid(y_arr, x_arr)

        for _ in range(leg_steps):
            if planned_snapshot_steps is not None:
                while next_snapshot_idx < len(planned_snapshot_steps) and planned_snapshot_steps[next_snapshot_idx] == global_step:
                    _emit_snapshot(u)
                    next_snapshot_idx += 1
            else:
                if global_step % stride == 0:
                    _emit_snapshot(u)
            global_step += 1
            if cache is not None:
                apply_domain_movement(
                    ctx=ctx, u=u, x_arr=x_arr, y_arr=y_arr, z_arr=z_arr,
                    dx_step=dx, dy_step=dy, cache=cache,
                )
                if leg_src:
                    Y_lin, X_lin = cp.meshgrid(y_arr, x_arr)

            if leg_src:
                qs[:] = ctx.n1 * cp.exp(-2.0 * ((X_lin - x) ** 2 + (Y_lin - y) ** 2) / (ctx.sigma_nd ** 2))
            else:
                qs[:] = 0.0

            if profile is None:
                u = _solve_one_step(ctx, u, qs)
            else:
                u, _step_stats = _solve_one_step_fused(ctx, u, qs, return_stats=True)
                profile.setdefault("cg_iters", []).append(int(_step_stats["cg_iters"]))
            x += dx
            y += dy

        if max_steps is not None and global_step >= max_steps:
            break

    if num_snapshots == 0 and not explicit_snapshot_steps:
        _emit_snapshot(u)

    return snapshots, u.copy()
