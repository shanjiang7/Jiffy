from __future__ import annotations

import time

import cupy as cp
import numpy as np

from hermes.motion.executor import (
    apply_domain_movement,
    canonicalize_leg,
    prepare_movement_cache,
    reset_cache_origin,
    speed_key,
)


def run_sequential_outer(
    ctx,
    u_init: cp.ndarray,
    x_start: float,
    y_start: float,
    path_defs,
    steps_per_ss: int,
    snapshot_stride_steps: int | None = None,
    snapshot_steps_by_component: list[list[int]] | None = None,
) -> list[list[np.ndarray]]:
    """
    Run the sequential simulation by computing each PathDef (Component) sequentially.
    Returns a list of snapshot lists; each snapshot is the temperature field
    at the beginning of the corresponding supersegment.
    """
    u = u_init.copy()
    x_arr = ctx.x_init_cp.copy()
    y_arr = ctx.y_init_cp.copy()
    z_arr = ctx.z_init_cp.copy()
    qs = cp.zeros((ctx.nx, ctx.ny), dtype=u.dtype)

    total_ss = sum(path_def.weight for path_def in path_defs)
    snapshots: list[list[np.ndarray]] = []

    x = float(x_start)
    y = float(y_start)
    x_arr += x
    y_arr += y
    X_lin = Y_lin = None

    completed_ss = 0
    stride = int(snapshot_stride_steps) if snapshot_stride_steps is not None else int(steps_per_ss)
    if stride <= 0:
        raise ValueError("snapshot_stride_steps must be >= 1")

    for comp_idx, path_def in enumerate(path_defs):
        comp_start_t = time.perf_counter()
        comp_snaps: list[np.ndarray] = []
        global_step = 0
        if snapshot_steps_by_component is not None:
            comp_snapshot_steps = sorted({int(s) for s in snapshot_steps_by_component[comp_idx] if int(s) >= 0})
            next_snapshot_idx = 0
        else:
            comp_snapshot_steps = None
            next_snapshot_idx = 0
        cache = None
        fast_caches: dict[tuple[float, float], dict[str, object]] = {}
        print(f"[serial] comp {comp_idx}  legs={len(path_def.legs)}", flush=True)
        # for i, leg in enumerate(path_def.legs):
        #     print(f"  leg {i:3d}  dx={leg.dx_step:.6e}  dy={leg.dy_step:.6e}  steps={leg.steps}  src={leg.source_on}", flush=True)
        for leg in path_def.legs:
            cur_dx, cur_dy = canonicalize_leg(leg.dx_step, leg.dy_step)
            vx = abs(cur_dx)
            vy = abs(cur_dy)
            if vx > 0 or vy > 0:
                key = speed_key(vx, vy)
                if key not in fast_caches:
                    fast_caches[key] = prepare_movement_cache(
                        nx=ctx.nx,
                        ny=ctx.ny,
                        nz=ctx.nz,
                        u=u,
                        x_arr=x_arr,
                        y_arr=y_arr,
                        z_arr=z_arr,
                        vx=vx,
                        vy=vy,
                    )
                cache = fast_caches[key]
                reset_cache_origin(cache, x_arr, y_arr, z_arr)

            if leg.source_on:
                Y_lin, X_lin = cp.meshgrid(y_arr, x_arr)

            for _ in range(leg.steps):
                should_capture = False
                if comp_snapshot_steps is not None:
                    while next_snapshot_idx < len(comp_snapshot_steps) and comp_snapshot_steps[next_snapshot_idx] == global_step:
                        comp_snaps.append(cp.asnumpy(u))
                        next_snapshot_idx += 1
                        should_capture = True
                elif global_step % stride == 0:
                    comp_snaps.append(cp.asnumpy(u))
                    should_capture = True
                if should_capture:
                    elapsed = time.perf_counter() - comp_start_t
                    if comp_snapshot_steps is None and stride == steps_per_ss:
                        completed_ss += 1
                        print(
                            f"[serial] comp {comp_idx}  SS {len(comp_snaps)}/{path_def.weight}"
                            f"  ({completed_ss}/{total_ss} global)  +{elapsed:.1f}s",
                            flush=True,
                        )
                global_step += 1
                if cache is not None:
                    apply_domain_movement(
                        ctx=ctx,
                        u=u,
                        x_arr=x_arr,
                        y_arr=y_arr,
                        z_arr=z_arr,
                        dx_step=cur_dx,
                        dy_step=cur_dy,
                        cache=cache,
                    )
                    if leg.source_on:
                        Y_lin, X_lin = cp.meshgrid(y_arr, x_arr)

                if leg.source_on:
                    qs[:] = ctx.n1 * cp.exp(-2.0 * ((X_lin - x) ** 2 + (Y_lin - y) ** 2) / (ctx.sigma_nd ** 2))
                else:
                    qs[:] = 0.0

                u = ctx.solve_one_step(u, qs)
                x += cur_dx
                y += cur_dy

        if len(comp_snaps) == 0 and comp_snapshot_steps is None:
            comp_snaps.append(cp.asnumpy(u))
        print(f"[serial] comp {comp_idx} done in {time.perf_counter()-comp_start_t:.1f}s", flush=True)
        snapshots.append(comp_snaps)

    return snapshots
