from __future__ import annotations

from typing import Any

import cupy as cp

from hermes.runtime.movement import (
    Index_3D_to_1D,
    update_after_movement_level3x_2,
    update_after_movement_level3x_2_negative,
    update_after_movement_level3y_2,
    update_after_movement_level3y_2_negative,
    xy_index,
    xy_index_negative,
)


def speed_key(vx: float, vy: float) -> tuple[float, float]:
    # Cache uses abs velocities because precompute prepares both +/- slices.
    return (round(float(vx), 15), round(float(vy), 15))


def canonicalize_leg(dx_step: float, dy_step: float) -> tuple[float, float]:
    dx = float(dx_step)
    dy = float(dy_step)
    scale = max(abs(dx), abs(dy))
    if scale == 0.0:
        return 0.0, 0.0
    tol = max(1e-20, 1e-12 * scale)
    if abs(dx) <= tol:
        dx = 0.0
    if abs(dy) <= tol:
        dy = 0.0
    return dx, dy


def reset_cache_origin(
    cache: dict[str, object],
    x_arr: cp.ndarray,
    y_arr: cp.ndarray,
    z_arr: cp.ndarray,
) -> None:
    cache["xoldmin"] = float(x_arr[0].item())
    cache["yoldmin"] = float(y_arr[0].item())
    cache["zoldmin"] = float(z_arr[0].item())


def prepare_movement_cache(
    *,
    nx: int,
    ny: int,
    nz: int,
    u: cp.ndarray,
    x_arr: cp.ndarray,
    y_arr: cp.ndarray,
    z_arr: cp.ndarray,
    vx: float,
    vy: float,
) -> dict[str, object]:
    array1 = cp.arange(nx * ny * nz).reshape([nx, ny, nz], order="F")
    threads = 128

    cache: dict[str, object] = {
        "vx": vx,
        "vy": vy,
        "threads_in": threads,
        "xoldmin": float(x_arr[0].item()),
        "yoldmin": float(y_arr[0].item()),
        "zoldmin": float(z_arr[0].item()),
    }

    if vy > 0:
        index_y_lin = xy_index(y_arr, vy)
        index_y_lin_neg = xy_index_negative(y_arr, vy)

        slice_1d_in_liny = Index_3D_to_1D(0, x_arr.shape[0], 0, index_y_lin - 1, array1)
        slice_1d_out_liny = Index_3D_to_1D(0, x_arr.shape[0], index_y_lin, y_arr.shape[0], array1)
        slice_1d_in_liny_negative = Index_3D_to_1D(0, x_arr.shape[0], index_y_lin_neg, y_arr.shape[0], array1)
        slice_1d_out_liny_negative = Index_3D_to_1D(0, x_arr.shape[0], 0, index_y_lin_neg - 1, array1)

        ny_in_pos = int(y_arr[0:index_y_lin].shape[0])
        ny_in_neg = int(y_arr[index_y_lin_neg:].shape[0])

        cache.update({
            "index_y_lin": index_y_lin,
            "index_y_lin_neg": index_y_lin_neg,
            "slice_1d_in_liny": slice_1d_in_liny,
            "slice_1d_out_liny": slice_1d_out_liny,
            "slice_1d_in_liny_negative": slice_1d_in_liny_negative,
            "slice_1d_out_liny_negative": slice_1d_out_liny_negative,
            "uin_y_pos": cp.zeros_like(u[slice_1d_in_liny], dtype=u.dtype),
            "uin_y_neg": cp.zeros_like(u[slice_1d_in_liny_negative], dtype=u.dtype),
            "ny_in_pos": ny_in_pos,
            "ny_in_neg": ny_in_neg,
            "blocks_in_y_pos": (nx * ny_in_pos * nz + (threads - 1)) // threads,
            "blocks_in_y_neg": (nx * ny_in_neg * nz + (threads - 1)) // threads,
        })

    if vx > 0:
        index_x_lin = xy_index(x_arr, vx)
        index_x_lin_neg = xy_index_negative(x_arr, vx)

        slice_1d_in_linx = Index_3D_to_1D(0, index_x_lin - 1, 0, y_arr.shape[0], array1)
        slice_1d_out_linx = Index_3D_to_1D(index_x_lin, x_arr.shape[0], 0, y_arr.shape[0], array1)
        slice_1d_in_linx_negative = Index_3D_to_1D(index_x_lin_neg, x_arr.shape[0], 0, y_arr.shape[0], array1)
        slice_1d_out_linx_negative = Index_3D_to_1D(0, index_x_lin_neg - 1, 0, y_arr.shape[0], array1)

        nx_in_pos = int(x_arr[0:index_x_lin].shape[0])
        nx_in_neg = int(x_arr[index_x_lin_neg:].shape[0])

        cache.update({
            "index_x_lin": index_x_lin,
            "index_x_lin_neg": index_x_lin_neg,
            "slice_1d_in_linx": slice_1d_in_linx,
            "slice_1d_out_linx": slice_1d_out_linx,
            "slice_1d_in_linx_negative": slice_1d_in_linx_negative,
            "slice_1d_out_linx_negative": slice_1d_out_linx_negative,
            "uin_x_pos": cp.zeros_like(u[slice_1d_in_linx], dtype=u.dtype),
            "uin_x_neg": cp.zeros_like(u[slice_1d_in_linx_negative], dtype=u.dtype),
            "nx_in_pos": nx_in_pos,
            "nx_in_neg": nx_in_neg,
            "blocks_in_x_pos": (nx_in_pos * ny * nz + (threads - 1)) // threads,
            "blocks_in_x_neg": (nx_in_neg * ny * nz + (threads - 1)) // threads,
        })

    return cache


def apply_domain_movement(
    *,
    ctx: Any,
    u: cp.ndarray,
    x_arr: cp.ndarray,
    y_arr: cp.ndarray,
    z_arr: cp.ndarray,
    dx_step: float,
    dy_step: float,
    cache: dict[str, object],
) -> None:
    vy = float(cache["vy"])
    vx = float(cache["vx"])

    if dy_step > 0 and vy > 0:
        y_arr += vy
        update_after_movement_level3y_2(
            x=x_arr,
            y=y_arr,
            z=z_arr,
            val=u,
            nx_old=ctx.nx,
            ny_old=ctx.ny,
            nz_old=ctx.nz,
            hxval=ctx.h,
            hyval=ctx.h,
            hzval=ctx.h,
            xoldmin_lin=float(cache["xoldmin"]),
            yoldmin_lin=float(cache["yoldmin"]),
            zoldmin_lin=float(cache["zoldmin"]),
            nx=ctx.nx,
            ny_in=int(cache["ny_in_pos"]),
            nz=ctx.nz,
            uin=cache["uin_y_pos"],
            slice_1d_in=cache["slice_1d_in_liny"],
            slice_1d_out=cache["slice_1d_out_liny"],
            index_y=int(cache["index_y_lin"]),
            u0=ctx.u0,
            one=1,
            val3=u,
            threads_per_block_in=int(cache["threads_in"]),
            blocks_per_grid_in=int(cache["blocks_in_y_pos"]),
        )
        cache["yoldmin"] = float(cache["yoldmin"]) + vy
    elif dy_step < 0 and vy > 0:
        y_arr -= vy
        update_after_movement_level3y_2_negative(
            x=x_arr,
            y=y_arr,
            z=z_arr,
            val=u,
            nx_old=ctx.nx,
            ny_old=ctx.ny,
            nz_old=ctx.nz,
            hxval=ctx.h,
            hyval=ctx.h,
            hzval=ctx.h,
            xoldmin_lin=float(cache["xoldmin"]),
            yoldmin_lin=float(cache["yoldmin"]),
            zoldmin_lin=float(cache["zoldmin"]),
            nx=ctx.nx,
            ny_in=int(cache["ny_in_neg"]),
            nz=ctx.nz,
            uin=cache["uin_y_neg"],
            slice_1d_in=cache["slice_1d_in_liny_negative"],
            slice_1d_out=cache["slice_1d_out_liny_negative"],
            index_y=int(cache["index_y_lin_neg"]),
            u0=ctx.u0,
            one=1,
            val3=u,
            threads_per_block_in=int(cache["threads_in"]),
            blocks_per_grid_in=int(cache["blocks_in_y_neg"]),
        )
        cache["yoldmin"] = float(cache["yoldmin"]) - vy

    if dx_step > 0 and vx > 0:
        x_arr += vx
        update_after_movement_level3x_2(
            x=x_arr,
            y=y_arr,
            z=z_arr,
            val=u,
            nx_old=ctx.nx,
            ny_old=ctx.ny,
            nz_old=ctx.nz,
            hxval=ctx.h,
            hyval=ctx.h,
            hzval=ctx.h,
            xoldmin_lin=float(cache["xoldmin"]),
            yoldmin_lin=float(cache["yoldmin"]),
            zoldmin_lin=float(cache["zoldmin"]),
            nx_in=int(cache["nx_in_pos"]),
            ny=ctx.ny,
            nz=ctx.nz,
            uin=cache["uin_x_pos"],
            slice_1d_in=cache["slice_1d_in_linx"],
            slice_1d_out=cache["slice_1d_out_linx"],
            index_x=int(cache["index_x_lin"]),
            u0=ctx.u0,
            one=1,
            val3=u,
            threads_per_block_in=int(cache["threads_in"]),
            blocks_per_grid_in=int(cache["blocks_in_x_pos"]),
        )
        cache["xoldmin"] = float(cache["xoldmin"]) + vx
    elif dx_step < 0 and vx > 0:
        x_arr -= vx
        update_after_movement_level3x_2_negative(
            x=x_arr,
            y=y_arr,
            z=z_arr,
            val=u,
            nx_old=ctx.nx,
            ny_old=ctx.ny,
            nz_old=ctx.nz,
            hxval=ctx.h,
            hyval=ctx.h,
            hzval=ctx.h,
            xoldmin_lin=float(cache["xoldmin"]),
            yoldmin_lin=float(cache["yoldmin"]),
            zoldmin_lin=float(cache["zoldmin"]),
            nx_in=int(cache["nx_in_neg"]),
            ny=ctx.ny,
            nz=ctx.nz,
            uin=cache["uin_x_neg"],
            slice_1d_in=cache["slice_1d_in_linx_negative"],
            slice_1d_out=cache["slice_1d_out_linx_negative"],
            index_x=int(cache["index_x_lin_neg"]),
            u0=ctx.u0,
            one=1,
            val3=u,
            threads_per_block_in=int(cache["threads_in"]),
            blocks_per_grid_in=int(cache["blocks_in_x_neg"]),
        )
        cache["xoldmin"] = float(cache["xoldmin"]) - vx


__all__ = [
    "apply_domain_movement",
    "canonicalize_leg",
    "prepare_movement_cache",
    "reset_cache_origin",
    "speed_key",
]
