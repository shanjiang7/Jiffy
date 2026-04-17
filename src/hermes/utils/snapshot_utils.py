"""Utilities for cropping and saving temperature-field snapshots."""
from __future__ import annotations

import numpy as np


def crop_snapshot(
    arr: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    h_m: float,
    roi_xy_m: float = 1e-3,
    roi_z_m: float = 1e-3,
) -> np.ndarray:
    """
    Crop a flat 1-D temperature snapshot to a centred roi_xy_m × roi_xy_m ROI
    in the x-y plane and keep only the top-surface-adjacent roi_z_m extent in
    z, then return it as a 3-D array.

    The flat array is stored in Fortran (column-major) order matching the GPU
    kernel layout.  The crop is centred on (nx//2, ny//2) and clamped to grid
    bounds so it never raises an IndexError regardless of grid size. The z crop
    keeps the last nz_crop cells, i.e. the sub-volume nearest the top surface.

    Parameters
    ----------
    arr   : flat 1-D array of length nx*ny*nz (Fortran order, host numpy)
    nx, ny, nz : grid cell counts
    h_m   : uniform physical cell spacing in metres
    roi_xy_m : side length of the square x-y ROI in metres (default 2 mm)
    roi_z_m  : retained z extent near the top surface in metres (default 1 mm)

    Returns
    -------
    3-D numpy array of shape (ncx, ncy, ncz) where:
    - ncx/ncy ≤ floor(roi_xy_m / h_m)
    - ncz ≤ floor(roi_z_m / h_m)
    """
    n_crop_xy = max(1, int(roi_xy_m / h_m))
    n_crop_z = max(1, int(roi_z_m / h_m))
    u3d = arr.reshape((nx, ny, nz), order="F")

    half = n_crop_xy // 2
    cx, cy = nx // 2, ny // 2

    x0, x1 = max(0, cx - half), min(nx, cx - half + n_crop_xy)
    y0, y1 = max(0, cy - half), min(ny, cy - half + n_crop_xy)
    z1 = nz
    z0 = max(0, z1 - n_crop_z)

    return u3d[x0:x1, y0:y1, z0:z1].copy()
