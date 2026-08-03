#!/usr/bin/env python3
"""
Build the ParaView bundle for the 15-layer x 64-rank hero visualization.

Everything is real data, no new solver runs:
  - partition: the exact_dp 64-rank plan for the 15-layer bull
    (outputs/cr_strong_scaling_ml15/bull/plan_dump_64r/plan_bounds.json)
  - temperature: the stride-10 serial reference windows (1 mm ROI of the
    comoving domain, h = 30 um). Every layer traces the same path, so a
    rank's mid-chunk instant in layer L maps to the same within-layer step
    of the single-layer serial run.

Output (outputs/pv_bundle_ml15_bull/):
  path_layer_XX.vtk   per-layer laser path, point scalar 'rank' (global cuts)
  pool_rank_XX.vtk    per-rank 1 mm temperature window at its true mid-chunk
                      laser position, layer-offset in z
  README.txt          provenance + the disclosure notes for the caption

Layer spacing in z is exaggerated (DZ_MM per layer vs 50 um physical) for
visibility; state this in the caption.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hermes.laser_path.path_loader import build_path_sections_nd_from_ini

PLAN = Path("outputs/cr_strong_scaling_ml15/bull/plan_dump_64r/plan_bounds.json")
SNAP = Path("outputs/accuracy_bull_tol1e4_h30/serial_s10/snapshots_ser")
OUT = Path("outputs/pv_bundle_ml15_bull")
N_LAYERS = 15
STRIDE = 10
LAYER_STEPS = 218_160   # steps per layer in the serial reference run
H_MM = 0.030            # snapshot grid spacing [mm]
DZ_MM = 3.0             # exploded layer spacing [mm] (physical: 0.05 mm)


def write_structured_points(path, arr, origin, spacing, name="excessT"):
    nx, ny, nz = arr.shape
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\n"
                 f"{path.stem}\n"
                 "ASCII\nDATASET STRUCTURED_POINTS\n"
                 f"DIMENSIONS {nx} {ny} {nz}\n"
                 f"ORIGIN {origin[0]:.4f} {origin[1]:.4f} {origin[2]:.4f}\n"
                 f"SPACING {spacing} {spacing} {spacing}\n"
                 f"POINT_DATA {arr.size}\n"
                 f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
        np.savetxt(fh, arr.reshape(-1, order="F")[:, None], fmt="%.4g")


def write_polyline(path, pts_xyz, scalars, scalar_name="rank"):
    n = len(pts_xyz)
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\n"
                 f"{path.stem}\nASCII\nDATASET POLYDATA\n"
                 f"POINTS {n} float\n")
        np.savetxt(fh, pts_xyz, fmt="%.4f")
        fh.write(f"LINES 1 {n + 1}\n{n} " + " ".join(map(str, range(n))) + "\n")
        fh.write(f"POINT_DATA {n}\nSCALARS {scalar_name} float 1\n"
                 "LOOKUP_TABLE default\n")
        np.savetxt(fh, np.asarray(scalars)[:, None], fmt="%.2f")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bounds = np.array(json.load(open(PLAN)))          # (64, 2) SS fractions
    n_ranks = len(bounds)
    edges = np.concatenate([[0.0], bounds[:, 1]])

    # One layer of the bull path (mm) + arclength.
    pts = np.vstack([a for a, on in build_path_sections_nd_from_ini(
        "configs/examples/fast_heat.ini", len_scale=1.0) if on]) * 1e3
    s = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    layer_len = s[-1]

    def pos_at(frac_in_layer: float) -> np.ndarray:
        target = frac_in_layer * layer_len
        i = int(np.clip(np.searchsorted(s, target), 1, len(s) - 1))
        f = (target - s[i - 1]) / max(s[i] - s[i - 1], 1e-12)
        return pts[i - 1] * (1 - f) + pts[i] * f

    # Per-layer path files, colored by the GLOBAL rank along the build.
    for l in range(N_LAYERS):
        frac = (l + s / layer_len) / N_LAYERS
        rank = np.clip(np.searchsorted(edges, frac, side="right") - 1,
                       0, n_ranks - 1)
        xyz = np.column_stack([pts, np.full(len(pts), l * DZ_MM)])
        write_polyline(OUT / f"path_layer_{l:02d}.vtk", xyz, rank)

    # Per-rank pool windows at mid-chunk instants.
    half = 16 * H_MM
    nz_span = 32 * H_MM
    meta = []
    for r in range(n_ranks):
        mid = (bounds[r, 0] + bounds[r, 1]) / 2.0    # fraction of build
        layer = min(int(mid * N_LAYERS), N_LAYERS - 1)
        g = mid * N_LAYERS - layer                    # fraction within layer
        step = int(round(g * LAYER_STEPS / STRIDE) * STRIDE)
        step = min(step, LAYER_STEPS - 60)
        a = np.load(SNAP / f"layer_00_step_{step:09d}.npy")
        # Solver field is nondimensional u = (T - Ts)/deltaT (Ts=1658 K
        # solidus, deltaT=65 K); ambient u0 = -20.92. Export absolute Kelvin.
        w = (298.0 + np.clip(a - a.min(), 0.0, None) * 65.0).astype(np.float32)
        p = pos_at(g)
        origin = (p[0] - half, p[1] - half, layer * DZ_MM - nz_span)
        write_structured_points(OUT / f"pool_rank_{r:02d}.vtk", w,
                                origin, H_MM, name="T_K")
        meta.append(dict(rank=r, layer=layer, frac_in_layer=round(g, 4),
                         serial_step=step, x_mm=round(float(p[0]), 3),
                         y_mm=round(float(p[1]), 3)))
    json.dump(meta, open(OUT / "pools_meta.json", "w"), indent=1)

    (OUT / "README.txt").write_text(f"""\
ParaView bundle: 15-layer bull build, 64-rank parallel-in-time partition
========================================================================
Open scene.pvsm in ParaView 5.13 (File > Load State; if asked, point the
data directory at this folder). All data are from real runs:

- path_layer_XX.vtk : the laser path of each layer (identical geometry per
  layer), point scalar 'rank' = owning MPI rank from the REAL exact_dp
  64-rank partition of the 15-layer build (plan_dump_64r, predicted skew
  1.001). Color by 'rank' to see the time-partition: cuts fall mid-layer.
- pool_rank_XX.vtk : the temperature field around rank XX's laser at the
  midpoint of its chunk, placed at the true path position. Scalar 'T_K' =
  absolute temperature [K], converted from the solver's nondimensional
  field u = (T - Ts)/deltaT (Ts = 1658 K solidus, deltaT = 65 K, ambient
  298 K). Solidus 1658 K / liquidus 1723 K are the physical melt bounds.
  Peak source-cell values (~30,000 K) are the usual pure-conduction
  overshoot (no evaporation sink in the model); color maps should clamp
  around ~2,500 K.

Disclosure notes for the paper caption:
1. Each box is the central 1 mm (33^3, h = 30 um) sampled region of the
   rank's 4.8 x 4.8 x 2.4 mm comoving solver domain; the field outside
   this region is ambient to within the correction threshold.
2. Fields are taken from the stride-10 serial reference run of the layer
   path; every layer traces the same path, so a rank's mid-chunk instant
   maps to a within-layer step of that run (pools_meta.json lists the
   mapping). The melt pool is quasi-steady, so these are the same fields
   each rank's local solve produces at that instant.
3. Layer spacing is exaggerated to {DZ_MM} mm (physical: 0.05 mm) so the
   {N_LAYERS} layers are individually visible.

The message: at any wall-clock instant of the parallel run, all 64 melt
pools exist SIMULTANEOUSLY - each rank marches its own comoving domain
over its chunk of build time. A serial solver has exactly one.
""")
    print(f"[ok] bundle in {OUT}: {N_LAYERS} paths + {n_ranks} pools")


if __name__ == "__main__":
    main()
