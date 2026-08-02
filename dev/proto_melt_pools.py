#!/usr/bin/env python3
"""
Prototype: "one wall-clock instant, N simultaneous melt pools" from existing
serial snapshots (single-layer bull, stride-10) + the real exact_dp partition.

The solver's snapshot is a laser-comoving ~1 mm window (33^3, level-3 h)
around the melt pool. For each rank r with step range [t0, t1) we take the
window at its mid-chunk instant t_mid (excess over ambient) and place it at
the laser's position on the path at that instant — that is exactly what rank
r's local solve is computing at that wall-clock moment. Serial has ONE such
window; the parallel run has N at once.

Outputs (outputs/proto_melt_pools/):
  composite.png     global layer view: rank-colored path + N glowing pools
  pool_zoom.png     one real pool window, all three projections
  rank_<r>.vtk      per-rank volume w/ physical ORIGIN+SPACING (ParaView)
  path_rank.csv     path polyline with rank ids (for a ParaView .vtp later)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from hermes.laser_path.path_loader import build_path_sections_nd_from_ini

ROOT = Path("outputs/accuracy_bull_tol1e4_h30")
SNAP = ROOT / "serial_s10" / "snapshots_ser"
PLAN = ROOT / "par8_s10_w021" / "planning_summary.json"
OUT = Path("outputs/proto_melt_pools")
STRIDE = 10
H_MM = 0.030  # level-3 spacing for the h30 accuracy grid [mm]
TOTAL_STEPS = 218_160


def load_window(step: int) -> np.ndarray:
    step = min(int(round(step / STRIDE) * STRIDE), 218_100)
    a = np.load(SNAP / f"layer_00_step_{step:09d}.npy")
    return np.clip(a - a.min(), 0.0, None)  # excess over ambient


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plan = json.load(open(PLAN))
    sps = plan["steps_per_ss"]
    ra = plan["partition"]["rank_assignments"]
    ranks = sorted(ra, key=int)
    n_ranks = len(ranks)
    cmap = plt.get_cmap("turbo")

    # Path polyline (mm) + cumulative arclength -> position at any step.
    sections = build_path_sections_nd_from_ini(
        "configs/examples/fast_heat.ini", len_scale=1.0)
    pts = np.vstack([a for a, on in sections if on]) * 1e3
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])

    def pos_at(step: int) -> np.ndarray:
        target = step / TOTAL_STEPS * s[-1]
        i = np.searchsorted(s, target)
        i = np.clip(i, 1, len(s) - 1)
        f = (target - s[i - 1]) / max(s[i] - s[i - 1], 1e-12)
        return pts[i - 1] * (1 - f) + pts[i] * f

    # Per-rank: mid-chunk instant, window, laser position.
    pools, mids, t_mids = [], [], []
    for r in ranks:
        ss = ra[r]
        t0, t1 = min(ss) * sps, (max(ss) + 1) * sps
        t_mid = (t0 + t1) // 2
        w = load_window(t_mid)
        p = pos_at(t_mid)
        pools.append(w)
        mids.append(p)
        t_mids.append(t_mid)
        print(f"rank {r}: mid step {t_mid}  laser at ({p[0]:.2f}, {p[1]:.2f}) mm"
              f"  pool max {w.max():.0f} K")

    # ---- composite.png: rank-colored path + pools at true positions ----
    fig, ax = plt.subplots(figsize=(11, 9))
    frac_mid = (s[:-1] + s[1:]) / 2 / s[-1]
    rank_of = np.clip((frac_mid * n_ranks).astype(int), 0, n_ranks - 1)
    lc = LineCollection(np.stack([pts[:-1], pts[1:]], axis=1),
                        colors=cmap(rank_of / (n_ranks - 1)),
                        linewidths=0.5, alpha=0.35)
    ax.add_collection(lc)
    half = 16 * H_MM
    for i, (w, p, r) in enumerate(zip(pools, mids, ranks)):
        proj = w.max(axis=2)  # assume axis 2 ~ depth
        col = np.array(cmap(i / (n_ranks - 1))[:3])
        norm = (proj / max(proj.max(), 1e-9)) ** 0.5
        rgba = np.zeros(proj.shape + (4,))
        rgba[..., :3] = col
        rgba[..., 3] = norm
        ax.imshow(rgba.transpose(1, 0, 2), origin="lower", zorder=5,
                  extent=[p[0] - half, p[0] + half, p[1] - half, p[1] + half])
        ax.annotate(f"rank {r}", p, textcoords="offset points",
                    xytext=(8, 8), fontsize=9, color=col * 0.8)
        ax.add_patch(plt.Circle(p, 0.55, fill=False, ec=col, lw=1.2,
                                alpha=0.9, zorder=6))
    ax.set_xlim(pts[:, 0].min() - 1, pts[:, 0].max() + 1)
    ax.set_ylim(pts[:, 1].min() - 1, pts[:, 1].max() + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"one wall-clock instant: {n_ranks} rank-local melt pools "
                 "(real fields, real exact_dp cuts)\ncircles mark the ~1 mm "
                 "comoving windows, enlarged glow for visibility")
    fig.tight_layout()
    fig.savefig(OUT / "composite.png", dpi=170)
    print(f"[ok] {OUT/'composite.png'}")

    # ---- pool_zoom.png: one real window, three projections ----
    w = pools[3]
    fig2, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis in range(3):
        im = axes[axis].imshow(w.max(axis=axis).T, origin="lower",
                               cmap="inferno",
                               extent=[0, w.shape[0] * H_MM * 1e3,
                                       0, w.shape[0] * H_MM * 1e3])
        axes[axis].set_title(f"max-projection axis {axis}")
        axes[axis].set_xlabel("µm")
    fig2.colorbar(im, ax=axes, shrink=0.8, label="excess T [K]")
    fig2.suptitle("one rank's comoving window (real level-3 field)")
    fig2.savefig(OUT / "pool_zoom.png", dpi=150)
    print(f"[ok] {OUT/'pool_zoom.png'}")

    # ---- VTK export with physical placement (mm units) ----
    for i, (w, p, r) in enumerate(zip(pools, mids, ranks)):
        nx, ny, nz = w.shape
        origin = (p[0] - half, p[1] - half, -nz * H_MM)
        with open(OUT / f"rank_{int(r):02d}.vtk", "w") as fh:
            fh.write("# vtk DataFile Version 3.0\n"
                     f"rank {r} pool at step {t_mids[i]}\n"
                     "ASCII\nDATASET STRUCTURED_POINTS\n"
                     f"DIMENSIONS {nx} {ny} {nz}\n"
                     f"ORIGIN {origin[0]:.4f} {origin[1]:.4f} {origin[2]:.4f}\n"
                     f"SPACING {H_MM} {H_MM} {H_MM}\n"
                     f"POINT_DATA {w.size}\n"
                     "SCALARS excessT float 1\nLOOKUP_TABLE default\n")
            np.savetxt(fh, w.reshape(-1, order="F")[:, None], fmt="%.4g")
    np.savetxt(OUT / "path_rank.csv",
               np.column_stack([pts[:-1], rank_of]),
               header="x_mm,y_mm,rank", delimiter=",", comments="")
    print(f"[ok] wrote {n_ranks} .vtk volumes + path_rank.csv")


if __name__ == "__main__":
    main()
