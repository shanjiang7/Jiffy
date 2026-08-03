#!/usr/bin/env python3
"""
Hero-figure concept: 15-layer build as ONE time axis, partitioned across ranks.

Panel A (3D): the 15-layer Bull build, exploded in z. The laser path is a single
timeline threading through all layers; color = owning rank. Rank boundaries fall
mid-layer (time partition, not layer partition); short red tails behind each cut
mark the eps-horizon correction sources the next rank must replay.

Panel B: the same 64 colored chunks laid end-to-end (serial wall clock) and
folded into a concurrent block (parallel wall clock), annotated with the
measured 55.5x.

Mockup uses a uniform 64-way split; swap in the real exact_dp boundaries via
--plan-json (a JSON list of the 64 (start_ss, end_ss) global SS ranges).

Usage:
    python experiments/plot_multilayer_hero.py [--ranks 64] [--layers 15]
        [--out outputs/ml15_parallel_hero.png]
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from hermes.laser_path.path_loader import build_path_sections_nd_from_ini

BASELINE_S = 12572.6  # measured 15-layer Bull 1-rank wall clock
PAR64_S = 226.4       # measured 64-rank exact_dp wall clock (55.5x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/examples/fast_heat.ini")
    ap.add_argument("--ranks", type=int, default=64)
    ap.add_argument("--layers", type=int, default=15)
    ap.add_argument("--downsample", type=int, default=3)
    ap.add_argument("--plan-json", default=None,
                    help="Optional real partition: JSON [[start_frac, end_frac], ...] per rank")
    ap.add_argument("--out", default="outputs/ml15_parallel_hero.png")
    args = ap.parse_args()

    sections = build_path_sections_nd_from_ini(args.config, len_scale=1.0)
    pts = np.vstack([a for a, on in sections if on])[:: args.downsample] * 1e3  # mm
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])  # arclength within one layer
    layer_len = s[-1]
    total_len = layer_len * args.layers

    # Global time fraction -> rank id.
    if args.plan_json:
        bounds = np.array(json.load(open(args.plan_json)))  # (R, 2) fractions
        edges = np.concatenate([[0.0], bounds[:, 1]])
    else:
        edges = np.linspace(0.0, 1.0, args.ranks + 1)

    def rank_of(frac):
        return np.clip(np.searchsorted(edges, frac, side="right") - 1, 0, args.ranks - 1)

    cmap = plt.get_cmap("turbo")
    dz = 9.0  # visual layer spacing, mm

    fig = plt.figure(figsize=(10.5, 11.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.0], hspace=0.14)
    ax3 = fig.add_subplot(gs[0], projection="3d")

    horizon_mm = 0.012 * layer_len  # illustrative eps-horizon tail behind a cut

    for l in range(args.layers):
        frac = (l * layer_len + s) / total_len
        ranks = rank_of(frac)
        mid_frac = (frac[:-1] + frac[1:]) / 2.0
        colors = cmap(rank_of(mid_frac) / (args.ranks - 1))
        z = np.full(pts.shape[0], l * dz)
        segs = np.stack([np.column_stack([pts[:-1], z[:-1]]),
                         np.column_stack([pts[1:], z[1:]])], axis=1)
        lc = Line3DCollection(segs, colors=colors, linewidths=0.4, alpha=0.75)
        ax3.add_collection3d(lc)

        # Cut markers + eps-horizon tails (tails on showcase cuts only).
        cut_idx = np.nonzero(np.diff(ranks))[0]
        for ci in cut_idx:
            ax3.scatter(pts[ci, 0], pts[ci, 1], l * dz, color="k", s=24,
                        depthshade=False, zorder=5)
            if int(rank_of(frac[ci])) % 20 != 10:
                continue
            tail = (s > s[ci] - horizon_mm) & (s <= s[ci])
            if tail.sum() > 1:
                tp = pts[tail]
                tz = np.full(tp.shape[0], l * dz)
                tsegs = np.stack([np.column_stack([tp[:-1], tz[:-1]]),
                                  np.column_stack([tp[1:], tz[1:]])], axis=1)
                ax3.add_collection3d(Line3DCollection(
                    tsegs, colors="crimson", linewidths=1.6, alpha=0.85))

    ax3.set_xlim(pts[:, 0].min(), pts[:, 0].max())
    ax3.set_ylim(pts[:, 1].min(), pts[:, 1].max())
    ax3.set_zlim(0, (args.layers - 1) * dz)
    ax3.set_box_aspect((1.0, np.ptp(pts[:, 1]) / np.ptp(pts[:, 0]), 1.15))
    ax3.view_init(elev=22, azim=-58)
    ax3.set_axis_off()
    ax3.set_title(
        f"{args.layers}-layer build, one time axis: color = owning rank "
        f"({args.ranks} ranks)\nblack dots = partition cuts (mid-layer), "
        "red tails = $\\varepsilon$-horizon correction sources",
        fontsize=11)

    # Colorbar as a time/rank legend.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, args.ranks - 1))
    cb = fig.colorbar(sm, ax=ax3, shrink=0.55, pad=0.0, aspect=30)
    cb.set_label("rank (= position along build time)", fontsize=9)

    # ---- Panel B: serial strip vs folded parallel block ----
    axg = fig.add_subplot(gs[1])
    chunk_edges = edges * BASELINE_S
    for r in range(args.ranks):
        axg.add_patch(plt.Rectangle((chunk_edges[r], 3.2),
                                    chunk_edges[r + 1] - chunk_edges[r], 1.4,
                                    color=cmap(r / (args.ranks - 1)), lw=0))
        axg.add_patch(plt.Rectangle((0.0, 0.0), PAR64_S, 2.4 / args.ranks * args.ranks,
                                    color="none"))
    # Parallel block: 64 thin rows, all starting at t=0.
    row_h = 2.4 / args.ranks
    for r in range(args.ranks):
        axg.add_patch(plt.Rectangle((0.0, r * row_h),
                                    PAR64_S, row_h,
                                    color=cmap(r / (args.ranks - 1)), lw=0))
    axg.annotate("", xy=(PAR64_S * 6, 1.2), xytext=(BASELINE_S * 0.45, 3.9),
                 arrowprops=dict(arrowstyle="->", color="k", lw=1.2,
                                 connectionstyle="arc3,rad=0.25"))
    axg.text(BASELINE_S * 0.47, 4.9, f"1 rank: {BASELINE_S:,.0f} s",
             fontsize=10, ha="center")
    axg.text(PAR64_S * 7, 0.9,
             f"{args.ranks} ranks: {PAR64_S:.0f} s  (55.5$\\times$, measured)",
             fontsize=10)
    axg.set_xlim(-100, BASELINE_S * 1.02)
    axg.set_ylim(-0.4, 5.6)
    axg.set_yticks([])
    axg.set_xlabel("wall-clock time [s]")
    axg.spines[["left", "top", "right"]].set_visible(False)
    axg.set_title("the same colored chunks, folded in time", fontsize=10)

    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"[ok] saved: {args.out}")


if __name__ == "__main__":
    main()
