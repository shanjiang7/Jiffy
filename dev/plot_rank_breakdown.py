#!/usr/bin/env python3
"""
Per-rank runtime breakdown (stacked source-on solve + correction per rank)
for optimized vs uniform partitioning.

Default mode uses a compressed piecewise-linear y scale: the low-variation
band (default 50--170 s) is squeezed by SQUEEZE_F so bars remain continuous
(no white gap) while the variation near the top of the bars stays legible;
break marks on the y spine indicate the compression. --no-break gives a
plain linear axis (use when the variation is large, e.g. single layer).

Examples:
  # 15-layer Bull, 64 ranks (paper default)
  python dev/plot_rank_breakdown.py
  # single-layer, flat axis, gate-style layout
  python dev/plot_rank_breakdown.py --root outputs/cr_strong_scaling_h18/bull \
      --run parallel_64r --no-break --out .../rank_breakdown_64r_1layer.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="outputs/cr_strong_scaling_ml15/bull")
ap.add_argument("--run", default="parallel_64r")
ap.add_argument("--out",
                default="outputs/cr_strong_scaling_ml15/rank_breakdown_64r.png")
ap.add_argument("--squeeze", nargs=2, type=float, default=(50.0, 170.0),
                help="y band to compress (start, end)")
ap.add_argument("--ymax", type=float, default=250.0)
ap.add_argument("--no-break", action="store_true",
                help="plain linear y axis (large-variation cases)")
ap.add_argument("--dp-dir", default=None,
                help="override subdir for the optimized run (flat layouts)")
ap.add_argument("--uniform-dir", default=None,
                help="override subdir for the uniform run (flat layouts)")
ARGS = ap.parse_args()

BASE_C = "#9dc3e6"
CORR_C = "#c1272d"
SQ_LO, SQ_HI = ARGS.squeeze
SQ_F = 12.0  # compression factor for the squeezed band


def fwd(y):
    y = np.asarray(y, dtype=float)
    return np.where(y < SQ_LO, y,
                    np.where(y <= SQ_HI, SQ_LO + (y - SQ_LO) / SQ_F,
                             SQ_LO + (SQ_HI - SQ_LO) / SQ_F + (y - SQ_HI)))


def inv(z):
    z = np.asarray(z, dtype=float)
    z1 = SQ_LO + (SQ_HI - SQ_LO) / SQ_F
    return np.where(z < SQ_LO, z,
                    np.where(z <= z1, SQ_LO + (z - SQ_LO) * SQ_F,
                             SQ_HI + (z - z1)))


def load(planner: str):
    override = ARGS.dp_dir if planner == "exact_dp" else ARGS.uniform_dir
    sub = override if override else f"{planner}/{ARGS.run}"
    d = json.load(open(f"{ARGS.root}/{sub}/timing_summary.json"))
    rows = d["rank_timing_breakdown"]
    rows = rows if isinstance(rows, list) else list(rows.values())
    base = [r["base_solve_seconds"] for r in rows]
    corr = [r["tracer_solve_seconds"] + r["local_superpose_seconds"]
            for r in rows]
    return base, corr


def draw(ax, planner: str, title: str, squeeze: bool):
    base, corr = load(planner)
    ranks = range(len(base))
    tot = [b + c for b, c in zip(base, corr)]
    mx, mean = max(tot), sum(tot) / len(tot)
    ax.bar(ranks, base, width=0.82, color=BASE_C, label="Source-on solve")
    ax.bar(ranks, corr, width=0.82, bottom=base, color=CORR_C,
           label="Correction")
    ax.axhline(mx, ls="--", color="k", lw=1.4)
    ax.text(len(base) - 0.4, mx + 0.02 * mx,
            f"max {mx:.0f} s   (max/mean = {mx/mean:.2f})",
            ha="right", fontsize=15)
    ax.set_xlim(-1, len(base))
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=14)
    ax.set_title(f"{title} partitioning", fontsize=19, loc="left", pad=8)
    ax.set_ylabel("Busy time [s]", fontsize=17)
    if not squeeze:
        ax.set_ylim(0, 1.18 * mx)
        return
    ax.set_yscale("function", functions=(fwd, inv))
    ax.set_ylim(0, ARGS.ymax)
    lo_ticks = [0, SQ_LO]
    hi_start = int(np.ceil(SQ_HI / 20.0)) * 20
    ax.set_yticks(lo_ticks + list(range(hi_start, int(ARGS.ymax) + 1, 20)))
    # Break marks on the y spine at the compressed band.
    d = 0.4
    kw = dict(marker=[(-1, -d), (1, d)], markersize=10, linestyle="none",
              color="k", mec="k", mew=1, clip_on=False,
              transform=ax.get_yaxis_transform())
    mid = (SQ_LO + SQ_HI) / 2.0
    ax.plot([0, 0], [SQ_LO + 0.25 * (SQ_HI - SQ_LO), mid + 0.25 * (SQ_HI - SQ_LO)], **kw)


def main() -> None:
    squeeze = not ARGS.no_break
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(12.6, 7.8), dpi=250,
                                 sharex=True)
    draw(a0, "exact_dp", "Optimized", squeeze)
    draw(a1, "uniform", "Uniform", squeeze)
    a0.legend(fontsize=15, loc="lower right", framealpha=0.95)
    a1.set_xlabel("Rank", fontsize=17)
    fig.tight_layout()
    fig.savefig(ARGS.out)
    print(f"[ok] {ARGS.out}")


if __name__ == "__main__":
    main()
