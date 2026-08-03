#!/usr/bin/env python3
"""
Per-rank runtime breakdown at 64 ranks for the 15-layer Bull build:
optimized (exact_dp) vs uniform partitioning, stacked source-on solve +
correction (tracer) time per rank. Communication waits are <1.5% of the
runtime and are omitted (stated in the caption).

The y axis is broken (50--150 s hidden) so the rank-to-rank variation near
the top of the bars is legible while the bars stay anchored at zero.

Data: outputs/cr_strong_scaling_ml15/bull/<planner>/parallel_64r/
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "outputs/cr_strong_scaling_ml15/bull"
BASE_C = "#9dc3e6"
CORR_C = "#c1272d"
Y_LO = (0, 50)
Y_HI = (150, 250)


def load(planner: str):
    d = json.load(open(f"{ROOT}/{planner}/parallel_64r/timing_summary.json"))
    rows = d["rank_timing_breakdown"]
    rows = rows if isinstance(rows, list) else list(rows.values())
    base = [r["base_solve_seconds"] for r in rows]
    corr = [r["tracer_solve_seconds"] + r["local_superpose_seconds"]
            for r in rows]
    return base, corr


def draw_panel(ax_hi, ax_lo, planner: str, title: str):
    base, corr = load(planner)
    ranks = range(len(base))
    tot = [b + c for b, c in zip(base, corr)]
    mx, mean = max(tot), sum(tot) / len(tot)
    for ax in (ax_hi, ax_lo):
        ax.bar(ranks, base, width=0.82, color=BASE_C, label="Source-on solve")
        ax.bar(ranks, corr, width=0.82, bottom=base, color=CORR_C,
               label="Correction")
        ax.set_xlim(-1, 64)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=14)
    ax_hi.axhline(mx, ls="--", color="k", lw=1.4)
    ax_hi.text(63.6, mx + 2.5, f"max {mx:.0f} s   (max/mean = {mx/mean:.2f})",
               ha="right", fontsize=15)
    ax_hi.set_ylim(*Y_HI)
    ax_lo.set_ylim(*Y_LO)
    ax_lo.set_yticks([0, 50])
    ax_hi.set_title(f"{title} partitioning", fontsize=19, loc="left", pad=8)
    # Broken-axis cosmetics.
    ax_hi.spines.bottom.set_visible(False)
    ax_lo.spines.top.set_visible(False)
    ax_hi.tick_params(bottom=False, labelbottom=False)
    d = 0.4
    kw = dict(marker=[(-1, -d), (1, d)], markersize=10, linestyle="none",
              color="k", mec="k", mew=1, clip_on=False)
    ax_hi.plot([0, 1], [0, 0], transform=ax_hi.transAxes, **kw)
    ax_lo.plot([0, 1], [1, 1], transform=ax_lo.transAxes, **kw)


def main() -> None:
    fig, axes = plt.subplots(
        5, 1, figsize=(12.6, 9.0), dpi=250, sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1, 0.55, 3, 1], hspace=0.08))
    axes[2].set_visible(False)  # spacer between the two broken-axis panels
    draw_panel(axes[0], axes[1], "exact_dp", "Optimized")
    draw_panel(axes[3], axes[4], "uniform", "Uniform")
    axes[0].legend(fontsize=15, loc="lower right", framealpha=0.95)
    axes[4].set_xlabel("Rank", fontsize=17)
    fig.supylabel("Busy time [s]", fontsize=17, x=0.02)
    out = "outputs/cr_strong_scaling_ml15/rank_breakdown_64r.png"
    fig.savefig(out)
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
