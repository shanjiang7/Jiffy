#!/usr/bin/env python3
"""
Camera-ready strong-scaling figures, matching the submitted figure style
(2x2 per-path panels, Uniform orange vs Optimized blue vs dashed Ideal,
per-point speedup labels, baseline duration note).

  fig 1: strong_scaling_8ranks.png   1-8 ranks, h=18um, four paths
         (outputs/cr_strong_scaling_h18/<path>/scaling_summary.csv)
  fig 2: strong_scaling_ml15.png     15-layer builds, 1,8,16,32,64 ranks
         (outputs/cr_strong_scaling_ml15/<path>/scaling_summary.csv)

Speedup is relative to each strategy's own 1-rank baseline (submitted
convention); the printed baseline duration is the optimized-run baseline.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNI = "#f0a202"
OPT = "#2e75b6"
PATHS_8R = [("bull", "Bull"), ("texas", "Texas"),
            ("continuous_hybrid", "Spiral-Raster"), ("hilbert", "Hilbert")]
PATHS_ML = [("bull", "Bull"), ("continuous_hybrid", "Spiral-Raster")]


def load(root: str, path: str) -> dict:
    out = {"exact_dp": {}, "uniform": {}}
    with open(Path(root) / path / "scaling_summary.csv") as f:
        for row in csv.DictReader(f):
            out[row["planner"]][int(row["ranks"])] = (
                float(row["seconds"]), float(row["speedup"]))
    return out


def panel(ax, data, title, label_ranks, ideal_max, spread=1.0):
    ranks = sorted(data["exact_dp"])
    for mode, color, name, dy in (("uniform", UNI, "Uniform", -0.52 * spread),
                                  ("exact_dp", OPT, "Optimized", 0.34 * spread)):
        xs = ranks
        ys = [data[mode][r][1] for r in xs]
        ax.plot(xs, ys, color=color, lw=2.4, marker="o", ms=7, zorder=3,
                label=name)
        for r, s in zip(xs, ys):
            if r in label_ranks:
                ax.annotate(f"{s:.2f}x", (r, s), textcoords="offset points",
                            xytext=(2, 18 * dy), fontsize=15, color=color,
                            ha="center", zorder=4)
    ax.plot([1, ideal_max], [1, ideal_max], ls="--", color="0.6", lw=1.6,
            label="Ideal", zorder=2)
    base = data["exact_dp"][1][0]
    ax.text(0.97, 0.06, f"Baseline duration: {base:.1f} s",
            transform=ax.transAxes, ha="right", fontsize=17)
    ax.set_title(f"Scan Path: {title}", fontsize=20, pad=10)
    ax.set_xlabel("Ranks", fontsize=18)
    ax.set_ylabel("Speedup", fontsize=18)
    ax.tick_params(labelsize=15)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=16, loc="upper left", framealpha=0.9)


def fig_8ranks() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 11.8), dpi=250)
    for ax, (key, title) in zip(axes.flat, PATHS_8R):
        data = load("outputs/cr_strong_scaling_h18", key)
        panel(ax, data, title, label_ranks=set(range(2, 9)), ideal_max=8)
        ax.set_xticks(range(1, 9))
        ax.set_ylim(0, 8.6)
    fig.tight_layout()
    fig.savefig("outputs/cr_strong_scaling_h18/strong_scaling_8ranks.png")
    print("[ok] outputs/cr_strong_scaling_h18/strong_scaling_8ranks.png")


def fig_ml15() -> None:
    # Linear axes: near-ideal curves on log-log collapse onto the diagonal
    # and hide the DP-vs-uniform gap at 32-64 ranks.
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.9), dpi=250)
    for ax, (key, title) in zip(axes.flat, PATHS_ML):
        data = load("outputs/cr_strong_scaling_ml15", key)
        panel(ax, data, f"{title} (15 layers)",
              label_ranks={16, 32, 64}, ideal_max=64, spread=2.2)
        ax.set_xticks([1, 8, 16, 32, 64])
        ax.set_xlim(-1, 67)
        ax.set_ylim(0, 68)
    fig.tight_layout()
    fig.savefig("outputs/cr_strong_scaling_ml15/strong_scaling_ml15.png")
    print("[ok] outputs/cr_strong_scaling_ml15/strong_scaling_ml15.png")


if __name__ == "__main__":
    fig_8ranks()
    fig_ml15()
