#!/usr/bin/env python3
"""
Parametric-study figure: spiral-raster strong scaling 1-8 ranks under the
two calibrated configurations, CG tolerance matched to the error target:
  left : 1e-4 target (eps=5K,    CG 1e-5)  outputs/.../continuous_hybrid_eps5
  right: 1e-7 target (eps=0.01K, CG 1e-10) outputs/.../continuous_hybrid
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNI = "#f0a202"
OPT = "#2e75b6"
ROOT = Path("outputs/cr_strong_scaling_h18")
PANELS = [("continuous_hybrid_eps5", "Spiral-Raster, Accuracy: 1e-4"),
          ("continuous_hybrid", "Spiral-Raster, Accuracy: 1e-7")]


def load(path: str) -> dict:
    out = {"exact_dp": {}, "uniform": {}}
    with open(ROOT / path / "scaling_summary.csv") as f:
        for row in csv.DictReader(f):
            out[row["planner"]][int(row["ranks"])] = (
                float(row["seconds"]), float(row["speedup"]))
    return out


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.9), dpi=250)
    for ax, (key, title) in zip(axes, PANELS):
        data = load(key)
        for mode, color, name, dy in (("uniform", UNI, "Uniform", -19),
                                      ("exact_dp", OPT, "Optimized", 7)):
            xs = sorted(data[mode])
            ys = [data[mode][r][1] for r in xs]
            ax.plot(xs, ys, color=color, lw=2.4, marker="o", ms=7, zorder=3,
                    label=name)
            for r, s in zip(xs, ys):
                if r >= 2:
                    ax.annotate(f"{s:.2f}x", (r, s),
                                textcoords="offset points", xytext=(2, dy),
                                fontsize=14, color=color, ha="center")
        ax.plot([1, 8], [1, 8], ls="--", color="0.6", lw=1.6, label="Ideal",
                zorder=2)
        base = data["exact_dp"][1][0]
        ax.text(0.97, 0.06, f"Baseline duration: {base:.1f} s",
                transform=ax.transAxes, ha="right", fontsize=16)
        ax.set_title(title, fontsize=19, pad=10)
        ax.set_xlabel("Ranks", fontsize=17)
        ax.set_ylabel("Speedup", fontsize=17)
        ax.set_xticks(range(1, 9))
        ax.set_ylim(0, 8.6)
        ax.tick_params(labelsize=14)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=15, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(ROOT / "parametric_study.png")
    print(f"[ok] {ROOT}/parametric_study.png")


if __name__ == "__main__":
    main()
