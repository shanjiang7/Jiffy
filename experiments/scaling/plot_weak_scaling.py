#!/usr/bin/env python3
"""
Weak-scaling figure: efficiency vs ranks (1,8,16,32,64), one
panel per path, two curves per panel — the calibrated 1e-4 target
(Lseg=0.9mm, eps=5K) and 1e-7 target (Lseg=1.3mm, eps=0.01K).

Data: outputs/weak_scaling_h18/<path>/<tol>/scaling_summary.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("outputs/weak_scaling_h18")
PATHS = [("bull", "Bull"), ("spiral_raster", "Spiral-Raster")]
TOLS = [("tol1e4", r"$e_{\rm tol}=10^{-4}$ ($\varepsilon=5$ K)",
         "#2e75b6", "o"),
        ("tol1e7", r"$e_{\rm tol}=10^{-7}$ ($\varepsilon=0.01$ K)",
         "#c1272d", "s")]


def load(path: str, tol: str) -> dict[int, float]:
    out = {}
    with open(ROOT / path / tol / "scaling_summary.csv") as f:
        for row in csv.DictReader(f):
            out[int(row["ranks"])] = float(row["efficiency"])
    return out


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), dpi=250)
    for ax, (key, title) in zip(axes, PATHS):
        # Categorical x positions over the union of measured rank counts, so
        # partially complete sweeps still plot correctly.
        union = sorted({r for tol, *_ in TOLS for r in load(key, tol)})
        pos = {r: i for i, r in enumerate(union)}
        for tol, label, color, marker in TOLS:
            eff = load(key, tol)
            xs = sorted(eff)
            ys = [eff[r] for r in xs]
            ax.plot([pos[r] for r in xs], ys, color=color, lw=2.4,
                    marker=marker, ms=8, label=label, zorder=3)
            for r, e in zip(xs, ys):
                if r >= 8:
                    ax.annotate(f"{e:.2f}", (pos[r], e),
                                textcoords="offset points", xytext=(0, 10),
                                fontsize=17, color=color, ha="center")
        ax.axhline(1.0, ls="--", color="0.6", lw=1.6, zorder=2)
        ax.set_xticks(range(len(union)))
        ax.set_xticklabels([str(r) for r in union])
        ax.set_ylim(0.5, 1.1)
        ax.set_title(f"Scan Path: {title}", fontsize=20, pad=10)
        ax.set_xlabel("Ranks", fontsize=18)
        ax.set_ylabel("Weak-scaling efficiency", fontsize=18)
        ax.tick_params(labelsize=15)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=15, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(ROOT / "weak_scaling.png")
    print(f"[ok] {ROOT}/weak_scaling.png")


if __name__ == "__main__":
    main()
