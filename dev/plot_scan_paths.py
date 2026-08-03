#!/usr/bin/env python3
"""
Regenerate the scan-path overview figure (fig:scan_paths): Bull, Texas,
spiral-raster (one unit of the continuous path), and Hilbert, drawn from
the same configs the solver runs.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hermes.laser_path.path_loader import build_path_sections_nd_from_ini

PANELS = [
    ("configs/examples/fast_heat.ini", "Bull"),
    ("configs/examples/texas.ini", "Texas"),
    ("configs/dev/continuous_hybrid_unit.ini", "Spiral-Raster"),
    ("configs/examples/hilbert.ini", "Hilbert"),
]


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.8), dpi=250)
    for ax, (cfg, title) in zip(axes.flat, PANELS):
        for arr, source_on in build_path_sections_nd_from_ini(cfg,
                                                              len_scale=1.0):
            a = np.asarray(arr) * 1e3
            style = dict(lw=0.6, color="tab:blue") if source_on else \
                dict(lw=0.6, color="tab:gray", ls="--", alpha=0.6)
            ax.plot(a[:, 0], a[:, 1], **style)
        ax.set_title(title, fontsize=24, pad=12)
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig("outputs/scan_path.png")
    print("[ok] outputs/scan_path.png")


if __name__ == "__main__":
    main()
