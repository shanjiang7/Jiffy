#!/usr/bin/env python3
"""
Regenerate the scan-path overview figure (fig:scan_paths): Bull, Texas,
spiral-raster (one unit of the continuous path), and Hilbert, drawn from
the same configs the solver runs. Each path is colored by scan order
(dark = start, bright = end), which also conveys the time axis along the
path.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from hermes.laser_path.path_loader import build_path_sections_nd_from_ini

# Bull/Texas use 4x-coarsened scan-line spacing and Hilbert order 5 (viz-only
# variants) so the path structure is visible at figure scale; the
# spiral-raster unit is shown at true geometry.
PANELS = [
    ("configs/experiments/bull_viz.ini", "Bull"),
    ("configs/experiments/texas_viz.ini", "Texas"),
    ("configs/experiments/continuous_hybrid_unit.ini", "Spiral-Raster"),
    ("configs/experiments/hilbert_viz.ini", "Hilbert"),
]
CMAP = plt.get_cmap("viridis")


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.8), dpi=250)
    for ax, (cfg, title) in zip(axes.flat, PANELS):
        pts = np.vstack([a for a, on in
                         build_path_sections_nd_from_ini(cfg, len_scale=1.0)
                         if on]) * 1e3
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        frac = np.linspace(0.0, 1.0, len(segs))
        lc = LineCollection(segs, colors=CMAP(frac), linewidths=1.0)
        ax.add_collection(lc)
        ax.set_xlim(pts[:, 0].min() - 1, pts[:, 0].max() + 1)
        ax.set_ylim(pts[:, 1].min() - 1, pts[:, 1].max() + 1)
        # 'datalim' keeps the axes boxes identical, so the four titles align.
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(title, fontsize=24, pad=12)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig("outputs/scan_path.png")
    print("[ok] outputs/scan_path.png")


if __name__ == "__main__":
    main()
