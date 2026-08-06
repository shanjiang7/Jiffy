#!/usr/bin/env python3
"""
Regenerate the scan-path overview figure (fig:scan_paths): Bull, Texas,
spiral-raster (one unit of the continuous path), and Hilbert, drawn from
the same configs the solver runs. Each path is colored by scan order
(dark = start, bright = end), which also conveys the time axis along the
path.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from hermes.laser_path.path_loader import build_path_sections_nd_from_ini

# Visualization-only overrides on the production configs, applied via
# generated [include] stubs: Bull/Texas use 2.5x-coarsened scan-line spacing
# (500um) and Hilbert order 5 (instead of 7) so the path structure is visible
# at figure scale; the spiral-raster panel shows one unit (repeats = 1) at
# true geometry. None of these variants is used for any measurement.
REPO = Path(__file__).resolve().parents[1]
PANELS = [
    ("bull.ini", "Bull",
     f"[path.picture]\nimage = {REPO}/configs/images/longhorn.jpg\n"
     "column_res_x = 500um\ncolumn_res_y = 500um\n"),
    ("texas.ini", "Texas",
     f"[path.picture]\nimage = {REPO}/configs/images/texas.jpg\n"
     "column_res_x = 500um\ncolumn_res_y = 500um\n"),
    ("spiral_raster.ini", "Spiral-Raster", "[path.spiral_raster]\nrepeats = 1\n"),
    ("hilbert.ini", "Hilbert", "[path.hilbert]\norder = 5\n"),
]
CMAP = plt.get_cmap("viridis")


def viz_config(tmpdir: str, base_name: str, overrides: str) -> str:
    stub = Path(tmpdir) / f"viz_{base_name}"
    stub.write_text(
        f"[include]\nbase = {REPO}/configs/examples/{base_name}\n\n{overrides}"
    )
    return str(stub)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.8), dpi=250)
    tmpdir = tempfile.mkdtemp(prefix="scan_path_viz_")
    for ax, (base, title, overrides) in zip(axes.flat, PANELS):
        cfg = viz_config(tmpdir, base, overrides)
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
