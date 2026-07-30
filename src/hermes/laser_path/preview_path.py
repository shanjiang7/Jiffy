#!/usr/bin/env python3
"""
Preview the laser path defined in a path INI.

Uses the same loader as the solver (`build_path_sections_nd_from_ini`), so the
preview is exactly the path a run would execute: source-on sections are drawn
solid, source-off connectors dashed.

Usage:
    python src/hermes/laser_path/preview_path.py \
        --config configs/examples/continuous_hybrid.ini
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hermes.laser_path.path_loader import build_path_sections_nd_from_ini


def densify_polyline(points: np.ndarray, samples_per_segment: int) -> np.ndarray:
    """Linearly interpolate `samples_per_segment` points between waypoints."""
    if points.shape[0] <= 1 or samples_per_segment <= 1:
        return points.copy()
    segs = []
    for i in range(points.shape[0] - 1):
        a, b = points[i], points[i + 1]
        t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
        segs.append(a[None, :] * (1.0 - t[:, None]) + b[None, :] * t[:, None])
    segs.append(points[-1][None, :])
    return np.vstack(segs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", required=True, help="Path INI (any configs/*.ini with a [path.*] section)")
    parser.add_argument("--samples-per-seg", type=int, default=80,
                        help="Interpolation samples per segment for the visual trace (default: 80).")
    parser.add_argument("--label-every", type=int, default=200,
                        help="Label every Nth waypoint with its index (0 disables; default: 200).")
    parser.add_argument("--out", default=None,
                        help="Output PNG (default: preview_<config-stem>.png in the current directory).")
    args = parser.parse_args()

    # len_scale=1.0: build in meters so the axes read physically.
    sections = build_path_sections_nd_from_ini(args.config, len_scale=1.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    n_points = 0
    for arr_m, source_on in sections:
        dense = densify_polyline(arr_m, args.samples_per_seg)
        style = dict(lw=1.0) if source_on else dict(lw=0.8, ls="--", alpha=0.5)
        ax.plot(dense[:, 0], dense[:, 1], **style,
                color="tab:red" if source_on else "tab:gray")
        if args.label_every > 0:
            for idx in range(0, arr_m.shape[0], args.label_every):
                ax.text(arr_m[idx, 0], arr_m[idx, 1], str(n_points + idx),
                        fontsize=7, color="tab:blue")
        n_points += arr_m.shape[0]

    n_on = sum(1 for _, on in sections if on)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{Path(args.config).name}: {len(sections)} section(s) "
                 f"({n_on} source-on), {n_points} waypoints")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = Path(args.out) if args.out else Path.cwd() / f"preview_{Path(args.config).stem}.png"
    fig.savefig(out_path, dpi=200)
    print(f"[ok] saved: {out_path}")


if __name__ == "__main__":
    main()
