#!/usr/bin/env python3
"""
Export the base / correction / full temperature-field decomposition around a
partition cut as VTK time series for ParaView (candidate spotlight figure:
superposition made visible).

Requires a run of main.py executed with HERMES_SAVE_BASE_SNAPSHOTS=1, which
saves the pre-correction states to <run>/snapshots_base alongside the normal
<run>/snapshots_par (base + corrections). Per matching snapshot:

    base_K       absolute temperature of the rank-local source-on solve
    full_K       absolute temperature after correction superposition
    correction_K full - base (the superposed source-off corrections), Kelvin

Frames are cropped to the stored 1 mm x 1 mm ROI and the top --z-mm of the
column, positioned at their absolute x/y location (the domain moves along the
scan path), with the powder surface at z = 0. One .pvd series file per field;
open the three .pvd files in ParaView.

Example (after the sbatch run):
    python experiments/visualization/export_cut_decomposition.py \
        --run-dir outputs/fig_decomposition/bull_8r \
        --sim-config configs/examples/sim_ex1.ini --dt-us 10 \
        --cut-rank 1 --before 600 --after 1800
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hermes.runtime.setup import load_sim_setup


def _load_meta(meta_dir: Path) -> dict[int, dict]:
    """Within-layer step -> meta record (layer 0 only)."""
    by_step: dict[int, dict] = {}
    for f in sorted(meta_dir.glob("rank_*.jsonl")):
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            if int(rec["layer"]) == 0:
                by_step[int(rec["index"])] = rec
    return by_step


def _write_vtk(path: Path, arr: np.ndarray, name: str, origin_m, h_m: float) -> None:
    """Legacy STRUCTURED_POINTS, binary scalars (big-endian float32)."""
    ncx, ncy, ncz = arr.shape
    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(f"{name}\nBINARY\nDATASET STRUCTURED_POINTS\n".encode())
        f.write(f"DIMENSIONS {ncx} {ncy} {ncz}\n".encode())
        f.write(f"ORIGIN {origin_m[0]:.9e} {origin_m[1]:.9e} {origin_m[2]:.9e}\n".encode())
        f.write(f"SPACING {h_m:.9e} {h_m:.9e} {h_m:.9e}\n".encode())
        f.write(f"POINT_DATA {arr.size}\n".encode())
        f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n".encode())
        # VTK structured points expect x fastest; arr is (x, y, z) C-order,
        # so transpose to (z, y, x) before flattening.
        f.write(arr.astype(">f4").transpose(2, 1, 0).tobytes())


def _write_pvd(path: Path, entries: list[tuple[float, str]]) -> None:
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1" byte_order="BigEndian">',
             "  <Collection>"]
    for t, fname in entries:
        lines.append(f'    <DataSet timestep="{t:.9e}" group="" part="0" file="{fname}"/>')
    lines += ["  </Collection>", "</VTKFile>"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="Run dir with snapshots_par and snapshots_base.")
    ap.add_argument("--sim-config", default="configs/examples/sim_ex1.ini")
    ap.add_argument("--dt-us", type=float, default=10.0)
    ap.add_argument("--cut-rank", type=int, default=1,
                    help="Export the window around the cut at the START of this rank's block (default: 1 = first cut).")
    ap.add_argument("--before", type=int, default=600, help="Window extent before the cut, in steps.")
    ap.add_argument("--after", type=int, default=1800, help="Window extent after the cut, in steps.")
    ap.add_argument("--z-mm", type=float, default=0.5, help="Retained depth below the top surface (mm).")
    ap.add_argument("--t0-k", type=float, default=None,
                    help="Reference temperature offset (K); default: phys.Ts, matching surface_export_segment_correction.")
    ap.add_argument("--out-dir", default=None, help="Default: <run-dir>/vtk_decomposition")
    args = ap.parse_args()

    run = Path(args.run_dir)
    plan = json.loads((run / "planning_summary.json").read_text())
    steps_per_ss = int(plan["steps_per_ss"])
    part = plan["partition"]["rank_assignments"]
    cut_ss = min(int(s) for s in part[str(args.cut_rank)])
    cut_step = cut_ss * steps_per_ss
    lo, hi = cut_step - args.before, cut_step + args.after
    print(f"[cut] rank {args.cut_rank} block starts at SS {cut_ss} -> step {cut_step}; window [{lo}, {hi}]")

    setup = load_sim_setup(args.sim_config, dt_us=args.dt_us)
    phys = setup.phys
    len_scale = float(phys.len_scale)
    deltaT = float(phys.deltaT)
    h_m = float(setup.rc.level3.h_tuple[0])
    t0_k = float(args.t0_k) if args.t0_k is not None else float(phys.Ts)
    dt_s = float(args.dt_us) * 1e-6
    print(f"[scales] len_scale={len_scale:.4e} m  deltaT={deltaT:.2f} K  Ts={t0_k:.1f} K  h={h_m*1e6:.1f} um")

    full_meta = _load_meta(run / "snapshots_par_meta")
    base_meta = _load_meta(run / "snapshots_base_meta")
    steps = sorted(s for s in full_meta if lo <= s <= hi and s in base_meta)
    if not steps:
        raise SystemExit("no matching snapshots in the window — check stride/window")
    print(f"[frames] {len(steps)} matched snapshots in window")

    out = Path(args.out_dir) if args.out_dir else run / "vtk_decomposition"
    series: dict[str, list[tuple[float, str]]] = {"base": [], "correction": [], "full": []}
    for name in series:
        (out / name).mkdir(parents=True, exist_ok=True)

    ncz_keep = None
    for s in steps:
        rec = full_meta[s]
        full = np.load(run / "snapshots_par" / rec["file"])
        base = np.load(run / "snapshots_base" / base_meta[s]["file"])
        if ncz_keep is None:
            ncz_keep = min(full.shape[2], max(1, round(args.z_mm * 1e-3 / h_m)))
        full = full[:, :, -ncz_keep:]
        base = base[:, :, -ncz_keep:]
        corr = full - base

        cx = float(rec["center_x_nd"]) * len_scale
        cy = float(rec["center_y_nd"]) * len_scale
        ncx, ncy, _ = full.shape
        origin = (
            cx - 0.5 * (ncx - 1) * h_m,
            cy - 0.5 * (ncy - 1) * h_m,
            -(ncz_keep - 1) * h_m,
        )
        t = s * dt_s
        fields = {
            "base": t0_k + base * deltaT,
            "full": t0_k + full * deltaT,
            "correction": corr * deltaT,
        }
        for name, arr in fields.items():
            fname = f"{name}_{s:09d}.vtk"
            _write_vtk(out / name / fname, arr, f"{name}_K", origin, h_m)
            series[name].append((t, f"{name}/{fname}"))

    for name, entries in series.items():
        _write_pvd(out / f"{name}_series.pvd", entries)
    print(f"[ok] wrote {len(steps)} frames x 3 fields under {out}")
    print("[hint] open base_series.pvd / correction_series.pvd / full_series.pvd in ParaView")


if __name__ == "__main__":
    main()
