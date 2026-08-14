#!/usr/bin/env python3
"""
Export one wall-clock instant of a multi-layer, multi-rank run for the
parallel-in-time spotlight figure: every rank has completed the same
fraction f of its own contiguous block, so the scene shows N scan fronts
advancing simultaneously across the layer stack.

Outputs (under <run>/vtk_instant/):
  domains/rank_XX.vtk   temperature volume of each rank's moving domain at
                        its f-progress snapshot (full_K, Kelvin), placed at
                        its absolute x/y and its layer's exaggerated z
  paths.vtk             all layers' scan paths as polylines with cell data:
                        rank (0..N-1), layer (0..L-1), done (1 = scanned at
                        this instant, 0 = not yet)
  instant_summary.json  per-rank: chosen step, layer, position, block range

ParaView styling hints: volume-render the domains with an opacity ramp that
makes ambient transparent (< ~800 K), add an Outline representation for the
moving-domain wireframes, and Threshold the paths on `done` to style traced
vs untraced track separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hermes.runtime.setup import load_sim_setup
from hermes.laser_path.path_loader import build_path_sections_nd_from_ini


def _write_vtk_volume(path: Path, arr: np.ndarray, name: str, origin_m, h_m: float) -> None:
    ncx, ncy, ncz = arr.shape
    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(f"{name}\nBINARY\nDATASET STRUCTURED_POINTS\n".encode())
        f.write(f"DIMENSIONS {ncx} {ncy} {ncz}\n".encode())
        f.write(f"ORIGIN {origin_m[0]:.9e} {origin_m[1]:.9e} {origin_m[2]:.9e}\n".encode())
        f.write(f"SPACING {h_m:.9e} {h_m:.9e} {h_m:.9e}\n".encode())
        f.write(f"POINT_DATA {arr.size}\n".encode())
        f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n".encode())
        f.write(arr.astype(">f4").transpose(2, 1, 0).tobytes())


def _write_vtk_polylines(path: Path, pts: np.ndarray, lines: list[list[int]],
                         cell_data: dict[str, list[int]]) -> None:
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\npaths\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(pts)} float\n")
        for p in pts:
            f.write(f"{p[0]:.7e} {p[1]:.7e} {p[2]:.7e}\n")
        total = sum(len(l) + 1 for l in lines)
        f.write(f"LINES {len(lines)} {total}\n")
        for l in lines:
            f.write(str(len(l)) + " " + " ".join(map(str, l)) + "\n")
        f.write(f"CELL_DATA {len(lines)}\n")
        for name, vals in cell_data.items():
            f.write(f"SCALARS {name} int 1\nLOOKUP_TABLE default\n")
            f.write("\n".join(str(v) for v in vals) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--path-config", required=True, help="Path config of the run (for the polylines).")
    ap.add_argument("--sim-config", default="configs/examples/sim_ex1.ini")
    ap.add_argument("--dt-us", type=float, default=10.0)
    ap.add_argument("--fraction", type=float, default=0.55, help="Per-rank block progress fraction.")
    ap.add_argument("--layer-spacing-mm", type=float, default=1.0,
                    help="Exaggerated z spacing between layers (real thickness is ~0.1 mm).")
    ap.add_argument("--step-m", type=float, default=1e-5, help="Laser advance per step (m).")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run = Path(args.run_dir)
    plan = json.loads((run / "planning_summary.json").read_text())
    steps_per_ss = int(plan["steps_per_ss"])
    ss_per_layer = int(plan["ss_per_layer"])
    part = {int(r): sorted(int(s) for s in ss) for r, ss in plan["partition"]["rank_assignments"].items()}
    nranks = len(part)

    setup = load_sim_setup(args.sim_config, dt_us=args.dt_us)
    phys = setup.phys
    len_scale, deltaT, Ts = float(phys.len_scale), float(phys.deltaT), float(phys.Ts)
    h_m = float(setup.rc.level3.h_tuple[0])
    zsp = args.layer_spacing_mm * 1e-3

    # --- per-rank snapshot at fraction f of its block -----------------------
    # Multi-layer snapshot metadata (layer/index/center) is unreliable for
    # components that straddle layer boundaries, so everything is derived
    # from the trustworthy fields (component_id, component_step) plus the
    # plan's SS ranges and the path geometry itself.
    meta_by_rank: dict[int, list[dict]] = {r: [] for r in part}
    for f in sorted((run / "snapshots_par_meta").glob("rank_*.jsonl")):
        r = int(f.stem.split("_")[1])
        for line in f.read_text().splitlines():
            meta_by_rank.setdefault(r, []).append(json.loads(line))

    # geometry: the (single-layer) path, densely resampled by arc length
    sections = build_path_sections_nd_from_ini(args.path_config, len_scale=1.0)
    pts_nd = np.vstack([a for a, on in sections if on])
    pts_m = pts_nd * len_scale
    seg = np.linalg.norm(np.diff(pts_m, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])

    def pos_at_within_layer_step(w: int) -> tuple[float, float]:
        a = min(w * args.step_m, arc[-1])
        return float(np.interp(a, arc, pts_m[:, 0])), float(np.interp(a, arc, pts_m[:, 1]))

    # per-rank components = the rank's contiguous block split at layer
    # boundaries, in path order; meta component_ids sort in the same order
    def layer_pieces(block: list[int]) -> list[tuple[int, int]]:
        pieces = []
        cur = [block[0]]
        for s in block[1:]:
            if s // ss_per_layer == cur[-1] // ss_per_layer:
                cur.append(s)
            else:
                pieces.append((cur[0], cur[-1]))
                cur = [s]
        pieces.append((cur[0], cur[-1]))
        return pieces

    out = Path(args.out_dir) if args.out_dir else run / "vtk_instant"
    (out / "domains").mkdir(parents=True, exist_ok=True)
    summary = []
    done_ss: set[int] = set()
    for r, block in sorted(part.items()):
        pieces = layer_pieces(block)
        comp_ids = sorted({int(rec["component_id"]) for rec in meta_by_rank.get(r, [])})
        if len(comp_ids) != len(pieces):
            print(f"[warn] rank {r}: {len(comp_ids)} snapshot components vs "
                  f"{len(pieces)} layer pieces — matching by order anyway")
        start_ss_by_comp = {cid: pieces[i][0] for i, cid in enumerate(comp_ids) if i < len(pieces)}

        target_global = int((block[0] + args.fraction * len(block)) * steps_per_ss)
        best, best_d, best_g = None, None, None
        for rec in meta_by_rank.get(r, []):
            cid = int(rec["component_id"])
            if cid not in start_ss_by_comp:
                continue
            g = start_ss_by_comp[cid] * steps_per_ss + int(rec["component_step"])
            d = abs(g - target_global)
            if best_d is None or d < best_d:
                best, best_d, best_g = rec, d, g
        if best is None:
            raise SystemExit(f"rank {r}: no snapshots found")
        ss_of_snap = best_g // steps_per_ss
        layer = ss_of_snap // ss_per_layer
        w = best_g - layer * ss_per_layer * steps_per_ss
        cx, cy = pos_at_within_layer_step(w)
        arr = np.load(run / "snapshots_par" / best["file"])
        lz = layer * zsp
        ncx, ncy, ncz = arr.shape
        origin = (cx - 0.5 * (ncx - 1) * h_m, cy - 0.5 * (ncy - 1) * h_m, lz - (ncz - 1) * h_m)
        _write_vtk_volume(out / "domains" / f"rank_{r:02d}.vtk", Ts + arr * deltaT, "full_K", origin, h_m)
        done_ss.update(range(block[0], min(block[-1] + 1, ss_of_snap + 1)))
        summary.append({"rank": r, "block": [block[0], block[-1]], "global_step": best_g,
                        "layer": int(layer), "x_mm": cx * 1e3, "y_mm": cy * 1e3})
        print(f"rank {r:2d}: SS [{block[0]:4d},{block[-1]:4d}]  step {best_g:7d}  "
              f"layer {layer}  at ({cx*1e3:7.3f}, {cy*1e3:7.3f}) mm")

    # --- path polylines, one copy per layer, chopped per SS ----------------
    ss_len_m = steps_per_ss * args.step_m
    ss_to_rank = {s: r for r, block in part.items() for s in block}

    all_pts: list[list[float]] = []
    lines: list[list[int]] = []
    cd: dict[str, list[int]] = {"rank": [], "layer": [], "done": []}
    num_layers = int(plan["num_layers"])
    # resample the path densely so per-SS chopping is smooth
    dense_arc = np.arange(0.0, arc[-1], args.step_m * 10)
    dense_x = np.interp(dense_arc, arc, pts_m[:, 0])
    dense_y = np.interp(dense_arc, arc, pts_m[:, 1])
    dense_ss = np.minimum((dense_arc / ss_len_m).astype(int), ss_per_layer - 1)
    for layer in range(num_layers):
        z = layer * zsp
        base_idx = len(all_pts)
        all_pts.extend([[x, y, z] for x, y in zip(dense_x, dense_y)])
        start = 0
        for i in range(1, len(dense_ss) + 1):
            if i == len(dense_ss) or dense_ss[i] != dense_ss[start]:
                s_local = int(dense_ss[start])
                s_global = layer * ss_per_layer + s_local
                lines.append(list(range(base_idx + start, base_idx + min(i + 1, len(dense_ss)))))
                cd["rank"].append(int(ss_to_rank.get(s_global, -1)))
                cd["layer"].append(layer)
                cd["done"].append(1 if s_global in done_ss else 0)
                start = i
    _write_vtk_polylines(out / "paths.vtk", np.array(all_pts), lines, cd)

    (out / "instant_summary.json").write_text(json.dumps(
        {"fraction": args.fraction, "layer_spacing_mm": args.layer_spacing_mm,
         "num_ranks": nranks, "num_layers": num_layers, "ranks": summary}, indent=2))
    print(f"[ok] {nranks} domain volumes + paths.vtk under {out}")
    print("[hint] volume-render domains with ambient transparent; Outline for the boxes; "
          "Threshold paths.vtk on 'done' and color by 'rank'")


if __name__ == "__main__":
    main()
