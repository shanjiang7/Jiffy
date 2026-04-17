#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hermes post-process to VTK (Level 3 / Outer Solver):
- Reads PATH/snapshots/step_XXXXXX.{npz|npy} (looking for u_lin)
- Reconstructs u_lin, x_lin, y_lin, z_lin (Fortran order)
- Builds phys from sim.ini via load_config -> phys_parameter
- Finds deepest melt plane (u>1), extracts Tl isosurface
- Writes:
    PATH/VTK/T_Level3/UT_level3_step_XXXXXX.vtk
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

# VTK / TVTK
from tvtk.api import tvtk, write_data
from vtk.util import numpy_support

# SciPy / scikit-image
from skimage.measure import marching_cubes
from scipy.interpolate import RegularGridInterpolator

# ---- import project helpers ----
# config loader & phys
from hermes.runtime.config import load_config
from hermes.physics.material import phys_parameter


# --------------------------
# Utils
# --------------------------

def parse_index_spec(spec: str, available: List[int]) -> List[int]:
    """Parse 'all' | 'N' | 'A:B' | 'a,b,c' to a sorted unique list filtered by `available`."""
    s = spec.strip().lower()
    if s == "all":
        return sorted(available)
    if re.match(r"^\d+:\d+$", s):
        a, b = map(int, s.split(":"))
        sel = [v for v in available if a <= v <= b]
        return sorted(sel)
    if re.match(r"^\d+(,\d+)*$", s):
        wanted = set(map(int, s.split(",")))
        sel = [v for v in available if v in wanted]
        return sorted(sel)
    if s.isdigit():
        val = int(s)
        return [val] if val in available else []
    raise ValueError(f"Unrecognized index spec: {spec}")

def detect_layered_scheme(snapshot_dir: Path) -> bool:
    """Return True if files look like 'layer_<L>_step_<STEP>_*.npy' or '.npz'."""
    if any(snapshot_dir.glob("layer_*_step_*.npz")):
        return True
    # We look for u_lin as representative for Level 3
    if any(snapshot_dir.glob("layer_*_step_*_u_lin.npy")):
        return True
    return False

def list_indexed(snapshot_dir: Path) -> Tuple[bool, List[Tuple[Optional[int], int]]]:
    """
    Return (layered, items) where items is a list of (layer, step) tuples.
    """
    items: set[Tuple[Optional[int], int]] = set()
    layered = detect_layered_scheme(snapshot_dir)

    if layered:
        # Accept both .npz and .npy sets
        for p in snapshot_dir.glob("layer_*_step_*.npz"):
            m = re.match(r"layer_(\d+)_step_(\d+)\.npz$", p.name)
            if m:
                L = int(m.group(1)); S = int(m.group(2))
                items.add((L, S))
        for p in snapshot_dir.glob("layer_*_step_*_u_lin.npy"):
            m = re.match(r"layer_(\d+)_step_(\d+)_u_lin\.npy$", p.name)
            if m:
                L = int(m.group(1)); S = int(m.group(2))
                items.add((L, S))
    else:
        # Legacy scheme
        for p in snapshot_dir.glob("step_*.npz"):
            m = re.match(r"step_(\d+)\.npz$", p.name)
            if m:
                S = int(m.group(1))
                items.add((None, S))
        for p in snapshot_dir.glob("step_*_u_lin.npy"):
            m = re.match(r"step_(\d+)_u_lin\.npy$", p.name)
            if m:
                S = int(m.group(1))
                items.add((None, S))

    return layered, sorted(items, key=lambda t: (t[0] or 0, t[1]))

def load_indexed(snapshot_dir: Path, layer: Optional[int], step: int):
    """
    Load one snapshot (Level 3 data). 
    Expects keys: 'u_lin', 'x_lin', 'y_lin', 'z_lin'.
    """
    data = {}
    keys_to_load = ("u_lin", "x_lin", "y_lin", "z_lin")
    
    if layer is not None:
        base_npz = snapshot_dir / f"layer_{layer}_step_{step:09d}.npz"
        if base_npz.exists():
            with np.load(base_npz) as Z:
                for k in keys_to_load:
                    if k in Z:
                        data[k] = Z[k]
                    else:
                        raise KeyError(f"Key {k} not found in {base_npz}. Make sure solver is saving level 3 data.")
            return data

        # fall back to individual .npy
        req = {
            "u_lin": snapshot_dir / f"layer_{layer}_step_{step:09d}_u_lin.npy",
            "x_lin": snapshot_dir / f"layer_{layer}_step_{step:09d}_x_lin.npy",
            "y_lin": snapshot_dir / f"layer_{layer}_step_{step:09d}_y_lin.npy",
            "z_lin": snapshot_dir / f"layer_{layer}_step_{step:09d}_z_lin.npy",
        }
        for k, p in req.items():
            if not p.exists():
                # Also try 6-digit for robustness
                alt = snapshot_dir / f"layer_{layer}_step_{step:06d}_{k}.npy"
                if not alt.exists():
                    raise FileNotFoundError(f"Missing required file: {p} (or {alt})")
                data[k] = np.load(alt)
            else:
                data[k] = np.load(p)
        return data

    # Legacy scheme (no layer)
    base_npz = snapshot_dir / f"step_{step:06d}.npz"
    if base_npz.exists():
        with np.load(base_npz) as Z:
            for k in keys_to_load:
                data[k] = Z[k]
        return data

    req = {
        "u_lin": snapshot_dir / f"step_{step:06d}_u_lin.npy",
        "x_lin": snapshot_dir / f"step_{step:06d}_x_lin.npy",
        "y_lin": snapshot_dir / f"step_{step:06d}_y_lin.npy",
        "z_lin": snapshot_dir / f"step_{step:06d}_z_lin.npy",
    }
    for k, p in req.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing required file for step {step}: {p}")
        data[k] = np.load(p)

    return data

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def shape_back_3d(u_flat: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    """Flattened Fortran-ordered -> (nx, ny, nz) array."""
    return np.reshape(u_flat, (nx, ny, nz), order="F")


def G2L_3D_arr(idxx: np.ndarray, nx: int, ny: int, nz: int):
    """Flat (Fortran) indices -> (i,j,k)."""
    k = idxx // (nx * ny)
    idx_z = idxx - k * nx * ny
    j = idx_z // nx
    i = idx_z - j * nx
    return i, j, k

def max_depth_calculator(ny_s: int, melt_pool_indices, z_s: np.ndarray) -> int:
    """Pick y-plane with max melt depth; tie -> middle index among maxima."""
    max_depths = []
    for y_plane in range(1, ny_s - 1):
        z_idx = melt_pool_indices[2][melt_pool_indices[1] == y_plane]
        if len(z_idx) > 0:
            max_depth = np.max(z_s[z_idx]) - np.min(z_s[z_idx])
        else:
            max_depth = -1
        max_depths.append(max_depth)
    max_depths = np.array(max_depths)
    if len(max_depths) == 0:
        return 0
    y_candidates = np.where(max_depths == np.max(max_depths))[0]
    return int(y_candidates[len(y_candidates) // 2])

# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Hermes post-process (Level 3): export Ts isosurface and temperature volume to VTK."
    )
    ap.add_argument("--output_path", required=True,
                    help="Path to output tag dir (contains 'snapshots/'). Example: /abs/.../outputs/demo_run")
    ap.add_argument("--steps", default="last",
                    help="Which steps: 'all', 'last', 'N', 'N:M', or comma list '10,20,30'. Default: last")
    ap.add_argument("--write-temp", action="store_true",
                    help="Also write temperature volume as VTK ImageData.")
    ap.add_argument("--config", default=None,
                    help="Optional path to sim.ini. If omitted, tries PATH/sim.ini; else falls back to repo configs/sim.ini.")
    ap.add_argument("--layers", default="all",
                help="Which layers to export: 'all', 'L', 'L1,L2', or 'A:B'. Default: all")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    base_path =  (project_root / args.output_path).resolve()
    snapshot_dir = base_path / "snapshots"
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"Snapshots folder not found: {snapshot_dir}")

    # ---- locate config ----
    cfg_path = None
    if args.config:
        c =  (project_root / args.config).resolve()
        if not c.is_file():
            raise FileNotFoundError(f"--config not found: {c}")
        cfg_path = c
    else:
        # try next to outputs tag dir
        local_cfg = base_path / "sim.ini"
        if local_cfg.is_file():
            cfg_path = local_cfg
        else:
            # fallback to repo default (../../../../configs/sim.ini relative to this file)
            repo_root = Path(__file__).resolve().parents[3]
            default_cfg = repo_root / "configs" / "sim.ini"
            if default_cfg.is_file():
                cfg_path = default_cfg
            else:
                raise FileNotFoundError("Could not find sim.ini (tried PATH/sim.ini and repo configs/sim.ini). "
                                        "Pass --config explicitly.")

    # ---- build phys from config (same as solver) ----
    rc = load_config(cfg_path)
    mat_override = rc.material.to_override_dict()
    t_spot_on = 2 * rc.laser.x_span_m / rc.laser.v
    phys = phys_parameter(rc.laser.Q, rc.laser.x_span_m, t_spot_on, mat_ch=mat_override)

    Ts = phys.Ts
    Tl = phys.Tl
    deltaT = phys.deltaT
    len_scale = phys.len_scale
    time_scale = phys.time_scale

    def temp_dim(u_nd):    # nondimensional -> K
        return u_nd * deltaT + Ts

    # ---- discover available (layer, step) pairs ----
    
    layered, pairs = list_indexed(snapshot_dir)
    if not pairs:
        raise RuntimeError(f"No snapshots found in {snapshot_dir}")
   
    avail_layers = sorted(set(L for (L, _) in pairs if L is not None)) if layered else []
    avail_steps  = sorted(set(S for (_, S) in pairs))

    sel_steps = args.steps.strip().lower()
    if sel_steps == "last":
        wanted_steps = [avail_steps[-1]]
    else:
        wanted_steps = parse_index_spec(sel_steps, avail_steps)
    if not wanted_steps:
        raise SystemExit(f"No steps matched --steps={args.steps!r} among {avail_steps}")

    if layered:
        wanted_layers = parse_index_spec(args.layers, avail_layers) if args.layers != "all" else avail_layers
        if not wanted_layers:
            raise SystemExit(f"No layers matched --layers={args.layers!r} among {avail_layers}")
    else:
        wanted_layers = [None]  # legacy

    print(f"[info] layered naming: {layered}")
    if layered:
        print(f"[info] Layers available: {avail_layers}; selected: {wanted_layers}")
    print(f"[info] Steps available: {avail_steps[:8]}{'...' if len(avail_steps)>8 else ''}; selected: {wanted_steps}")

    sel_pairs = [(L, S) for (L, S) in pairs
                 if (S in wanted_steps) and ((L in wanted_layers) if layered else True)]
    if not sel_pairs:
        raise SystemExit("No (layer, step) pairs matched your selection.")

    # ---- output dirs ----
    out_root = ensure_dir(base_path / "VTK_Level3") # Separate output folder
    out_T = ensure_dir(out_root / "T") 
    
    print(f"[info] sim.ini: {cfg_path}")
    print(f"[info] Ts={Ts} K, Tl={Tl} K, ΔT={deltaT} K, len_scale={len_scale}, time_scale={time_scale}")
    # print(f"[info] Steps: {steps}")
    print(f"[info] Writing under: {out_root}")
    

    for L, step in sel_pairs:
        if layered:
            print(f"\n=== layer {L} — step {step:09d} ===")
        else:
            print(f"\n=== step {step:06d} ===")

        try:
            D = load_indexed(snapshot_dir, L, step)
        except (KeyError, FileNotFoundError) as e:
            print(f"  [skip] Could not load Level 3 data for this step: {e}")
            continue

        u_lin      = D["u_lin"]       # flat (Fortran order)
        x_lin      = D["x_lin"]
        y_lin      = D["y_lin"]
        z_lin      = D["z_lin"]

        nx, ny, nz = len(x_lin), len(y_lin), len(z_lin)
        u3d     = shape_back_3d(u_lin, nx, ny, nz)

        # spacings (code units -> dimensional via len_scale when used)
        if len(x_lin) > 1:
            hx = float(x_lin[1] - x_lin[0])
            hy = float(y_lin[1] - y_lin[0])
            hz = float(z_lin[1] - z_lin[0])
        else:
            # Fallback if grid is degenerate (unlikely for Level 3)
            hx = hy = hz = 1.0

        # Dimensional coords
        x_dim = x_lin * len_scale
        y_dim = y_lin * len_scale
        z_dim = z_lin * len_scale

        # ---- Temperature volume ----
        # Always write if requested (or maybe default to true? User flag --write-temp)
        if args.write_temp:
            Tdim = temp_dim(u3d)
            grid = tvtk.ImageData(
                spacing=(hx*len_scale, hy*len_scale, hz*len_scale),
                origin=(x_dim[0], y_dim[0], z_dim[0]),
                dimensions=Tdim.shape,
            )
            scal = numpy_support.numpy_to_vtk(Tdim.ravel(order="F"))
            scal.SetName("temperature_K")
            # Using the fix for VTK 9.5 compat
            grid._vtk_obj.GetPointData().SetScalars(scal)
            
            fname = f"T_level3_{('layer_%d_'%L) if layered else ''}step_{step:09d}.vtk"
            outT_path = out_T / fname
            write_data(grid, str(outT_path))
            print(f"  Wrote temperature volume: {outT_path}")
        
        # Melt pool & deepest plane (Optional for Level 3, but useful if melting happens)
        melt_idx_flat = np.where(u_lin > 1.0)[0]
        if melt_idx_flat.size > 0:
            print(f"  [info] Melt detected in Level 3 ({len(melt_idx_flat)} cells u>1.0). Generating isosurface.")
            
            # Use whole domain for Level 3 marching cubes
            Tdim_full = temp_dim(u3d)
            
            try:
                verts, faces, normals, values = marching_cubes(
                    Tdim_full,
                    level=Tl,
                    spacing=(hx*len_scale, hy*len_scale, hz*len_scale),
                )
                verts[:, 0] += x_dim[0]
                verts[:, 1] += y_dim[0]
                verts[:, 2] += z_dim[0]
                
                # Check for empty verts
                if len(verts) > 0:
                    pts = tvtk.Points()
                    pts.from_array(verts)
                    cells = tvtk.CellArray()
                    for f in faces:
                        cells.insert_next_cell(len(f), f.tolist())
                    poly = tvtk.PolyData(points=pts, polys=cells)
                    
                    # No G/R for Level 3 yet
                    
                    fname_iso = f"Isosurface_level3_{('layer_%d_'%L) if layered else ''}step_{step:09d}.vtk"
                    out_iso_path = out_root / fname_iso
                    write_data(poly, str(out_iso_path))
                    print(f"  Wrote Level 3 Isosurface: {out_iso_path}")

            except Exception as e:
                print(f"  [warn] Marching cubes failed: {e}")

        else:
            print("  [info] No melted cells (u>1). Skipping isosurface.")

    print("\n[done]")

if __name__ == "__main__":
    main()
