"""
Compare parallel vs serial temperature snapshots.

Loads ``layer_{LL}_ss_{SSSS}.npy`` files (or legacy ``ss_{SSSS}.npy``) from
--par-snap-dir and --ser-snap-dir, computes relative L2 error at each common
(layer, SS) index, and saves:
  - comparison.csv          (layer_idx, ss_idx, rel_l2_error)
  - comparison_summary.json (max, mean, all values)
  - comparison.png          (bar chart)

Usage:
    python src/hermes/scripts/segment_correction/compare_runs.py \\
        --par-snap-dir outputs/segment_correction_g8_dt2.0/snapshots_par \\
        --ser-snap-dir outputs/serial_dt2.0/snapshots_ser \\
        --out-dir outputs/comparison_g8_dt2.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compare parallel vs serial snapshots")
    p.add_argument("--par-snap-dir", required=True,
                   help="Dir containing parallel layer_*_ss_*.npy files")
    p.add_argument("--ser-snap-dir", required=True,
                   help="Dir containing serial layer_*_ss_*.npy files")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for comparison results")
    return p.parse_args(argv)


def load_snapshots(snap_dir: Path) -> dict[tuple, np.ndarray]:
    """Load snapshots keyed by (layer_idx, step_idx).

    Supports filename format: layer_{L:02d}_step_{STEP:09d}.npy
    """
    snaps: dict[tuple, np.ndarray] = {}
    for f in sorted(snap_dir.glob("layer_*_step_*.npy")):
        parts = f.stem.split("_")  # ["layer", L, "step", STEP]
        layer_idx = int(parts[1])
        step_idx = int(parts[3])
        snaps[(layer_idx, step_idx)] = np.load(f)
    return snaps


def rel_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    diff = a.astype(np.float64) - b.astype(np.float64)
    denom = float(np.linalg.norm(b.astype(np.float64)))
    return float(np.linalg.norm(diff) / max(denom, eps))


def main(argv=None):
    args = parse_args(argv)
    par_dir = Path(args.par_snap_dir).expanduser().resolve()
    ser_dir = Path(args.ser_snap_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading parallel snapshots from: {par_dir}")
    par_snaps = load_snapshots(par_dir)
    print(f"  {len(par_snaps)} files found")

    print(f"Loading serial snapshots from:   {ser_dir}")
    ser_snaps = load_snapshots(ser_dir)
    print(f"  {len(ser_snaps)} files found")

    common = sorted(set(par_snaps) & set(ser_snaps))
    if not common:
        print("[error] No common (layer, step) indices found. Check that both runs used the same --snap-every-steps and num-layers.")
        return

    print(f"\nComparing {len(common)} common snapshots...")
    rows: list[tuple[int, int, float]] = []
    for (layer_idx, step_idx) in common:
        err = rel_l2(par_snaps[(layer_idx, step_idx)], ser_snaps[(layer_idx, step_idx)])
        rows.append((layer_idx, step_idx, err))
        print(f"  layer {layer_idx:2d}  step {step_idx:6d}: rel-L2 = {err:.4e}")

    errors = [e for _, _, e in rows]
    labels = [f"L{l}s{s}" for l, s, _ in rows]

    # CSV
    csv_path = out_dir / "comparison.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("layer_idx,step_idx,rel_l2_error\n")
        for layer_idx, step_idx, err in rows:
            f.write(f"{layer_idx},{step_idx},{err:.6e}\n")
    print(f"\nSaved: {csv_path}")

    # Summary JSON
    summary = {
        "num_compared": len(rows),
        "max_rel_l2": float(max(errors)),
        "mean_rel_l2": float(np.mean(errors)),
        "keys": [(l, s) for l, s, _ in rows],
        "rel_l2_errors": errors,
    }
    json_path = out_dir / "comparison_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")

    print(f"\nMax  rel-L2 error: {summary['max_rel_l2']:.4e}")
    print(f"Mean rel-L2 error: {summary['mean_rel_l2']:.4e}")

    print(f"\nOutput dir: {out_dir}")


if __name__ == "__main__":
    main()
