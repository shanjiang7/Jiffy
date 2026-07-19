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
import sys
from pathlib import Path

import numpy as np

_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compare parallel vs serial snapshots")
    p.add_argument("--par-snap-dir", required=True,
                   help="Dir containing parallel layer_*_ss_*.npy files")
    p.add_argument("--ser-snap-dir", required=True,
                   help="Dir containing serial layer_*_ss_*.npy files")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for comparison results")
    p.add_argument(
        "--source-on-only",
        action="store_true",
        help="Compare only snapshots whose outgoing path step has laser power on.",
    )
    p.add_argument("--config", default="configs/examples/sim_ex1.ini", help="Simulation config, required by --source-on-only.")
    p.add_argument("--path-config", help="Path config, required by --source-on-only.")
    p.add_argument("--dt-us", type=float, help="Override dt in microseconds for --source-on-only path sampling.")
    p.add_argument("--num-layers", type=int, help="Optional layer count override for --source-on-only path sampling.")
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


def _build_source_on_ranges_by_layer(
    *,
    config_path: Path,
    path_config_path: Path,
    dt_us: float | None,
    num_layers: int | None,
) -> dict[int, list[tuple[int, int]]]:
    from hermes.pipelines.config import PipelineConfig
    from hermes.pipelines.ss_builder import build_ss_from_cfg
    from hermes.runtime.setup import load_sim_setup

    setup = load_sim_setup(config_path, dt_us=dt_us)
    rc, dt_s = setup.rc, setup.dt_s

    pipeline_cfg = PipelineConfig.from_ini(path_config_path, num_layers=num_layers)
    pipeline_cfg = pipeline_cfg.with_solver_motion(dt_s=dt_s, solver_velocity_mps=rc.laser.v)
    ss_result = build_ss_from_cfg(pipeline_cfg)

    ranges_by_layer: dict[int, list[tuple[int, int]]] = {}
    segs_per_layer = int(ss_result.segments_per_layer)
    for layer_idx in range(int(ss_result.num_layers)):
        start = int(layer_idx) * int(segs_per_layer)
        end = start + int(segs_per_layer)
        step_offset = 0
        ranges: list[tuple[int, int]] = []
        for seg in ss_result.segments[start:end]:
            n_moves = int(seg.n_moves)
            if n_moves <= 0:
                continue
            if float(seg.power_W) > 0.0:
                range_start = int(step_offset)
                range_end = int(step_offset) + int(n_moves)
                if ranges and ranges[-1][1] == range_start:
                    ranges[-1] = (ranges[-1][0], range_end)
                else:
                    ranges.append((range_start, range_end))
            step_offset += int(n_moves)
        ranges_by_layer[int(layer_idx)] = ranges
    return ranges_by_layer


def _step_in_ranges(step_idx: int, ranges: list[tuple[int, int]]) -> bool:
    step = int(step_idx)
    lo = 0
    hi = len(ranges)
    while lo < hi:
        mid = (lo + hi) // 2
        start, end = ranges[mid]
        if step < int(start):
            hi = mid
        elif step >= int(end):
            lo = mid + 1
        else:
            return True
    return False


def _source_on_filter(ranges_by_layer: dict[int, list[tuple[int, int]]]):
    def _include(layer_idx: int, step_idx: int) -> bool:
        return _step_in_ranges(int(step_idx), ranges_by_layer.get(int(layer_idx), []))

    return _include


def compare_snapshot_dirs(
    *,
    test_snap_dir: Path,
    reference_snap_dir: Path,
    out_dir: Path,
    tol: float | None = None,
    test_label: str = "test",
    reference_label: str = "reference",
    include_key=None,
    filter_description: str | None = None,
) -> dict:
    test_dir = Path(test_snap_dir).expanduser().resolve()
    ref_dir = Path(reference_snap_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {test_label} snapshots from: {test_dir}")
    test_snaps = load_snapshots(test_dir)
    print(f"  {len(test_snaps)} files found")

    print(f"Loading {reference_label} snapshots from: {ref_dir}")
    ref_snaps = load_snapshots(ref_dir)
    print(f"  {len(ref_snaps)} files found")

    common_all = sorted(set(test_snaps) & set(ref_snaps))
    common = list(common_all)
    if include_key is not None:
        common = [
            (layer_idx, step_idx)
            for layer_idx, step_idx in common
            if include_key(int(layer_idx), int(step_idx))
        ]
    if not common:
        raise RuntimeError(
            "No common (layer, step) indices found. Check that both runs used "
            "the same snapshot stride and number of layers, and that the "
            "selected filter keeps at least one snapshot."
        )

    if filter_description is not None:
        print(
            f"\nFilter: {filter_description} "
            f"({len(common)} of {len(common_all)} common snapshots kept)"
        )
    print(f"\nComparing {len(common)} common snapshots...")
    rows: list[tuple[int, int, float]] = []
    for (layer_idx, step_idx) in common:
        err = rel_l2(test_snaps[(layer_idx, step_idx)], ref_snaps[(layer_idx, step_idx)])
        rows.append((layer_idx, step_idx, err))
        print(f"  layer {layer_idx:2d}  step {step_idx:6d}: rel-L2 = {err:.4e}")

    errors = [e for _, _, e in rows]

    csv_path = out_dir / "comparison.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("layer_idx,step_idx,rel_l2_error\n")
        for layer_idx, step_idx, err in rows:
            f.write(f"{layer_idx},{step_idx},{err:.6e}\n")
    print(f"\nSaved: {csv_path}")

    # Summary JSON
    summary = {
        "test_snapshot_dir": str(test_dir),
        "reference_snapshot_dir": str(ref_dir),
        "test_label": str(test_label),
        "reference_label": str(reference_label),
        "num_compared": len(rows),
        "num_common_before_filter": int(len(common_all)),
        "max_rel_l2": float(max(errors)),
        "mean_rel_l2": float(np.mean(errors)),
        "keys": [(l, s) for l, s, _ in rows],
        "rel_l2_errors": errors,
    }
    if filter_description is not None:
        summary["filter"] = str(filter_description)
        summary["num_filtered_out"] = int(len(common_all) - len(common))
    if tol is not None:
        summary["tol"] = float(tol)
        summary["passed"] = bool(float(summary["max_rel_l2"]) < float(tol))
    json_path = out_dir / "comparison_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")

    print(f"\nMax  rel-L2 error: {summary['max_rel_l2']:.4e}")
    print(f"Mean rel-L2 error: {summary['mean_rel_l2']:.4e}")
    if tol is not None:
        status = "PASS" if bool(summary["passed"]) else "FAIL"
        print(f"Diagnostic status: {status}  (tol={float(tol):.4e})")

    print(f"\nOutput dir: {out_dir}")
    return summary


def main(argv=None):
    args = parse_args(argv)
    include_key = None
    filter_description = None
    if bool(args.source_on_only):
        if not args.path_config:
            raise ValueError("--source-on-only requires --path-config.")
        ranges_by_layer = _build_source_on_ranges_by_layer(
            config_path=Path(args.config),
            path_config_path=Path(args.path_config),
            dt_us=args.dt_us,
            num_layers=args.num_layers,
        )
        include_key = _source_on_filter(ranges_by_layer)
        filter_description = "source_on_only"

    compare_snapshot_dirs(
        test_snap_dir=Path(args.par_snap_dir),
        reference_snap_dir=Path(args.ser_snap_dir),
        out_dir=Path(args.out_dir),
        test_label="parallel",
        reference_label="serial",
        include_key=include_key,
        filter_description=filter_description,
    )


if __name__ == "__main__":
    main()
