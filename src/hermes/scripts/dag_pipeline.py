#!/usr/bin/env python3
"""
CLI entry point for the supersegment DAG pipeline.

Usage:
    python src/hermes/scripts/dag_pipeline.py \
        --path-config configs/fast_heat.ini --out-dir outputs/dag
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hermes.DAG.dependency import LookupRuntime
from hermes.pipelines.config import PipelineConfig
from hermes.pipelines.components import compute_dag_and_components, export_dag_results
from hermes.physics.material import phys_parameter
from hermes.runtime.config import load_config


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Full supersegment DAG pipeline: build, analyse, and visualize."
    )
    p.add_argument("--path-config", required=True,
                   help="Path-config INI file (e.g. configs/fast_heat.ini)")
    p.add_argument("--config", default="configs/sim_ex1.ini",
                   help="Base simulation config used to derive dt and laser velocity.")
    p.add_argument("--out-dir", default="outputs/dag",
                   help="Output directory (default: outputs/dag)")
    p.add_argument("--dt-us", type=float, help="Override dt in microseconds")
    p.add_argument(
        "--solver-mode",
        choices=("fused", "legacy"),
        default="fused",
        help="Solver mode used by numerical lookup.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg_path = Path(args.path_config).expanduser().resolve()
    if not cfg_path.is_file():
        print(f"[error] path-config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    sim_cfg_path = Path(args.config).expanduser().resolve()
    if not sim_cfg_path.is_file():
        print(f"[error] sim config not found: {sim_cfg_path}", file=sys.stderr)
        sys.exit(1)

    rc = load_config(sim_cfg_path)
    mat_override = rc.material.to_override_dict()
    t_spot_on = 2.0 * rc.laser.x_span_m / rc.laser.v
    phys = phys_parameter(rc.laser.Q, rc.laser.x_span_m, t_spot_on, mat_ch=mat_override)
    if args.dt_us is not None:
        dt_s = float(args.dt_us) * 1e-6
    elif rc.time.CFL is not None:
        dt_s = (float(rc.time.CFL) * (float(rc.level1.h_tuple[0]) ** 2)) / float(phys.kappa)
    elif rc.time.dt is not None:
        dt_s = float(rc.time.dt)
    else:
        print("[error] need either --dt-us, [time].CFL, or [time].dt in sim config", file=sys.stderr)
        sys.exit(1)

    cfg = PipelineConfig.from_ini(cfg_path).with_solver_motion(
        dt_s=dt_s,
        solver_velocity_mps=rc.laser.v,
    )
    import cupy as cp

    float_type = cp.float64 if rc.float_type_str.lower() == "float64" else cp.float32
    lookup_runtime = LookupRuntime(
        rc=rc,
        phys=phys,
        float_type=float_type,
        solver_mode=args.solver_mode,
        source_substeps=cfg.dependency.mock_numerical_source_steps,
    )
    result = compute_dag_and_components(cfg, lookup_runtime=lookup_runtime)
    
    if result is not None:
        export_dag_results(
            result=result,
            out_dir=Path(args.out_dir).expanduser().resolve(),
        )


if __name__ == "__main__":
    main()
