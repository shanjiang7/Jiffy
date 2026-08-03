#!/usr/bin/env python3
"""
DAG in-degree statistics per path/configuration, for the accuracy tables:
in-degree of a supersegment = number of retained predecessor dependencies
feeding it (including the chain edge, so a straight path reads 1), a
partition-independent measure of revisit density (replaces "global max cut
depth", which conflates path structure with the partition).

Builds only the dependency DAG (no partitioning, no solve). The r_eps
lookup is served from .hermes_cache, so a GPU is not needed when the cache
is warm (all accuracy configs have been built many times).

Usage: python dev/dag_indegree_stats.py [config.ini ...]
"""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from hermes.DAG.dependency import LookupRuntime
from hermes.pipelines.components import compute_dag_and_components
from hermes.pipelines.config import PipelineConfig
from hermes.runtime.setup import load_sim_setup
from dataclasses import replace

DEFAULT = [
    "configs/accuracy/straight_line_tol1e4.ini",
    "configs/accuracy/straight_line_tol1e7.ini",
    "configs/accuracy/bull_tol1e4.ini",
    "configs/accuracy/bull_tol1e7.ini",
    "configs/accuracy/continuous_hybrid_tol1e4.ini",
    "configs/accuracy/continuous_hybrid_tol1e7.ini",
]


def main() -> None:
    configs = sys.argv[1:] or DEFAULT
    setup = load_sim_setup("configs/examples/sim_calibration.ini", dt_us=10)
    print(f"{'config':44s} {'SS':>6s} {'chain':>6s} {'cross':>7s} "
          f"{'mean-in':>8s} {'p99-in':>7s} {'max-in':>7s} @SS")
    for cfg in configs:
        pc = PipelineConfig.from_ini(cfg)
        pc = replace(pc, segments_per_supersegment=1)
        pc = pc.with_solver_motion(dt_s=setup.dt_s,
                                   solver_velocity_mps=setup.rc.laser.v)
        rt = LookupRuntime(
            rc=setup.rc, phys=setup.phys, float_type=np.float64,
            solver_mode="fused",
            source_on_steps=pc.dependency.lookup_source_on_steps,
            source_substeps=pc.dependency.mock_numerical_source_steps)
        res = compute_dag_and_components(pc, lookup_runtime=rt)
        n = len(res.supersegments)
        chain = sum(1 for e in res.edges if int(e.dst) == int(e.src) + 1)
        cross = [(int(e.src), int(e.dst)) for e in res.edges
                 if int(e.dst) != int(e.src) + 1]
        indeg = Counter(int(e.dst) for e in res.edges)
        vals = np.array([indeg.get(i, 0) for i in range(n)])
        arg = int(vals.argmax())
        print(f"{cfg:44s} {n:6d} {chain:6d} {len(cross):7d} "
              f"{vals.mean():8.2f} {np.percentile(vals, 99):7.0f} "
              f"{vals.max():7d} {arg}")


if __name__ == "__main__":
    main()
