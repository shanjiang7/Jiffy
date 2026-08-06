# Camera-ready reproduction pipeline

Every experimental result in the paper maps to a runner (sbatch, submit
from a Vista login node after `source env_vista.sh` in the repo root) and
a plot/analysis script. All runners are resumable: completed cases are
skipped, so re-submitting after a timeout continues where it stopped.

| Paper artifact | Runner(s) | Plot / analysis |
|---|---|---|
| Strong scaling, 1-8 ranks, four paths (fig) | `run_strong_scaling_sbatch.sh <path>` | `plot_strong_scaling.py` |
| 15-layer strong scaling to 64 ranks (fig) | `run_multilayer_baseline_sbatch.sh <path>` then `run_multilayer_sweep_sbatch.sh <path>` | `plot_strong_scaling.py` |
| Per-rank busy-time breakdown (fig) | (uses the 15-layer runs) | `plot_rank_breakdown.py` |
| Weak scaling to 64 ranks, two targets (fig) | `run_weak_scaling_sbatch.sh <path>` | `plot_weak_scaling.py` |
| Parametric study (fig) | `run_strong_scaling_sbatch.sh spiral_raster_eps5` (+ the tight arm above) | `plot_parametric_study.py` |
| Accuracy tables: serial references | `run_accuracy_serial_refs_sbatch.sh` | — |
| Accuracy tables: 32-rank observed errors | `run_accuracy_sbatch.sh <path>` | (summary printed by the job) |
| Accuracy tables: straight-line rows | `../scripts/accuracy/run_accuracy_straight_sbatch.sh` | — |
| Accuracy tables: max DAG in-degree column | — (CPU-only, cached r_eps) | `dag_indegree_stats.py` |
| Self-convergence table | `run_self_convergence_sbatch.sh` | (table printed by the job) |
| Scan-path overview figure | — | `plot_scan_paths.py` (viz-only coarsening applied in-script) |
| Multi-layer visualization (ParaView bundle) | — | `build_pv_bundle.py`, `make_pv_state.py`, `render_single_layer_pv.py`, `plot_multilayer_hero.py` |

Measurement protocol, applied by every runner:

- uniform and DP partitionings of the same point run back-to-back in the
  SAME allocation, which removes node-draw offsets from the comparison;
- 1-rank baselines and scaling points are timing-only (no snapshots);
- the load-imbalance metric is busy-time max/mean (skew), with waits shown
  as their own segment in the breakdown figure;
- every job sanitizes the inherited environment
  (`unset HERMES_CORRECTION_FIXED_COST_SS HERMES_TRACER_PROFILE`), and
  `planning_summary.json` records the cost-model constants per run.

Speedup/efficiency tables are collected by `../scripts/scaling/collect_scaling.py`.

Historical experiment drivers (cost-model gates, epsilon sweeps, the
removed discontinuous-hybrid path, ablations) were deleted from the tree
during the camera-ready cleanup; they remain available in git history.
