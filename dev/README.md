# Camera-ready reproduction pipeline

Every experimental result in the paper maps to a runner (sbatch, submit
from a Vista login node after `source env_vista.sh` in the repo root) and
a plot/analysis script. All runners are resumable: completed cases are
skipped, so re-submitting after a timeout continues where it stopped.

| Paper artifact | Runner(s) | Plot / analysis |
|---|---|---|
| Strong scaling, 1-8 ranks, four paths (fig) | `run_cr_strong_scaling_sbatch.sh <path>` | `plot_cr_strong_scaling.py` |
| 15-layer strong scaling to 64 ranks (fig) | `run_cr_ml_baseline_sbatch.sh <path>` then `run_cr_ml_sweep_sbatch.sh <path>` | `plot_cr_strong_scaling.py` |
| Per-rank busy-time breakdown (fig) | (uses the 15-layer runs; single-layer variant: `run_cr_strong_scaling64_sbatch.sh <path>`) | `plot_rank_breakdown.py` |
| Weak scaling to 64 ranks, two targets (fig) | `run_cr_weak_scaling_sbatch.sh <path>` | `plot_cr_weak_scaling.py` |
| Parametric study (fig) | `run_cr_strong_scaling_sbatch.sh continuous_hybrid_eps5` (+ the tight arm above) | `plot_cr_parametric.py` |
| Accuracy tables: serial references | `run_cr_serial_refs_s10_sbatch.sh` | — |
| Accuracy tables: 32-rank observed errors | `run_cr_accuracy_cuts_s10_sbatch.sh <path>` | (summary printed by the job) |
| Accuracy tables: straight-line rows | `../scripts/accuracy/run_accuracy_straight_sbatch.sh` | — |
| Accuracy tables: max DAG in-degree column | — (CPU-only, cached r_eps) | `dag_indegree_stats.py` |
| Self-convergence table | `run_cr_selfcheck_step4_sbatch.sh` | (table printed by the job) |
| Scan-path overview figure | — | `plot_scan_paths.py` (viz-only coarsened configs in `configs/dev/`) |
| 64-rank partition dump (visualizations) | `run_dump_ml15_plan_sbatch.sh <path>` (or `dump_ml15_plan.py` on a GPU node) | — |
| Multi-layer visualization (ParaView bundle) | — | `build_pv_bundle.py`, `make_pv_state.py`, `render_single_layer_pv.py`, `plot_ml15_parallel_hero.py` |

`camera_ready_experiments.md` documents the measurement protocol
(same-allocation pairing, timing-only runs, environment sanitization).
Speedup/efficiency tables are collected by `../scripts/scaling/collect_scaling.py`.

Historical experiment drivers (cost-model gates, epsilon sweeps, the
removed discontinuous-hybrid path, ablations) were deleted from the tree
during the camera-ready cleanup; they remain available in git history.
