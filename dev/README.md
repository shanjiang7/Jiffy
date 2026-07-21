# dev/ — experimental and debugging material (not part of the artifact surface)

Nothing here is needed to reproduce the paper. Contents:

- `serial_run.py` — executes the *planned* runtime components sequentially
  (plan-emulation debug tool). NOT the ground-truth reference; that is
  `src/hermes/scripts/segment_correction/serial_reference_run.py`.
- `serial_emulated_run.py` + `emulated_runtime.py` — single-GPU emulation of an
  N-rank run (debugging multi-rank behaviour on one device).
- `dag_pipeline.py` — standalone DAG export CLI (superseded by
  `plan_only.py --export-dag`).
- `run_accuracy_cuts_*.sh` — error-vs-cuts study: 32 MPI ranks on 8 GPUs,
  chord vs 10-step lookup sources (shows the tol1e4 guarantee is
  rank-count-dependent; see the A_path discussion in the top-level README).
- `run_accuracy_hybrid_fix_sbatch.sh` — 8-rank lookup-source ablation runner.
- `run_selfcheck_{hybrid,bull}_sbatch.sh` — self-convergence ladder studies
  (estimate-vs-truth tables at 31 cuts; see docs/error_analysis.md).
- `run_mpi_diagnostic_*.sh` — gamma-diagnostic runs. **Superseded and no longer
  runnable**: they pass `--diagnostic-check`/`--diagnostic-config`, which were
  retired from `main.py`. That mode ran a second complete pass at level_K/gamma
  and compared snapshots; `--self-check` obtains the same information
  incrementally, without repeating any base solve. Kept for reference only
  (`configs/dev/diagnostic_check.ini` holds their settings).
- `run_mpi_sbatch_legacy.sh`, `run_mpi_fast_heat_uniform_8rank_sbatch.sh`,
  `run_mpi_strong_scaling_*.sh` — superseded strong-scaling runners (16-64 rank
  sweeps and env-var-driven variants). The paper's Sec. V-C experiment is
  `scripts/scaling/run_strong_scaling_sbatch.sh`. Note the legacy runner
  hard-coded a PROJECT_DIR pointing at a sibling repository.
- `quick_start.sh` — legacy environment snippet (superseded by `env_vista.sh`).

Ablation configs live in `configs/dev/` (`*_aabb.ini` pins the published AABB
pair test + single-pulse lookup; `*_lookup10.ini` uses a 10-step point-like
lookup source).
