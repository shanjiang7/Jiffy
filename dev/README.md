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
- `run_accuracy_pathcomplexity_sbatch.sh` — 32-rank validation of the
  `--path-complexity` (A_path) error-budget correction.
- `run_mpi_diagnostic_*.sh` — gamma-diagnostic runs.
- `quick_start.sh` — legacy environment snippet (superseded by `env_vista.sh`).

Ablation configs live in `configs/dev/` (`*_aabb.ini` pins the published AABB
pair test + single-pulse lookup; `*_lookup10.ini` uses a 10-step point-like
lookup source).
