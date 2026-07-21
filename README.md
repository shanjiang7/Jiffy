# JIFFY — Printing in a JIFFY: a parallel-in-time heat transfer solver for additive manufacturing

Artifact for the SC26 paper. JIFFY parallelizes laser powder-bed heat transfer
simulation **in time**: the laser path is split into segments, each rank solves
its segments' source-on fields independently, and inter-segment thermal
influence is restored by superposing source-off *corrections* along a
segment-level dependency DAG. A calibrated threshold ε (Table I) bounds the
error of every neglected dependency, giving a tunable accuracy target
(rel-L2 1e-4 or 1e-7) at parallel speed.

## Artifact workflow

```
calibrate (optional, Table I)
   └─ src/hermes/scripts/segment_correction/calibrate_straight_line.py
serial reference                 ──►  ground truth temperature snapshots
   └─ src/hermes/scripts/segment_correction/serial_reference_run.py
parallel run                     ──►  DAG build + partition + multi-rank correction
   └─ src/hermes/scripts/segment_correction/main.py
compare                          ──►  rel-L2 (parallel vs serial)
   └─ src/hermes/scripts/segment_correction/compare_runs.py
```

Inspection tools: `plan_only.py` (DAG/partition preview without solving, incl.
`--path-complexity-report` A_path reporting) and `post/global_view*.py` (VTK
melt-history export for the paper's figures).

## Requirements

- NVIDIA GPU with CUDA 12.x (validated on GH200 120 GB, TACC Vista).
  Full 8-rank reproductions use 8 GPUs via Slurm; several MPI ranks may share
  one GPU (`bind_local_gpu` maps co-located ranks to the same device).
- MPI (validated with OpenMPI), Python 3.11.

```bash
module load cuda gcc openmpi          # site-specific; see env_vista.sh for TACC Vista
conda env create -f environment.yml
source env_vista.sh                   # activates env, sets PYTHONPATH/CUDA vars
```

On systems other than TACC Vista, adapt the `module load` / CUDA_HOME lines in
`env_vista.sh`; the PYTHONPATH line is location-independent.

## Reproducing the paper

All jobs are submitted from the repository root. Serial references are built
once and reused by later runs. Each accuracy job prints a final summary of
`max/mean rel-L2` per tolerance.

### 1. Straight-line accuracy (Tables IV/V anchor rows) — start here

```bash
sbatch scripts/accuracy/run_accuracy_straight_sbatch.sh
```

~5 min total on 8 GPUs (also the fastest end-to-end installation check).
Expected (digit-exact vs the printed values):

| target | max rel-L2 | paper |
|---|---|---|
| 1e-4 | 9.5957e-05 | 9.60e-05 |
| 1e-7 | 9.8060e-08 | 9.81e-08 |

### 2. Hybrid spiral-raster and Bull accuracy (Tables IV/V)

```bash
sbatch scripts/accuracy/run_accuracy_hybrid_sbatch.sh   # ~1.5 h first run (serial ref), ~15 min after
sbatch scripts/accuracy/run_accuracy_bull_sbatch.sh     # same shape
```

Expected: max rel-L2 well below the target at 8 ranks (typical observed:
hybrid ~1e-8 at both targets; Bull ~3e-6 / ~1e-8). The exact value depends on
the DP planner's cut placement; the target is a guaranteed upper bound, and
the straight line shows the bound is sharp when the worst case is realized.

### 3. Table I calibration (optional)

```bash
python src/hermes/scripts/segment_correction/calibrate_straight_line.py
```

Sweeps Lseg = 0.6–1.4 mm on the h = 30 µm calibration grid (single GPU, ~1 h)
and prints the (Lseg, ε, rel-L2) table against the built-in reference.

### 4. Strong scaling (Sec. V-C, one rank per GPU)

One job per scan path; each sweeps ranks 1-8 under both partitioning
strategies on the h = 18 um grid and prints a speedup table.

```bash
sbatch scripts/scaling/run_strong_scaling_sbatch.sh bull
sbatch scripts/scaling/run_strong_scaling_sbatch.sh texas
sbatch scripts/scaling/run_strong_scaling_sbatch.sh hybrid
sbatch scripts/scaling/run_strong_scaling_sbatch.sh hilbert
```

Aggregate all four paths into one table/figure once the jobs finish:

```bash
python scripts/scaling/collect_scaling.py --root outputs/strong_scaling_h18 --all --plot
```

Runs are resumable: a rank point whose `timing_summary.json` already exists is
skipped, so a timed-out job can simply be resubmitted.

### 5. Figures (melt-history views)

`src/hermes/post/global_view.py` converts a run's snapshots into VTK time
series (melt history + moving source plane) for ParaView.

## Repository layout

```
configs/examples/    canonical paths + simulation grids (sim_calibration.ini = h=30 µm grid)
configs/accuracy/    calibrated per-tolerance configs (tol1e4: Lseg 0.9 mm/ε 5 K; tol1e7: 1.3 mm/0.01 K)
configs/images/      raster path images (longhorn = Bull, texas, ...)
configs/dev/         ablation configs (_aabb published baseline, _lookup10) + experimental
scripts/accuracy/    official accuracy jobs (serial ref → 8-rank parallel → compare)
scripts/scaling/     strong-scaling / MPS jobs
src/hermes/          solver, DAG builder, partitioner, multi-rank runtime, post-processing
dev/                 debug tools and supplementary experiments (see dev/README.md)
```

## Key configuration knobs (`[dependency]` section)

- `level_K` — the ε threshold (K), calibrated in pair with `steps_per_segment`
  (Table I). Presets: `configs/accuracy/*_tol1e4.ini`, `*_tol1e7.ini`.
- `pair_test = chords` — segment pairs tested via exact chord-to-chord
  distances with per-chord deposit times (`aabb` reproduces the published
  bounding-box test; see `configs/dev/*_aabb.ini`).
- `lookup_source_on_steps = chord` — the influence-radius lookup deposits one
  chord of track (validated default). `segment` is the most conservative
  option; an integer gives an explicit step count (1 = the published
  single-pulse source, which under-resolves dense paths).
- `--path-complexity-report` (plan_only.py) — reports A_path, the max
  in-degree of the dependency DAG: the path-complexity metric that predicts
  error amplification at high rank counts (up to A_path sub-ε neglects
  superpose at dense cuts; see `dev/run_accuracy_cuts_*.sh` for the study).
- `--self-check` (main.py) — a-posteriori self-convergence error estimate and
  iterative repair: extends the retained corrections incrementally
  (`--self-check-horizon-step` supersegments per iteration,
  `--self-check-iters` iterations), reports the per-iteration rel-L2 shift and
  the cumulative estimate of the production error — no serial reference
  required. Validated at 31 cuts: estimated vs true production error agree to
  four digits on both test paths (see docs/error_analysis.md).

## Notes for reviewers

- Accuracy targets are guaranteed upper bounds; observed errors are usually
  far below because the exact-DP partitioner places cuts where coupling is
  weakest. The straight-line rows realize the worst case and match the bound
  within 4%.
- The `.hermes_cache/` directory memoizes the numerical influence-radius
  lookup tables; the first run of each configuration builds them on GPU
  (seconds to ~10 min depending on ε), later runs are cache-hot.
- `outputs/`, `logs/`, and `.hermes_cache/` are generated and git-ignored.
