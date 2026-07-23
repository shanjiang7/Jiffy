# JIFFY — Printing in a JIFFY: a parallel-in-time heat transfer solver for additive manufacturing

## 1. Description

Artifact for the SC26 paper. JIFFY parallelizes laser powder-bed heat transfer
simulation **in time**: the laser path is split into segments, each rank solves
its segments' source-on fields independently, and inter-segment thermal
influence is restored by superposing source-off *corrections* along a
segment-level dependency DAG. A calibrated threshold ε (Table I) bounds the
error of every neglected dependency, giving a tunable accuracy target
(rel-L2 1e-4 or 1e-7) at parallel speed.

Repository layout:

```
INSTALL.md           step-by-step installation instructions
examples/            small runnable cases (start here)
configs/examples/    canonical paths + simulation grids
configs/accuracy/    calibrated per-tolerance configs (tol1e4: ε 5 K; tol1e7: ε 0.01 K)
configs/images/      raster path images (longhorn = Bull, texas, ...)
configs/dev/         ablation + experimental configs
scripts/accuracy/    accuracy jobs (serial reference → 8-rank parallel → compare)
scripts/scaling/     strong-/weak-scaling and MPS jobs
src/hermes/          solver, DAG builder, partitioner, multi-rank runtime, post-processing
paper/               paper PDF (artifact appendix PDF to follow)
dev/                 debug tools and supplementary experiments (see dev/README.md)
```

## 2. Installing the artifact

Requirements in brief: Linux, an NVIDIA GPU with a CUDA 12.x driver, MPI
(CUDA-aware MPI **not** required), Python 3.11. All package versions are
pinned in `environment.yml`.

See **[INSTALL.md](INSTALL.md)** for the full steps: system modules, Python
environment creation (conda or pip), environment activation
(`source env_vista.sh`), and an installation check.

## 3. Running the artifact

Every run takes two input files:

- a **simulation config** (`configs/examples/sim_*.ini`) — grid spacing,
  moving-domain dimensions, physical coefficients, solver settings;
- a **path config** — the laser scan trajectory plus the dependency/accuracy
  settings (segment length, threshold ε).

Start with the worked example in [`examples/straight_line/`](examples/straight_line/):
a straight track solved serially for ground truth, then in parallel on
2 ranks, then compared. Inside an interactive 2-node GPU allocation
(on TACC Vista: `idev -p gh-dev -N 2 -n 2 -t 00:30:00`):

```bash
bash examples/straight_line/run_example.sh
```

The three steps it runs, and their outputs:

1. **Serial reference** (`serial_reference_run.py`, single GPU) →
   ground-truth temperature snapshots.
2. **Parallel run** (`main.py` under MPI) → builds the dependency DAG and
   partition (`planning_summary.json`), runs the per-rank source-on solves
   and inter-rank corrections, writes parallel snapshots and timing data.
3. **Comparison** (`compare_runs.py`) → prints the max/mean relative L2
   error between the two; the maximum should be at or below the 1e-4
   target (`comparison_summary.json`).

The same three-step pattern, at 8 ranks and on the other scan paths, is what
the batch jobs in `scripts/` automate.

## 4. Reproducing the paper's tables and figures

All jobs are submitted from the repository root; serial references are built
once and reused. Runs are resumable — a completed point is skipped on
resubmission, so a timed-out job can simply be resubmitted.

**Accuracy, Tables IV/V** (each job prints a final max/mean rel-L2 summary):

```bash
sbatch scripts/accuracy/run_accuracy_straight_sbatch.sh   # ~5 min on 8 GPUs
sbatch scripts/accuracy/run_accuracy_hybrid_sbatch.sh     # ~1.5 h first run, ~15 min after
sbatch scripts/accuracy/run_accuracy_bull_sbatch.sh       # same shape
```

Expected for the straight line (digit-exact vs the printed values):
max rel-L2 = 9.5957e-05 (1e-4 target, paper: 9.60e-05) and 9.8060e-08
(1e-7 target, paper: 9.81e-08). Bull and Spiral-Raster land well below
their targets: the DP partitioner places cuts where coupling is weakest,
while the straight line realizes the worst case.

**Table I calibration** (optional; presets already shipped):

```bash
python src/hermes/scripts/segment_correction/calibrate_straight_line.py
```

**Strong scaling, Sec. V-C** (one rank per GPU; one job per scan path):

```bash
sbatch scripts/scaling/run_strong_scaling_sbatch.sh bull     # also: texas, hybrid, hilbert
python scripts/scaling/collect_scaling.py --root outputs/strong_scaling_h18 --all --plot
```

**Weak scaling, Sec. V-D** (problem grows with the rank count):

```bash
sbatch scripts/scaling/run_weak_scaling_sbatch.sh hybrid     # also: bull
python scripts/scaling/collect_scaling.py --root outputs/weak_scaling_h18 --all --weak --plot
```

`collect_scaling.py` aggregates the per-run timing summaries into the
speedup/efficiency tables and plots corresponding to the paper's scaling
figures.

**Melt-history figures**: `src/hermes/post/global_view.py` converts a run's
snapshots into VTK time series for ParaView.

## Key configuration knobs (`[dependency]` section)

- `level_K` — the ε threshold (K), calibrated in pair with
  `steps_per_segment` (Table I). Presets: `configs/accuracy/*_tol1e4.ini`,
  `*_tol1e7.ini`.
- `pair_test = chords` — segment pairs tested via exact chord-to-chord
  distances with per-chord deposit times (`aabb` reproduces the published
  bounding-box test; see `configs/dev/*_aabb.ini`).
- `lookup_source_on_steps = chord` — the influence-radius lookup deposits one
  chord of track (validated default).
- `--path-complexity-report` (plan_only.py) — reports A_path, the max
  in-degree of the dependency DAG: predicts error amplification at high rank
  counts.
- `--self-check` (main.py) — a-posteriori self-convergence error estimate and
  iterative repair; no serial reference required (see docs/error_analysis.md).

## Notes for reviewers

- Accuracy targets are guaranteed upper bounds; observed errors are usually
  far below because the exact-DP partitioner places cuts where coupling is
  weakest. The straight-line rows realize the worst case and match the bound
  within 4%.
- The `.hermes_cache/` directory memoizes the numerical influence-radius
  lookup tables; the first run of each configuration builds them on GPU
  (seconds to ~10 min depending on ε), later runs are cache-hot.
- `outputs/`, `logs/`, and `.hermes_cache/` are generated and git-ignored.
- The repository also carries the inherited HERMES multi-level solver
  (`src/hermes/scripts/multi_level_solver.py` and modules only it uses),
  kept for provenance — see https://github.com/aydinalperen7/hermes-gpu-heat.
  It is not used by any experiment in this paper.
