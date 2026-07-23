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
scripts/accuracy/    accuracy jobs (serial reference → 8-rank parallel → compare)
scripts/scaling/     strong-/weak-scaling and MPS jobs
src/hermes/          solver, DAG builder, partitioner, multi-rank runtime, post-processing
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

**Table I straight-line calibration**:

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

**Accuracy, Tables IV/V** (each job prints a final max/mean rel-L2 summary):

```bash
sbatch scripts/accuracy/run_accuracy_straight_sbatch.sh
sbatch scripts/accuracy/run_accuracy_hybrid_sbatch.sh
sbatch scripts/accuracy/run_accuracy_bull_sbatch.sh
```

**Melt-history figures**: `src/hermes/post/global_view.py` converts a run's
snapshots into VTK time series for ParaView.
