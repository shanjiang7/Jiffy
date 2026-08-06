# JIFFY — Printing in a JIFFY: a parallel-in-time heat transfer solver for additive manufacturing

## 1. Description

JIFFY is a GPU-based solver for the transient heat equation in laser
powder-bed fusion problems that appear in metal additive manufacturing.
The related paper appears in the proceedings of ACM/IEEE SC'26.

JIFFY parallelizes laser powder-bed heat transfer simulations **both in
space and time**. In space, it relies on the HERMES single-GPU solver. In
time, it uses multiple GPUs: the laser path is split into segments, each rank solves
its segments' source-on fields independently, and inter-segment thermal
influence is restored by superposing source-off *corrections* along a
segment-level dependency DAG. A calibrated threshold ε (Table I) bounds the
error of every neglected dependency, giving a tunable accuracy target
(rel-L2 1e-4 or 1e-7) at parallel speed.

The instructions below are for the `Vista` system at the Texas
Advanced Computing Center (TACC).

Repository layout:

```
INSTALL.md           step-by-step installation instructions
examples/            small runnable cases (start here)
configs/examples/    canonical paths + simulation grids
configs/accuracy/    calibrated per-tolerance configs (tol1e4: ε 5 K; tol1e7: ε 0.01 K)
configs/images/      raster path images for the test cases in the SC'26 paper (Bull, Texas, and others)
configs/experiments/ 15-layer variants and the parametric study's low-accuracy arm
configs/weak_scaling/ per-rank-count weak-scaling problems (p1-p64, two tolerances)
experiments/         the paper's reproduction pipeline, one directory per
                     experiment family: scaling/, accuracy/, visualization/
                     (see experiments/README.md for the figure/table map)
src/hermes/          single-GPU moving laser solver, DAG builder, partitioner, multi-rank runtime, post-processing
legacy/              the original standalone multi-level HERMES solver (provenance only)
```

## 2. Installing the artifact

Requirements in brief: Linux, an NVIDIA GPU with a CUDA 12.x driver, MPI,
Python 3.11. All package versions are pinned in `environment.yml`.

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

The same three-step pattern, at 8–64 ranks and on the other scan paths, is
what the batch jobs in `experiments/` automate.

## 4. Reproducing the paper's tables and figures

All jobs are submitted from the repository root; serial references are built
once and reused. Runs are resumable — a completed point is skipped on
resubmission, so a timed-out job can simply be resubmitted.

**Table I straight-line calibration**:

```bash
# single GPU node; sweeps the straight-line cases behind the calibration table
python src/hermes/scripts/segment_correction/calibrate_straight_line.py
```

**Strong scaling** (one rank per GPU; one job per scan path; the
`spiral_raster_eps5` case is the parametric study's low-accuracy arm):

```bash
sbatch experiments/scaling/run_strong_scaling_sbatch.sh bull     # also: texas, spiral_raster, hilbert, spiral_raster_eps5
python experiments/scaling/plot_strong_scaling.py
```

**15-layer strong scaling to 64 ranks**:

```bash
# 1-rank baseline — run this first and let it finish:
sbatch experiments/scaling/run_multilayer_baseline_sbatch.sh bull        # also: spiral_raster
# then the 8-64-rank sweep:
sbatch experiments/scaling/run_multilayer_sweep_sbatch.sh bull           # also: spiral_raster
```

**Weak scaling** (problem grows with the rank count, two accuracy targets):

```bash
sbatch experiments/scaling/run_weak_scaling_sbatch.sh spiral_raster     # also: bull
python experiments/scaling/plot_weak_scaling.py
```

**Accuracy tables**:

```bash
# serial references — run this first and let it finish (built once, reused):
sbatch experiments/accuracy/run_accuracy_serial_refs_sbatch.sh
# then the 32-rank runs:
sbatch experiments/accuracy/run_accuracy_sbatch.sh bull      # also: spiral_raster
# straight-line rows (self-contained: builds its own reference):
sbatch experiments/accuracy/run_accuracy_straight_sbatch.sh
# max DAG in-degree column (CPU-only):
python experiments/accuracy/dag_indegree_stats.py
```

**Self-convergence table**:

```bash
sbatch experiments/accuracy/run_self_convergence_sbatch.sh
```

The full figure/table → script map and the measurement protocol are in
[`experiments/README.md`](experiments/README.md).

**Melt-history figures**: `src/hermes/post/global_view.py` converts a run's
snapshots into VTK time series for ParaView.
