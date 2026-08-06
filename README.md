# JIFFY — Printing in a JIFFY: a parallel-in-time heat transfer solver for additive manufacturing

## 1. Description

JIFFY is a GPU-based solver for the transient heat equation in laser
powder-bed fusion problems that appear in metal additive manufacturing.
The related paper appears in the ACM/IEEE SC'26 conference.

JIFFY parallelizes laser powder-bed heat transfer simulations  **both in space and time**.
In space, it relies on the HERMES single-GPU solver. In time it uses multi-GPUs: 
the laser path is split into segments, each rank solves
its segments' source-on fields independently, and inter-segment thermal
influence is restored by superposing source-off *corrections* along a
segment-level dependency DAG. A calibrated threshold ε (Table I) bounds the
error of every neglected dependency, giving a tunable accuracy target
(rel-L2 1e-4 or 1e-7) at parallel speed. 

The instructions below is for running on the `Vista` system on the Texas Advanced Computing Center (TACC)

Repository layout:

```
INSTALL.md           step-by-step installation instructions
examples/            small runnable cases (start here)
configs/examples/    canonical paths + simulation grids
configs/accuracy/    calibrated per-tolerance configs (tol1e4: ε 5 K; tol1e7: ε 0.01 K)
configs/images/      raster path images for the test cases in the SC'26 paper (Bull, Texas, and others)
experiments/         the paper's reproduction pipeline: one runner + one plot
                     script per figure/table (see experiments/README.md)
scripts/             shared utilities (scaling aggregation, straight-line accuracy job)
src/hermes/          single-GPU moving laser solver, DAG builder, partitioner, multi-rank runtime, post-processing
legacy/              the original standalone multi-level HERMES solver (provenance only)
docs/                error-analysis derivation notes
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

The same three-step pattern, at 8–64 ranks and on the other scan paths, is
what the batch jobs in `experiments/` automate.

## 4. Reproducing the paper's tables and figures

All jobs are submitted from the repository root; serial references are built
once and reused. Runs are resumable — a completed point is skipped on
resubmission, so a timed-out job can simply be resubmitted.

**Table I straight-line calibration**:

```bash
python src/hermes/scripts/segment_correction/calibrate_straight_line.py
```

**Strong scaling** (one rank per GPU; one job per scan path; the
`continuous_hybrid_eps5` case is the parametric study's low-accuracy arm):

```bash
sbatch experiments/run_strong_scaling_sbatch.sh bull     # also: texas, continuous_hybrid, hilbert, continuous_hybrid_eps5
python experiments/plot_strong_scaling.py
```

**15-layer strong scaling to 64 ranks** (baseline first, then the sweep):

```bash
B=$(sbatch --parsable experiments/run_multilayer_baseline_sbatch.sh bull | tail -1)
sbatch --dependency=afterok:$B experiments/run_multilayer_sweep_sbatch.sh bull   # also: continuous_hybrid
```

**Weak scaling** (problem grows with the rank count, two accuracy targets):

```bash
sbatch experiments/run_weak_scaling_sbatch.sh continuous_hybrid     # also: bull
python experiments/plot_weak_scaling.py
```

**Accuracy tables** (serial references once, then the 32-rank runs; the
straight-line rows come from `scripts/accuracy/`; the DAG in-degree column
is CPU-only):

```bash
R=$(sbatch --parsable experiments/run_accuracy_serial_refs_sbatch.sh | tail -1)
sbatch --dependency=afterok:$R experiments/run_accuracy_sbatch.sh bull   # also: continuous_hybrid, texas, hilbert
sbatch scripts/accuracy/run_accuracy_straight_sbatch.sh
python experiments/dag_indegree_stats.py
```

**Self-convergence table**:

```bash
sbatch experiments/run_self_convergence_sbatch.sh
```

The full figure/table → script map, with expected runtimes and the
measurement protocol, is in
[`experiments/README.md`](experiments/README.md).

**Melt-history figures**: `src/hermes/post/global_view.py` converts a run's
snapshots into VTK time series for ParaView.
