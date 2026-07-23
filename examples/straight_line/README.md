# Example: straight-line path, 1e-4 accuracy target

The smallest end-to-end JIFFY case: a single straight laser track, solved
serially for ground truth and in parallel on 2 ranks, then compared.

## Inputs

- `configs/examples/sim_calibration.ini` — simulation configuration: grid
  spacing (h = 30 µm), moving-domain dimensions, physical coefficients,
  solver settings.
- `configs/accuracy/straight_line_tol1e4.ini` — path configuration: the
  straight-line scan trajectory plus the calibrated 1e-4 accuracy preset
  (segment length and error threshold ε).

## Run

From the repository root, inside an interactive 2-node GPU allocation
(on TACC Vista: `idev -p gh-dev -N 2 -n 2 -t 00:30:00`):

```bash
bash examples/straight_line/run_example.sh
```

The script runs three steps: a single-GPU serial reference, a 2-rank
parallel run, and the comparison. On non-Slurm systems replace the `srun`
prefixes inside the script with `mpirun -np <n>`.

## Output

Under `outputs/example_straight_line/`:

- `serial/snapshots_ser/` — ground-truth temperature snapshots.
- `par2/` — the parallel run: `planning_summary.json` (segments, dependency
  DAG, partition, predicted per-rank loads), `snapshots_par/`, and timing
  information.
- `compare/comparison_summary.json` — the final verdict. The comparison
  step prints the maximum and mean relative L2 error over all compared
  snapshots; the maximum should be at or below the 1e-4 target.
