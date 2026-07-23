#!/bin/bash
# Minimal end-to-end JIFFY example: straight-line path, 1e-4 accuracy target.
#
#   1. serial reference (single GPU)  -> ground-truth snapshots
#   2. 2-rank parallel run            -> plan + source-on solves + corrections
#   3. comparison                     -> max/mean rel-L2 vs the reference
#
# Run inside an interactive allocation with 2 GPU nodes, from the repo root.
# On TACC Vista:
#
#   idev -p gh-dev -N 2 -n 2 -t 00:30:00
#   bash examples/straight_line/run_example.sh
#
# On non-Slurm systems, replace the srun prefixes with the equivalent
# mpirun invocations (e.g. `mpirun -np 2`).

set -euo pipefail

source env_vista.sh

SIM=configs/examples/sim_calibration.ini        # grid + physics (h = 30 um)
CFG=configs/accuracy/straight_line_tol1e4.ini   # scan path + 1e-4 preset
ROOT=outputs/example_straight_line

echo "== 1/3 serial reference (single GPU) =="
srun -N 1 -n 1 python src/hermes/scripts/segment_correction/serial_reference_run.py \
  --config "${SIM}" --path-config "${CFG}" \
  --dt-us 10 --snap-every-steps 25 \
  --out-dir "${ROOT}/serial"

echo "== 2/3 parallel run (2 ranks) =="
srun -N 2 -n 2 python src/hermes/scripts/segment_correction/main.py \
  --config "${SIM}" --path-config "${CFG}" \
  --dt-us 10 --snap-every-steps 25 \
  --planner-mode exact_dp --no-export-dag \
  --out-dir "${ROOT}/par2"

echo "== 3/3 rel-L2 comparison =="
srun -N 1 -n 1 python src/hermes/scripts/segment_correction/compare_runs.py \
  --par-snap-dir "${ROOT}/par2/snapshots_par" \
  --ser-snap-dir "${ROOT}/serial/snapshots_ser" \
  --out-dir "${ROOT}/compare" \
  --source-on-only \
  --config "${SIM}" --path-config "${CFG}" --dt-us 10 | tail -6
