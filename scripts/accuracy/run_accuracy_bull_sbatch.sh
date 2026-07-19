#!/bin/bash
# Accuracy validation on the bull path (paper Table IV/V Bull rows).
#
# For each tolerance target (tol1e4 = 1e-4, tol1e7 = 1e-7):
#   1. serial reference (single GPU) - reused if already present
#   2. 8-rank parallel run (chords DAG, chord lookup source - the defaults)
#   3. rel-L2 comparison vs the serial reference (source-on snapshots only)
#
# Usage:  sbatch scripts/accuracy/run_accuracy_bull_sbatch.sh   (from repo root)

#SBATCH -J acc_bull
#SBATCH -N 8
#SBATCH -n 8
#SBATCH -t 2:00:00
#SBATCH -o logs/acc_bull_%j.out
#SBATCH -e logs/acc_bull_%j.err
#SBATCH -p gh-dev
#SBATCH -A ASC21034

set -uo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
cd "${PROJECT_DIR}"
mkdir -p logs outputs

source "${PROJECT_DIR}/env_vista.sh"

SIM_CONFIG=configs/examples/sim_calibration.ini
SNAP_EVERY=25
PLANNER_MODE=exact_dp

for TOL in tol1e4 tol1e7; do
  ROOT=outputs/accuracy_bull_${TOL}_h30
  CFG=configs/accuracy/bull_${TOL}.ini

  if [ ! -d "${ROOT}/serial/snapshots_ser" ]; then
    echo "====== [$(date)] bull/${TOL}: serial reference (single GPU) ======"
    srun -N 1 -n 1 python src/hermes/scripts/segment_correction/serial_reference_run.py \
      --config "${SIM_CONFIG}" --path-config "${CFG}" \
      --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
      --out-dir "${ROOT}/serial"
  else
    echo " [$(date)] bull/${TOL}: reusing serial reference"
  fi

  echo "====== [$(date)] bull/${TOL}: parallel 8 ranks ======"
  srun -N 8 -n 8 python src/hermes/scripts/segment_correction/main.py \
    --config "${SIM_CONFIG}" --path-config "${CFG}" \
    --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
    --planner-mode "${PLANNER_MODE}" --no-export-dag \
    --out-dir "${ROOT}/par8"

  echo "------ [$(date)] bull/${TOL}: rel-L2 comparison (source-on only) ------"
  srun -N 1 -n 1 python src/hermes/scripts/segment_correction/compare_runs.py \
    --par-snap-dir "${ROOT}/par8/snapshots_par" \
    --ser-snap-dir "${ROOT}/serial/snapshots_ser" \
    --out-dir "${ROOT}/compare" \
    --source-on-only \
    --config "${SIM_CONFIG}" --path-config "${CFG}" --dt-us 10 | tail -6

  if [ -f "${ROOT}/compare/comparison_summary.json" ]; then
    rm -rf "${ROOT}/par8/snapshots_par" "${ROOT}/par8/snapshots_par_meta"
  fi
done

echo ""
echo "[$(date)] summary (bull):"
for TOL in tol1e4 tol1e7; do
  F=outputs/accuracy_bull_${TOL}_h30/compare/comparison_summary.json
  [ -f "$F" ] && python3 -c "
import json; d=json.load(open('$F'))
print(f'  bull/${TOL}: max={d[\"max_rel_l2\"]:.4e} mean={d[\"mean_rel_l2\"]:.4e} n={d[\"num_compared\"]}')"
done
