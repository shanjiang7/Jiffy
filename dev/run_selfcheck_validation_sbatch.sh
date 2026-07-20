#!/bin/bash
# Validate the --self-check error estimator against serial ground truth on the
# hybrid spiral-raster path (tol1e4), at two operating points:
#   par8  (7 cuts):  true error ~1.6e-8  (well under the 1e-4 target)
#   par32 (31 cuts): true error ~3.9e-4  (EXCEEDS the target)
# The estimator earns its place if it tracks the first and flags the second -
# all without touching the serial reference it is compared against here.
#
# Smoke pre-validation (2-rank straight line): estimate 4.2800e-5 vs true
# 4.2800e-5, digit-exact.
#
# Usage:  sbatch dev/run_selfcheck_validation_sbatch.sh

#SBATCH -J selfcheck_val
#SBATCH -N 8
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH -t 2:00:00
#SBATCH -o logs/selfcheck_val_%j.out
#SBATCH -e logs/selfcheck_val_%j.err
#SBATCH -p gh-dev
#SBATCH -A ASC21034

set -uo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
while [ ! -f "${PROJECT_DIR}/env_vista.sh" ] && [ "${PROJECT_DIR}" != "/" ]; do
  PROJECT_DIR=$(dirname "${PROJECT_DIR}")
done
if [ ! -f "${PROJECT_DIR}/env_vista.sh" ]; then
  echo "ERROR: could not locate the Jiffy repo root (env_vista.sh)" >&2
  exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p logs outputs

source "${PROJECT_DIR}/env_vista.sh"

SIM_CONFIG=configs/examples/sim_calibration.ini
CFG=configs/accuracy/hybrid_spiral_raster_tol1e4.ini
ROOT=outputs/accuracy_hybrid_tol1e4_h30
SNAP_EVERY=25

if [ ! -d "${ROOT}/serial/snapshots_ser" ]; then
  echo "ERROR: serial reference ${ROOT}/serial missing" >&2
  exit 1
fi

for NR in 8 32; do
  echo "======================================================"
  echo " [$(date)] hybrid/tol1e4: ${NR} ranks with --self-check"
  echo "======================================================"
  srun -N 8 -n "${NR}" --ntasks-per-node=$((NR / 8)) python src/hermes/scripts/segment_correction/main.py \
    --config "${SIM_CONFIG}" --path-config "${CFG}" \
    --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
    --planner-mode exact_dp --no-export-dag \
    --self-check --self-check-gamma 2.0 \
    --out-dir "${ROOT}/par${NR}_selfcheck"

  echo "------ [$(date)] par${NR}: TRUE error vs serial reference ------"
  srun -N 1 -n 1 --ntasks-per-node=1 python src/hermes/scripts/segment_correction/compare_runs.py \
    --par-snap-dir "${ROOT}/par${NR}_selfcheck/snapshots_par" \
    --ser-snap-dir "${ROOT}/serial/snapshots_ser" \
    --out-dir "${ROOT}/compare_par${NR}_selfcheck" \
    --source-on-only \
    --config "${SIM_CONFIG}" --path-config "${CFG}" --dt-us 10 | tail -6

  if [ -f "${ROOT}/compare_par${NR}_selfcheck/comparison_summary.json" ]; then
    rm -rf "${ROOT}/par${NR}_selfcheck/snapshots_par" "${ROOT}/par${NR}_selfcheck/snapshots_par_meta"
  fi
done

echo ""
echo "[$(date)] estimate vs truth:"
L=$(ls -t logs/selfcheck_val_*.out | head -1)
grep -h "\[self-check\] estimated" "$L" | sed 's/^/  ESTIMATE  /'
for NR in 8 32; do
  F=${ROOT}/compare_par${NR}_selfcheck/comparison_summary.json
  [ -f "$F" ] && python3 -c "
import json; d=json.load(open('$F'))
print(f'  TRUTH     par${NR}: max={d[\"max_rel_l2\"]:.4e} mean={d[\"mean_rel_l2\"]:.4e}')"
done
