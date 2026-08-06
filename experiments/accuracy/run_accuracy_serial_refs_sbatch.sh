#!/bin/bash
# Accuracy study, step 1: build the four serial references at snapshot
# stride 10 — bull/spiral-raster x tol1e4/tol1e7, h = 30 um accuracy grid.
# Four single-GPU runs execute CONCURRENTLY on four nodes; references land
# in <root>/serial_s10/. Resumable: an existing serial_s10/snapshots_ser is
# skipped.
#
# Usage:  sbatch experiments/accuracy/run_accuracy_serial_refs_sbatch.sh

#SBATCH -J ser_s10
#SBATCH -N 4
#SBATCH -n 4
#SBATCH --ntasks-per-node=1
#SBATCH -t 2:00:00
#SBATCH -o logs/ser_s10_%j.out
#SBATCH -e logs/ser_s10_%j.err
#SBATCH -p gh-dev
#SBATCH -A ASC21034

set -uo pipefail
PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
while [ ! -f "${PROJECT_DIR}/env_vista.sh" ] && [ "${PROJECT_DIR}" != "/" ]; do
  PROJECT_DIR=$(dirname "${PROJECT_DIR}")
done
cd "${PROJECT_DIR}" || exit 1
mkdir -p logs outputs
source "${PROJECT_DIR}/env_vista.sh"
unset HERMES_CORRECTION_FIXED_COST_SS HERMES_TRACER_PROFILE

SIM_CONFIG=configs/examples/sim_calibration.ini
SNAP_EVERY=10

run_ref () {
  local PATHKEY=$1 TOL=$2
  local ROOT=outputs/accuracy_${PATHKEY}_${TOL}_h30
  local CFG
  case "${PATHKEY}" in
    bull)              CFG=configs/accuracy/bull_${TOL}.ini ;;
    spiral_raster) CFG=configs/accuracy/spiral_raster_${TOL}.ini ;;
  esac
  if [ -d "${ROOT}/serial_s10/snapshots_ser" ]; then
    echo " [$(date)] ${PATHKEY}/${TOL}: stride-10 serial ref exists, skipping"
    return
  fi
  echo "====== [$(date)] ${PATHKEY}/${TOL}: serial reference, stride ${SNAP_EVERY} ======"
  srun --kill-on-bad-exit=1 -N 1 -n 1 --exact python src/hermes/scripts/segment_correction/serial_reference_run.py \
    --config "${SIM_CONFIG}" --path-config "${CFG}" \
    --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
    --out-dir "${ROOT}/serial_s10" \
    > "logs/ser_s10_${PATHKEY}_${TOL}.log" 2>&1
  echo " [$(date)] ${PATHKEY}/${TOL}: done (rc=$?)"
}

run_ref bull   tol1e4 &
run_ref bull   tol1e7 &
run_ref spiral_raster tol1e4 &
run_ref spiral_raster tol1e7 &
wait

echo ""
echo "[$(date)] all stride-10 serial references:"
for PATHKEY in bull spiral_raster; do
  for TOL in tol1e4 tol1e7; do
    D=outputs/accuracy_${PATHKEY}_${TOL}_h30/serial_s10/snapshots_ser
    if [ -d "${D}" ]; then
      echo "  ${PATHKEY}/${TOL}: $(ls "${D}" | wc -l) snapshots"
    else
      echo "  ${PATHKEY}/${TOL}: MISSING"
    fi
  done
done
