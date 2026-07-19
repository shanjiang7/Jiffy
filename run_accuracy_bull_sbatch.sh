#!/bin/bash
# Bull-path (longhorn raster, paper Fig. 8) accuracy validation on gh-dev.
#
# For each tolerance (tol1e4, tol1e7):
#   1. serial reference (single GPU)
#   2. 8-rank parallel runs with the chords DAG:
#        chord - validated one-chord lookup source (path_base default)
#        src10 - 10-step point-like lookup source (leaner candidate)
#   3. rel-L2 comparison of each, source-on snapshots only
#
# Path size (plan-only, h = 30 um grid):
#   tol1e4: 2424 segments x 90 steps  (~218 mm track), cut depth 48
#   tol1e7: 1678 segments x 130 steps (~218 mm track), cut depth 318
# Slightly smaller than the hybrid spiral-raster job (838610), which fit the
# same serial+parallel structure inside the 2 h window.
#
# Usage:  sbatch run_accuracy_bull_sbatch.sh

#SBATCH -J acc_bull
#SBATCH -N 8
#SBATCH -n 8
#SBATCH -t 2:00:00
#SBATCH -o logs/acc_bull_%j.out
#SBATCH -e logs/acc_bull_%j.err
#SBATCH -p gh-dev
#SBATCH -A ASC21034

set -uo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scratch/10226/shawnraul/work/Jiffy}
cd "${PROJECT_DIR}"
mkdir -p logs outputs

source "${PROJECT_DIR}/env_vista.sh"

SIM_CONFIG=configs/examples/sim_calibration.ini
SNAP_EVERY=25
PLANNER_MODE=exact_dp

for TOL in tol1e4 tol1e7; do
  ROOT=outputs/accuracy_bull_${TOL}_h30
  CFG=configs/examples/bull_${TOL}.ini

  if [ ! -d "${ROOT}/serial/snapshots_ser" ]; then
    echo "======================================================"
    echo " [$(date)] bull/${TOL}: serial reference (single GPU)"
    echo "======================================================"
    srun -N 1 -n 1 python src/hermes/scripts/segment_correction/serial_reference_run.py \
      --config "${SIM_CONFIG}" --path-config "${CFG}" \
      --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
      --out-dir "${ROOT}/serial"
  else
    echo " [$(date)] bull/${TOL}: reusing serial reference from ${ROOT}/serial"
  fi

  for SRC in chord src10; do
    if [ "${SRC}" = "chord" ]; then
      RUN_CFG="${CFG}"
    else
      RUN_CFG=configs/examples/bull_${TOL}_lookup10.ini
    fi
    echo "======================================================"
    echo " [$(date)] bull/${TOL}: parallel 8 ranks (chords, lookup = ${SRC})"
    echo "======================================================"
    srun -N 8 -n 8 python src/hermes/scripts/segment_correction/main.py \
      --config "${SIM_CONFIG}" --path-config "${RUN_CFG}" \
      --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
      --planner-mode "${PLANNER_MODE}" --no-export-dag \
      --out-dir "${ROOT}/par8_chords_${SRC}"

    echo "------ [$(date)] bull/${TOL}/${SRC}: rel-L2 comparison (source-on only) ------"
    srun -N 1 -n 1 python src/hermes/scripts/segment_correction/compare_runs.py \
      --par-snap-dir "${ROOT}/par8_chords_${SRC}/snapshots_par" \
      --ser-snap-dir "${ROOT}/serial/snapshots_ser" \
      --out-dir "${ROOT}/compare_chords_${SRC}" \
      --source-on-only \
      --config "${SIM_CONFIG}" \
      --path-config "${RUN_CFG}" \
      --dt-us 10 | tail -6
  done
done

echo ""
echo "[$(date)] all done. Summaries:"
for TOL in tol1e4 tol1e7; do
  for SRC in chord src10; do
    F=outputs/accuracy_bull_${TOL}_h30/compare_chords_${SRC}/comparison_summary.json
    [ -f "$F" ] && python3 -c "
import json; d=json.load(open('$F'))
print(f'  bull/${TOL}/${SRC}: max={d[\"max_rel_l2\"]:.4e} mean={d[\"mean_rel_l2\"]:.4e} n={d[\"num_compared\"]}')"
  done
done
