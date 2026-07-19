#!/bin/bash
# Straight-line accuracy re-validation after the midpoint-center radius change.
#
# The straight line is the calibration anchor: with the original lookup the
# 8-rank runs matched the paper digit-exact (tol1e4 max 9.5957e-5 vs printed
# 9.60e-5; tol1e7 9.8060e-8 vs 9.81e-8). The chord lookup source + midpoint
# radius changes the retained DAG, so the anchor must be re-measured:
#   {tol1e4, tol1e7} x {chord, src10}   -> 4 quick 8-rank runs (111/77 segs)
# Compared vs the existing serial references (snapshots every 25 steps).
#
# Usage:  sbatch run_accuracy_straight_sbatch.sh

#SBATCH -J acc_straight
#SBATCH -N 8
#SBATCH -n 8
#SBATCH -t 1:00:00
#SBATCH -o logs/acc_straight_%j.out
#SBATCH -e logs/acc_straight_%j.err
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
  ROOT=outputs/accuracy_straight_${TOL}_h30
  if [ ! -d "${ROOT}/serial/snapshots_ser" ]; then
    echo "ERROR: serial reference ${ROOT}/serial missing" >&2
    exit 1
  fi

  for SRC in chord src10; do
    if [ "${SRC}" = "chord" ]; then
      CFG=configs/straight_line_${TOL}.ini
    else
      CFG=configs/straight_line_${TOL}_lookup10.ini
    fi
    if [ -f "${ROOT}/compare_${SRC}/comparison_summary.json" ]; then
      echo " [$(date)] straight/${TOL}/${SRC}: already done, skipping"
      continue
    fi

    echo "======================================================"
    echo " [$(date)] straight/${TOL}: parallel 8 ranks (chords, lookup=${SRC})"
    echo "======================================================"
    srun -N 8 -n 8 python src/hermes/scripts/segment_correction/main.py \
      --config "${SIM_CONFIG}" --path-config "${CFG}" \
      --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
      --planner-mode "${PLANNER_MODE}" --no-export-dag \
      --out-dir "${ROOT}/par8_${SRC}"

    echo "------ [$(date)] straight/${TOL}/${SRC}: rel-L2 comparison (source-on only) ------"
    srun -N 1 -n 1 python src/hermes/scripts/segment_correction/compare_runs.py \
      --par-snap-dir "${ROOT}/par8_${SRC}/snapshots_par" \
      --ser-snap-dir "${ROOT}/serial/snapshots_ser" \
      --out-dir "${ROOT}/compare_${SRC}" \
      --source-on-only \
      --config "${SIM_CONFIG}" \
      --path-config "${CFG}" \
      --dt-us 10 | tail -6

    if [ -f "${ROOT}/compare_${SRC}/comparison_summary.json" ]; then
      echo " [$(date)] comparison saved - deleting par8 snapshots to save space"
      rm -rf "${ROOT}/par8_${SRC}/snapshots_par" "${ROOT}/par8_${SRC}/snapshots_par_meta"
    fi
  done
done

echo ""
echo "[$(date)] all done. Straight-line anchors (paper: 9.60e-5 / 9.81e-8):"
for TOL in tol1e4 tol1e7; do
  for MODE in compare compare_chord compare_src10; do
    F=outputs/accuracy_straight_${TOL}_h30/${MODE}/comparison_summary.json
    LABEL=${MODE/compare_chord/chord (new)}
    LABEL=${LABEL/compare_src10/src10 (new)}
    LABEL=${LABEL/compare/original anchor}
    [ -f "$F" ] && python3 -c "
import json; d=json.load(open('$F'))
print(f'  straight/${TOL} ${LABEL}: max={d[\"max_rel_l2\"]:.4e} mean={d[\"mean_rel_l2\"]:.4e} n={d[\"num_compared\"]}')"
  done
done
