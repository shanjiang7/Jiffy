#!/bin/bash
# Error-vs-cuts study (chords DAG) on the BULL (longhorn raster) path.
#
# 32 MPI ranks on 8 GH200 nodes (4 ranks per GPU; bind_local_gpu maps
# co-located ranks onto the shared device). Real 32-rank runs (31 cuts) for
# both lookup sources:
#   {tol1e4, tol1e7} x {chord, src10}   -> 4 parallel runs
# Compared vs the serial references from job 841088. Completed comparisons are
# skipped, so the job is resumable if it times out.
#
# Companion script: run_accuracy_cuts_hybrid_sbatch.sh (hybrid spiral-raster).
#
# Usage:  sbatch run_accuracy_cuts_bull_sbatch.sh

#SBATCH -J acc_cuts_bull
#SBATCH -N 8
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH -t 2:00:00
#SBATCH -o logs/acc_cuts_bull_%j.out
#SBATCH -e logs/acc_cuts_bull_%j.err
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
  if [ ! -d "${ROOT}/serial/snapshots_ser" ]; then
    echo "ERROR: serial reference ${ROOT}/serial missing" >&2
    exit 1
  fi

  for SRC in chord src10; do
    if [ "${SRC}" = "chord" ]; then
      CFG=configs/accuracy/bull_${TOL}.ini
    else
      CFG=configs/dev/bull_${TOL}_lookup10.ini
    fi
    if [ -f "${ROOT}/compare_par32_${SRC}/comparison_summary.json" ]; then
      echo " [$(date)] bull/${TOL}/par32-${SRC}: already done, skipping"
      continue
    fi

    echo "======================================================"
    echo " [$(date)] bull/${TOL}: parallel 32 ranks on 8 GPUs (chords, lookup=${SRC})"
    echo "======================================================"
    srun -N 8 -n 32 --ntasks-per-node=4 python src/hermes/scripts/segment_correction/main.py \
      --config "${SIM_CONFIG}" --path-config "${CFG}" \
      --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
      --planner-mode "${PLANNER_MODE}" --no-export-dag \
      --out-dir "${ROOT}/par32_${SRC}"

    echo "------ [$(date)] bull/${TOL}/par32-${SRC}: rel-L2 comparison (source-on only) ------"
    srun -N 1 -n 1 --ntasks-per-node=1 python src/hermes/scripts/segment_correction/compare_runs.py \
      --par-snap-dir "${ROOT}/par32_${SRC}/snapshots_par" \
      --ser-snap-dir "${ROOT}/serial/snapshots_ser" \
      --out-dir "${ROOT}/compare_par32_${SRC}" \
      --source-on-only \
      --config "${SIM_CONFIG}" \
      --path-config "${CFG}" \
      --dt-us 10 | tail -6

    if [ -f "${ROOT}/compare_par32_${SRC}/comparison_summary.json" ]; then
      echo " [$(date)] comparison saved - deleting par32 snapshots to save space"
      rm -rf "${ROOT}/par32_${SRC}/snapshots_par" "${ROOT}/par32_${SRC}/snapshots_par_meta"
    fi
  done
done

echo ""
echo "[$(date)] all done. Bull error vs number of partitions:"
for TOL in tol1e4 tol1e7; do
  for MODE in chords_chord chords_src10 par32_chord par32_src10; do
    F=outputs/accuracy_bull_${TOL}_h30/compare_${MODE}/comparison_summary.json
    LABEL=${MODE/chords_chord/par8  chord (7 cuts)}
    LABEL=${LABEL/chords_src10/par8  src10 (7 cuts)}
    LABEL=${LABEL/par32_chord/par32 chord (31 cuts)}
    LABEL=${LABEL/par32_src10/par32 src10 (31 cuts)}
    [ -f "$F" ] && python3 -c "
import json; d=json.load(open('$F'))
print(f'  bull/${TOL} ${LABEL}: max={d[\"max_rel_l2\"]:.4e} mean={d[\"mean_rel_l2\"]:.4e} n={d[\"num_compared\"]}')"
  done
done
