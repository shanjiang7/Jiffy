#!/bin/bash
# Accuracy validation of the chords DAG with a 10-step (point-like) mock source
# for the r_eps lookup, on the hybrid spiral-raster path.
#
# History:
#   job 838610 - single-pulse lookup: tol1e4/chords FAILED (max rel-L2 7.28e-2,
#                spiral centre; cut depth 6).
#   job 840888 - validated lookup_source_on_steps = segment (cut depth 114/365,
#                max rel-L2 5.98e-9 / 2.88e-8) and = chord (cut depth 48/352,
#                max rel-L2 1.58e-8 / 1.25e-8). Both pass; chord is now the
#                default in configs/examples/path_base.ini.
#   this job   - validates lookup_source_on_steps = 10 (cut depth 28/340), the
#                leanest point-like source candidate.
#
# Reuses the serial references from job 838610 (regenerated only if missing);
# the 10-step r_eps lookup tables are already cached in .hermes_cache.
#
# Usage:  sbatch run_accuracy_hybrid_fix_sbatch.sh

#SBATCH -J acc_hyb_fix
#SBATCH -N 8
#SBATCH -n 8
#SBATCH -t 2:00:00
#SBATCH -o logs/acc_hybrid_fix_%j.out
#SBATCH -e logs/acc_hybrid_fix_%j.err
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
  ROOT=outputs/accuracy_hybrid_${TOL}_h30
  CFG=configs/hybrid_spiral_raster_${TOL}_lookup10.ini

  if [ ! -d "${ROOT}/serial/snapshots_ser" ]; then
    echo "======================================================"
    echo " [$(date)] ${TOL}: serial reference missing - regenerating"
    echo "======================================================"
    srun -N 1 -n 1 python src/hermes/scripts/segment_correction/serial_reference_run.py \
      --config "${SIM_CONFIG}" --path-config "${CFG}" \
      --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
      --out-dir "${ROOT}/serial"
  else
    echo " [$(date)] ${TOL}: reusing serial reference from ${ROOT}/serial"
  fi

  echo "======================================================"
  echo " [$(date)] ${TOL}: parallel 8 ranks (chords, lookup = 10 steps)"
  echo "======================================================"
  srun -N 8 -n 8 python src/hermes/scripts/segment_correction/main.py \
    --config "${SIM_CONFIG}" --path-config "${CFG}" \
    --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
    --planner-mode "${PLANNER_MODE}" --no-export-dag \
    --out-dir "${ROOT}/par8_chords_src10"

  echo "------ [$(date)] ${TOL}/chords-src10: rel-L2 comparison (source-on only) ------"
  srun -N 1 -n 1 python src/hermes/scripts/segment_correction/compare_runs.py \
    --par-snap-dir "${ROOT}/par8_chords_src10/snapshots_par" \
    --ser-snap-dir "${ROOT}/serial/snapshots_ser" \
    --out-dir "${ROOT}/compare_chords_src10" \
    --source-on-only \
    --config "${SIM_CONFIG}" \
    --path-config "${CFG}" \
    --dt-us 10 | tail -6
done

echo ""
echo "[$(date)] all done. Summaries (src10 vs previously validated variants):"
for TOL in tol1e4 tol1e7; do
  for MODE in chords_src10 chords_chordsrc chords_segsrc chords aabb; do
    F=outputs/accuracy_hybrid_${TOL}_h30/compare_${MODE}/comparison_summary.json
    [ -f "$F" ] && python3 -c "
import json; d=json.load(open('$F'))
print(f'  ${TOL}/${MODE}: max={d[\"max_rel_l2\"]:.4e} mean={d[\"mean_rel_l2\"]:.4e} n={d[\"num_compared\"]}')"
  done
done
