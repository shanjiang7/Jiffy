#!/bin/bash
# Validate the A_path (max in-degree) error-budget correction at 32 ranks.
#
# Background: at 31 cuts the plain tol1e4 chords DAG exceeds its 1e-4 target
# (hybrid 4.84e-4, bull 1.08e-4) because up to A_path just-below-epsilon
# neglected sources superpose at a cut — outside the single-segment
# calibration. --path-complexity measures A_path as the max in-degree of the
# retained DAG, splits the budget (local = target / A_path), tightens epsilon
# via the Table I calibration, and rebuilds the DAG:
#   hybrid: A_path=35 -> epsilon 5 K -> 0.161 K, cut depth 48 -> 418
#   bull:   A_path (see pre-warm log)  -> its own adjusted epsilon/DAG
# Both adjusted lookups are pre-warmed in .hermes_cache (avoids a 32-rank
# cache-build race).
#
# tol1e7 is NOT included: the 31-cut runs showed it is already insensitive to
# cut count (its honest horizon retains everything relevant), and a budget
# split by A_path=196 would demand rel-L2 ~5e-10 — beyond the calibration.
#
# Setup mirrors run_accuracy_cuts_*_sbatch.sh: 32 MPI ranks on 8 GH200 nodes
# (4 ranks/GPU), serial references reused, resumable, snapshots deleted after
# each comparison is saved.
#
# Usage:  sbatch run_accuracy_pathcomplexity_sbatch.sh

#SBATCH -J acc_apath
#SBATCH -N 8
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH -t 2:00:00
#SBATCH -o logs/acc_apath_%j.out
#SBATCH -e logs/acc_apath_%j.err
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

for P in hybrid bull; do
  ROOT=outputs/accuracy_${P}_tol1e4_h30
  if [ "$P" = "hybrid" ]; then
    CFG=configs/hybrid_spiral_raster_tol1e4.ini
  else
    CFG=configs/examples/bull_tol1e4.ini
  fi

  if [ ! -d "${ROOT}/serial/snapshots_ser" ]; then
    echo "ERROR: serial reference ${ROOT}/serial missing" >&2
    exit 1
  fi
  if [ -f "${ROOT}/compare_par32_apath/comparison_summary.json" ]; then
    echo " [$(date)] ${P}/tol1e4/par32-apath: already done, skipping"
    continue
  fi

  echo "======================================================"
  echo " [$(date)] ${P}/tol1e4: parallel 32 ranks, --path-complexity (target 1e-4)"
  echo "======================================================"
  srun -N 8 -n 32 --ntasks-per-node=4 python src/hermes/scripts/segment_correction/main.py \
    --config "${SIM_CONFIG}" --path-config "${CFG}" \
    --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
    --planner-mode "${PLANNER_MODE}" --no-export-dag \
    --path-complexity --path-complexity-config configs/path_complexity.ini \
    --out-dir "${ROOT}/par32_apath"

  echo "------ [$(date)] ${P}/tol1e4/par32-apath: rel-L2 comparison (source-on only) ------"
  srun -N 1 -n 1 --ntasks-per-node=1 python src/hermes/scripts/segment_correction/compare_runs.py \
    --par-snap-dir "${ROOT}/par32_apath/snapshots_par" \
    --ser-snap-dir "${ROOT}/serial/snapshots_ser" \
    --out-dir "${ROOT}/compare_par32_apath" \
    --source-on-only \
    --config "${SIM_CONFIG}" \
    --path-config "${CFG}" \
    --dt-us 10 | tail -6

  if [ -f "${ROOT}/compare_par32_apath/comparison_summary.json" ]; then
    echo " [$(date)] comparison saved - deleting par32 snapshots to save space"
    rm -rf "${ROOT}/par32_apath/snapshots_par" "${ROOT}/par32_apath/snapshots_par_meta"
  fi
done

echo ""
echo "[$(date)] all done. tol1e4 at 31 cuts: plain chords DAG vs A_path-adjusted:"
for P in hybrid bull; do
  for MODE in par32_chord par32_apath; do
    F=outputs/accuracy_${P}_tol1e4_h30/compare_${MODE}/comparison_summary.json
    LABEL=${MODE/par32_chord/plain  }
    LABEL=${LABEL/par32_apath/A_path}
    [ -f "$F" ] && python3 -c "
import json; d=json.load(open('$F'))
print(f'  ${P}/tol1e4 par32 ${LABEL}: max={d[\"max_rel_l2\"]:.4e} mean={d[\"mean_rel_l2\"]:.4e} n={d[\"num_compared\"]}')"
  done
done
