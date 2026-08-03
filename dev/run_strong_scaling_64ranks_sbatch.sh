#!/bin/bash
# Camera-ready optional experiment: single-layer strong scaling extended to
# 64 ranks (dense rank counts), both partition modes. Writes into the same
# run root as the 1-8 campaign (outputs/cr_strong_scaling_h18/<path>), so
# the existing 1-8 results and baselines are reused and the final CSV spans
# 1-64. Note: at 64 ranks a single layer is ~26 SS/rank (starvation
# regime); this exists to show the granularity limit if wanted.
#
# Usage:  sbatch dev/run_strong_scaling_64ranks_sbatch.sh <bull|continuous_hybrid>

#SBATCH -J cr_strong64
#SBATCH -N 64
#SBATCH -n 64
#SBATCH --ntasks-per-node=1
#SBATCH -t 03:00:00
#SBATCH -o logs/cr_strong64_%j.out
#SBATCH -e logs/cr_strong64_%j.err
#SBATCH -p gh
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

PATH_NAME=${1:-bull}
case "${PATH_NAME}" in
  bull)              PATH_CFG=configs/examples/fast_heat.ini ;;
  continuous_hybrid) PATH_CFG=configs/examples/continuous_hybrid.ini ;;
  *) echo "ERROR: unknown path '${PATH_NAME}' (bull|continuous_hybrid)" >&2; exit 1 ;;
esac

SIM_CONFIG=configs/examples/sim_ex1.ini      # h = 18 um, CG 1e-10
DT_US=10
CORRECTION_WEIGHT=0.21
RANK_SWEEP=${RANK_SWEEP:-"16 24 32 40 48 56 64"}   # 1-8 already measured
PLANNER_MODES=${PLANNER_MODES:-"uniform exact_dp"}
RUN_ROOT=outputs/cr_strong_scaling_h18/${PATH_NAME}

echo "======================================================"
echo " CR single-layer strong scaling to 64 ranks, path=${PATH_NAME}"
echo "   sim config : ${SIM_CONFIG}  (eps = config default 0.01 K)"
echo "   ranks      : ${RANK_SWEEP}   planners: ${PLANNER_MODES}"
echo "   corr weight: ${CORRECTION_WEIGHT} (+ a0 = 7.9 SS code default)"
echo "   run root   : ${RUN_ROOT}  (1-8 results + baselines reused)"
echo "======================================================"

for MODE in ${PLANNER_MODES}; do
  for N in ${RANK_SWEEP}; do
    OUT_DIR="${RUN_ROOT}/${MODE}/parallel_${N}r"
    if [ -f "${OUT_DIR}/timing_summary.json" ]; then
      echo " [$(date)] ${PATH_NAME}/${MODE} ${N} ranks: already done, skipping"
      continue
    fi
    mkdir -p "${OUT_DIR}"
    echo ""
    echo "------ [$(date)] ${PATH_NAME}: ${MODE}, ${N} rank(s) ------"
    srun -N "${N}" -n "${N}" --ntasks-per-node=1 --kill-on-bad-exit=1 \
      python src/hermes/scripts/segment_correction/main.py \
        --config "${SIM_CONFIG}" \
        --path-config "${PATH_CFG}" \
        --dt-us "${DT_US}" \
        --planner-mode "${MODE}" --correction-weight "${CORRECTION_WEIGHT}" \
        --timing-only --no-export-dag \
        --out-dir "${OUT_DIR}"
  done
done

echo ""
echo "[$(date)] single-layer 1-64 sweep complete; speedup table:"
python3 scripts/scaling/collect_scaling.py --root "${RUN_ROOT}" --label "${PATH_NAME}_1layer64"
