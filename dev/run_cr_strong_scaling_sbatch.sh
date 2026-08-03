#!/bin/bash
# Camera-ready campaign, experiment 1: strong scaling 1-8 ranks (1 rank/GPU),
# one job per path, uniform + exact_dp. Differences vs the submitted-version
# sweep (scripts/scaling/run_strong_scaling_sbatch.sh):
#   - correction weight 0.21 (affine model, a0 = 7.9 SS is the code default);
#   - epsilon is the config default 0.01 K (path_base.ini level_K — the
#     artifact-reproducible setting; the submitted figure mixed eps=0.1K);
#   - fresh run root outputs/cr_strong_scaling_h18/ (old results preserved).
# The 1-rank runs are the baselines: timing-only, no snapshots.
#
# Usage:  sbatch dev/run_cr_strong_scaling_sbatch.sh <bull|texas|continuous_hybrid|hilbert>

#SBATCH -J cr_strong
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH -o logs/cr_strong_%j.out
#SBATCH -e logs/cr_strong_%j.err
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
  bull)    PATH_CFG=configs/examples/fast_heat.ini ;;
  texas)   PATH_CFG=configs/examples/texas.ini ;;
  continuous_hybrid) PATH_CFG=configs/examples/continuous_hybrid.ini ;;
  continuous_hybrid_eps5) PATH_CFG=configs/dev/continuous_hybrid_eps5.ini ;;  # parametric study, eps=5K arm
  hilbert) PATH_CFG=configs/examples/hilbert.ini ;;
  *) echo "ERROR: unknown path '${PATH_NAME}' (bull|texas|continuous_hybrid|continuous_hybrid_eps5|hilbert)" >&2; exit 1 ;;
esac

SIM_CONFIG=configs/examples/sim_ex1.ini      # h = 18 um
if [ "${PATH_NAME}" = "continuous_hybrid_eps5" ]; then
  # Parametric-study low-accuracy arm: CG tolerance matched to the 1e-4
  # target (1e-5); everything else identical to sim_ex1.ini.
  SIM_CONFIG=configs/dev/sim_ex1_cg1e5.ini
fi
DT_US=10
CORRECTION_WEIGHT=0.21
RANK_SWEEP=${RANK_SWEEP:-"1 2 3 4 5 6 7 8"}
PLANNER_MODES=${PLANNER_MODES:-"uniform exact_dp"}
RUN_ROOT=outputs/cr_strong_scaling_h18/${PATH_NAME}

echo "======================================================"
echo " CR strong scaling (1 rank/GPU), path=${PATH_NAME}"
echo "   sim config : ${SIM_CONFIG}"
echo "   path config: ${PATH_CFG}   (eps = config default 0.01 K)"
echo "   ranks      : ${RANK_SWEEP}   planners: ${PLANNER_MODES}"
echo "   corr weight: ${CORRECTION_WEIGHT} (+ a0 = 7.9 SS code default)"
echo "   run root   : ${RUN_ROOT}"
echo "======================================================"

for MODE in ${PLANNER_MODES}; do
  for N in ${RANK_SWEEP}; do
    OUT_DIR="${RUN_ROOT}/${MODE}/parallel_${N}r"
    if [ -f "${OUT_DIR}/timing_summary.json" ]; then
      echo " [$(date)] ${PATH_NAME}/${MODE}/${N} ranks: already done, skipping"
      continue
    fi
    mkdir -p "${OUT_DIR}"
    echo ""
    echo "------ [$(date)] ${PATH_NAME}: ${MODE}, ${N} rank(s) ------"
    srun -N "${N}" -n "${N}" --ntasks-per-node=1 \
      python src/hermes/scripts/segment_correction/main.py \
        --config "${SIM_CONFIG}" --path-config "${PATH_CFG}" \
        --dt-us "${DT_US}" \
        --planner-mode "${MODE}" --correction-weight "${CORRECTION_WEIGHT}" \
        --timing-only --no-export-dag \
        --out-dir "${OUT_DIR}"
  done
done

echo ""
echo "[$(date)] sweep complete for ${PATH_NAME}; speedup table:"
python3 scripts/scaling/collect_scaling.py --root "${RUN_ROOT}" --label "${PATH_NAME}"
