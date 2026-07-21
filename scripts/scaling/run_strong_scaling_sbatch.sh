#!/bin/bash
# Strong-scaling study, one rank per GPU (paper Sec. V-C, first experiment).
#
# Grid: configs/examples/sim_ex1.ini (h = 18 um). For one scan path, sweeps
# ranks 1..8 under both partitioning strategies (uniform and exact_dp), with
# timing-only runs (no snapshots, no serial reference). Speedup is computed
# against the 1-rank baseline of the same strategy.
#
# One job per path keeps each submission inside a reasonable walltime; the
# 1-rank baselines dominate the runtime.
#
# Usage:
#   sbatch scripts/scaling/run_strong_scaling_sbatch.sh bull
#   sbatch scripts/scaling/run_strong_scaling_sbatch.sh texas
#   sbatch scripts/scaling/run_strong_scaling_sbatch.sh hybrid
#   sbatch scripts/scaling/run_strong_scaling_sbatch.sh hilbert
#
# Optional overrides:  RANK_SWEEP="8 4 2 1"  PLANNER_MODES="exact_dp"

#SBATCH -J strong_scale
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH -o logs/strong_scale_%j.out
#SBATCH -e logs/strong_scale_%j.err
#SBATCH -p gh
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

# ---- experiment definition (paper Sec. V-C) --------------------------------
PATH_NAME=${1:-bull}
case "${PATH_NAME}" in
  bull)    PATH_CFG=configs/examples/fast_heat.ini ;;
  texas)   PATH_CFG=configs/examples/texas.ini ;;
  hybrid)  PATH_CFG=configs/examples/hybrid_spiral_raster.ini ;;
  hilbert) PATH_CFG=configs/examples/hilbert.ini ;;
  *) echo "ERROR: unknown path '${PATH_NAME}' (bull|texas|hybrid|hilbert)" >&2; exit 1 ;;
esac

SIM_CONFIG=configs/examples/sim_ex1.ini      # h = 18 um
DT_US=10
SOLVER_MODE=fused
# Boundary-correction weight of the predicted workload model. Pinned to the
# solver default (0.75), the same value used by the accuracy studies.
CORRECTION_WEIGHT=0.75
RANK_SWEEP=${RANK_SWEEP:-"1 2 3 4 5 6 7 8"}
PLANNER_MODES=${PLANNER_MODES:-"uniform exact_dp"}
RUN_ROOT=outputs/strong_scaling_h18/${PATH_NAME}

echo "======================================================"
echo " Strong scaling (1 rank/GPU), path=${PATH_NAME}"
echo "   sim config : ${SIM_CONFIG}"
echo "   path config: ${PATH_CFG}"
echo "   ranks      : ${RANK_SWEEP}"
echo "   planners   : ${PLANNER_MODES}"
echo "   corr weight: ${CORRECTION_WEIGHT}"
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
        --dt-us "${DT_US}" --solver-mode "${SOLVER_MODE}" \
        --planner-mode "${MODE}" --correction-weight "${CORRECTION_WEIGHT}" \
        --timing-only --no-export-dag \
        --out-dir "${OUT_DIR}"
  done
done

echo ""
echo "[$(date)] sweep complete for ${PATH_NAME}; speedup table:"
python3 scripts/scaling/collect_scaling.py --root "${RUN_ROOT}" --label "${PATH_NAME}"
