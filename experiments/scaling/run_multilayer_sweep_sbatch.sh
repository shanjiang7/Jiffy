#!/bin/bash
# Camera-ready experiment 2b, part 2: 15-layer Bull strong scaling sweep,
# ranks {8, 16, 32, 64} x {uniform, exact_dp}, 1 rank/GPU, timing-only.
# 25,170 SS -> exact_dp auto-delegates to the crossing-point search with
# layer-clamped charged spans. Submit with --dependency=afterok:<baseline>.
#
# Usage:  sbatch --dependency=afterok:<baseline job> experiments/scaling/run_multilayer_sweep_sbatch.sh

#SBATCH -J cr_ml_sweep
#SBATCH -N 64
#SBATCH -n 64
#SBATCH --ntasks-per-node=1
#SBATCH -t 08:00:00
#SBATCH -o logs/cr_ml_sweep_%j.out
#SBATCH -e logs/cr_ml_sweep_%j.err
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
  bull)              ML_CFG=configs/experiments/bull_ml15.ini ;;
  spiral_raster) ML_CFG=configs/experiments/spiral_raster_ml15.ini ;;
  *) echo "ERROR: unknown path '${PATH_NAME}' (bull|spiral_raster)" >&2; exit 1 ;;
esac
RUN_ROOT=outputs/cr_strong_scaling_ml15/${PATH_NAME}
RANK_SWEEP=${RANK_SWEEP:-"8 16 24 32 40 48 56 64"}
PLANNER_MODES=${PLANNER_MODES:-"exact_dp uniform"}

for N in ${RANK_SWEEP}; do
  for MODE in ${PLANNER_MODES}; do
    OUT_DIR="${RUN_ROOT}/${MODE}/parallel_${N}r"
    if [ -f "${OUT_DIR}/timing_summary.json" ]; then
      echo " [$(date)] ml15/${MODE}/${N} ranks: already done, skipping"
      continue
    fi
    mkdir -p "${OUT_DIR}"
    echo ""
    echo "------ [$(date)] 15-layer ${PATH_NAME}: ${MODE}, ${N} rank(s) ------"
    srun -N "${N}" -n "${N}" --ntasks-per-node=1 --kill-on-bad-exit=1 \
      python src/hermes/scripts/segment_correction/main.py \
        --config configs/examples/sim_ex1.ini \
        --path-config "${ML_CFG}" \
        --dt-us 10 \
        --planner-mode "${MODE}" --correction-weight 0.21 \
        --timing-only --no-export-dag \
        --out-dir "${OUT_DIR}"
  done
done

echo ""
echo "[$(date)] 15-layer sweep complete; speedup table:"
python3 experiments/scaling/collect_scaling.py --root "${RUN_ROOT}" --label "${PATH_NAME}_ml15"
