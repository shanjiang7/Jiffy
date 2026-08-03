#!/bin/bash
# Camera-ready experiment 2b, part 1: the 15-layer 1-rank baseline
# (~3.5-4 h, single GPU, timing-only). On one rank the partitioner assigns
# the whole path either way, so the exact_dp baseline is copied as the
# uniform baseline instead of burning a second multi-hour run (identical
# physics; the copy is logged).
#
# Usage:  sbatch dev/run_multilayer_baseline_sbatch.sh <bull|continuous_hybrid>

#SBATCH -J cr_ml_base
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH -o logs/cr_ml_base_%j.out
#SBATCH -e logs/cr_ml_base_%j.err
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
  bull)              ML_CFG=configs/dev/bull_ml15.ini ;;
  continuous_hybrid) ML_CFG=configs/dev/continuous_hybrid_ml15.ini ;;
  *) echo "ERROR: unknown path '${PATH_NAME}' (bull|continuous_hybrid)" >&2; exit 1 ;;
esac
RUN_ROOT=outputs/cr_strong_scaling_ml15/${PATH_NAME}
OUT_DIR="${RUN_ROOT}/exact_dp/parallel_1r"

if [ ! -f "${OUT_DIR}/timing_summary.json" ]; then
  mkdir -p "${OUT_DIR}"
  echo "------ [$(date)] 15-layer ${PATH_NAME}: 1-rank baseline (timing-only) ------"
  srun -N 1 -n 1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config configs/examples/sim_ex1.ini \
      --path-config "${ML_CFG}" \
      --dt-us 10 \
      --planner-mode exact_dp --correction-weight 0.21 \
      --timing-only --no-export-dag \
      --out-dir "${OUT_DIR}"
else
  echo " [$(date)] baseline already done, skipping"
fi

if [ -f "${OUT_DIR}/timing_summary.json" ] && [ ! -f "${RUN_ROOT}/uniform/parallel_1r/timing_summary.json" ]; then
  mkdir -p "${RUN_ROOT}/uniform/parallel_1r"
  cp "${OUT_DIR}/timing_summary.json" "${RUN_ROOT}/uniform/parallel_1r/"
  echo "[$(date)] copied 1-rank baseline to uniform/ (single-rank runs are planner-independent)"
fi
