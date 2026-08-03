#!/bin/bash
# Dump the real exact_dp 64-rank partition for the 15-layer bull (planning
# only, no solve; single GPU). Feeds experiments/plot_multilayer_hero.py.
#
# Usage:  sbatch experiments/run_partition_dump_sbatch.sh [bull|continuous_hybrid]

#SBATCH -J ml15_plan
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -o logs/ml15_plan_%j.out
#SBATCH -e logs/ml15_plan_%j.err
#SBATCH -p gh
#SBATCH -A ASC21034

set -uo pipefail
PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
while [ ! -f "${PROJECT_DIR}/env_vista.sh" ] && [ "${PROJECT_DIR}" != "/" ]; do
  PROJECT_DIR=$(dirname "${PROJECT_DIR}")
done
cd "${PROJECT_DIR}" || exit 1
mkdir -p logs
source "${PROJECT_DIR}/env_vista.sh"
unset HERMES_CORRECTION_FIXED_COST_SS HERMES_TRACER_PROFILE

PATH_NAME=${1:-bull}
case "${PATH_NAME}" in
  bull)              ML_CFG=configs/experiments/bull_ml15.ini ;;
  continuous_hybrid) ML_CFG=configs/experiments/continuous_hybrid_ml15.ini ;;
  *) echo "ERROR: unknown path '${PATH_NAME}'" >&2; exit 1 ;;
esac

srun -N 1 -n 1 --kill-on-bad-exit=1 \
  python experiments/dump_partition_plan.py \
    --config configs/examples/sim_ex1.ini \
    --path-config "${ML_CFG}" \
    --dt-us 10 \
    --planner-mode exact_dp --correction-weight 0.21 \
    --world-size 64 \
    --out-dir "outputs/cr_strong_scaling_ml15/${PATH_NAME}/plan_dump_64r"
