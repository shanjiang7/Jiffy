#!/bin/bash
# DP (exact_dp) vs uniform at level_K = 1 K -- the mid point of the epsilon
# sweep, closest in speedup to the paper's published figure (7.07x at 1 K vs
# 6.97x published). Complements job 854472 (tie at 5 K, DP +8.7% at 0.01 K).
# Predicted gap at 1 K: uniform/dp max-load ratio 1.035.

#SBATCH -J dpuni_e1
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:40:00
#SBATCH -o logs/dpuni_e1_%j.out
#SBATCH -e logs/dpuni_e1_%j.err
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

for MODE in exact_dp uniform; do
  OUT=outputs/dp_vs_uniform/eps1_${MODE}
  if [ -f "${OUT}/timing_summary.json" ]; then
    echo " [$(date)] eps1/${MODE}: already done, skipping"
    continue
  fi
  echo ""
  echo "###### [$(date)] Bull 8 ranks, eps=1K, planner=${MODE} ######"
  srun -N 8 -n 8 --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config configs/examples/sim_ex1.ini \
      --path-config configs/dev/bull_eps1.ini \
      --dt-us 10 --planner-mode "${MODE}" --correction-weight 0.25 \
      --timing-only --no-export-dag \
      --out-dir "${OUT}"
done

echo ""
echo "===== [$(date)] eps=1K: DP vs uniform, detailed ====="
python3 dev/compare_dp_uniform.py --tag eps1
