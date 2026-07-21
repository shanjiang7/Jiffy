#!/bin/bash
# Weak-scaling study, one rank per GPU (paper Sec. V-D).
#
# The problem grows with the rank count so the work per rank stays constant:
#   hybrid - P repeated spiral-raster motifs for a P-rank run
#   bull   - horizontal extent scaled by sqrt(P), so scanned area grows with P
# (verified: total solver steps per rank vary by <=1% across P = 1..8).
#
# Grid: configs/examples/sim_ex1.ini (h = 18 um), exact_dp partitioning,
# timing-only runs. Weak-scaling efficiency is T(1)/T(P); ideal is 1.0.
#
# One job per path family. Runs are resumable: a P whose timing_summary.json
# exists is skipped, so a timed-out job can simply be resubmitted.
#
# Usage:
#   sbatch scripts/scaling/run_weak_scaling_sbatch.sh hybrid
#   sbatch scripts/scaling/run_weak_scaling_sbatch.sh bull

#SBATCH -J weak_scale
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH -o logs/weak_scale_%j.out
#SBATCH -e logs/weak_scale_%j.err
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

FAMILY=${1:-hybrid}
case "${FAMILY}" in
  hybrid|bull) ;;
  *) echo "ERROR: unknown family '${FAMILY}' (hybrid|bull)" >&2; exit 1 ;;
esac

SIM_CONFIG=configs/examples/sim_ex1.ini      # h = 18 um
DT_US=10
PLANNER_MODE=exact_dp
# Measured source-off/source-on cost ratio at h = 18 um
# (dev/bench_source_on_off.py, job 852855: 0.232).
CORRECTION_WEIGHT=0.25
RANK_SWEEP=${RANK_SWEEP:-"1 2 3 4 5 6 7 8"}
RUN_ROOT=outputs/weak_scaling_h18/${FAMILY}

echo "======================================================"
echo " Weak scaling (1 rank/GPU), family=${FAMILY}"
echo "   sim config : ${SIM_CONFIG}"
echo "   configs    : configs/weak_scaling/${FAMILY}_p<P>.ini"
echo "   ranks      : ${RANK_SWEEP}   planner: ${PLANNER_MODE}"
echo "   run root   : ${RUN_ROOT}"
echo "======================================================"

for P in ${RANK_SWEEP}; do
  PATH_CFG="configs/weak_scaling/${FAMILY}_p${P}.ini"
  if [ ! -f "${PATH_CFG}" ]; then
    echo "ERROR: missing ${PATH_CFG}" >&2; exit 1
  fi
  OUT_DIR="${RUN_ROOT}/${PLANNER_MODE}/parallel_${P}r"
  if [ -f "${OUT_DIR}/timing_summary.json" ]; then
    echo " [$(date)] ${FAMILY} P=${P}: already done, skipping"
    continue
  fi
  mkdir -p "${OUT_DIR}"
  echo ""
  echo "------ [$(date)] ${FAMILY}: P=${P} ranks, path ${PATH_CFG} ------"
  srun -N "${P}" -n "${P}" --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config "${SIM_CONFIG}" --path-config "${PATH_CFG}" \
      --dt-us "${DT_US}" \
      --planner-mode "${PLANNER_MODE}" --correction-weight "${CORRECTION_WEIGHT}" \
      --timing-only --no-export-dag \
      --out-dir "${OUT_DIR}"
done

echo ""
echo "[$(date)] weak-scaling sweep complete for ${FAMILY}:"
python3 scripts/scaling/collect_scaling.py --root "${RUN_ROOT}" --label "${FAMILY}" --weak
