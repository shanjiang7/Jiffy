#!/bin/bash
# Camera-ready campaign, experiment 5: weak scaling to 64 ranks (1 rank/GPU)
# at TWO accuracy targets. P in {1, 8, 16, 32, 64}, exact_dp only (settled
# 2026-07-29), w = 0.21 + a0 = 7.9 SS defaults. Problem growth: bull extent
# x sqrt(P), spiral-raster motif x P. Per (P, tol) the calibrated (Lseg, eps)
# pair from Table I is baked into configs/weak_scaling/cr/:
#   tol1e4 -> Lseg 90 steps, eps 5 K;  tol1e7 -> Lseg 130 steps, eps 0.01 K.
# Timing-only runs; efficiency = T(1)/T(P) per tolerance curve. Resumable.
#
# Usage:  sbatch dev/run_cr_weak_scaling_sbatch.sh <bull|continuous_hybrid>

#SBATCH -J cr_weak
#SBATCH -N 64
#SBATCH -n 64
#SBATCH --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH -o logs/cr_weak_%j.out
#SBATCH -e logs/cr_weak_%j.err
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

FAMILY=${1:-bull}
case "${FAMILY}" in
  bull|continuous_hybrid) ;;
  *) echo "ERROR: unknown family '${FAMILY}' (bull|continuous_hybrid)" >&2; exit 1 ;;
esac

SIM_CONFIG=configs/examples/sim_ex1.ini      # h = 18 um
DT_US=10
CORRECTION_WEIGHT=0.21
RANK_SWEEP=${RANK_SWEEP:-"1 8 16 24 32 40 48 56 64"}
TOLS=${TOLS:-"tol1e4 tol1e7"}
RUN_ROOT=outputs/cr_weak_scaling_h18/${FAMILY}

echo "======================================================"
echo " CR weak scaling (1 rank/GPU), family=${FAMILY}"
echo "   sim config : ${SIM_CONFIG}"
echo "   ranks      : ${RANK_SWEEP}   tolerances: ${TOLS}"
echo "   planner    : exact_dp, w=${CORRECTION_WEIGHT} (+ a0 = 7.9 SS default)"
echo "   run root   : ${RUN_ROOT}"
echo "======================================================"

for TOL in ${TOLS}; do
  for P in ${RANK_SWEEP}; do
    PATH_CFG="configs/weak_scaling/cr/${FAMILY}_p${P}_${TOL}.ini"
    if [ ! -f "${PATH_CFG}" ]; then
      echo "ERROR: missing ${PATH_CFG}" >&2; exit 1
    fi
    OUT_DIR="${RUN_ROOT}/${TOL}/exact_dp/parallel_${P}r"
    if [ -f "${OUT_DIR}/timing_summary.json" ]; then
      echo " [$(date)] ${FAMILY}/${TOL} P=${P}: already done, skipping"
      continue
    fi
    mkdir -p "${OUT_DIR}"
    echo ""
    echo "------ [$(date)] ${FAMILY}/${TOL}: P=${P} ranks, path ${PATH_CFG} ------"
    srun -N "${P}" -n "${P}" --ntasks-per-node=1 --kill-on-bad-exit=1 \
      python src/hermes/scripts/segment_correction/main.py \
        --config "${SIM_CONFIG}" --path-config "${PATH_CFG}" \
        --dt-us "${DT_US}" \
        --planner-mode exact_dp --correction-weight "${CORRECTION_WEIGHT}" \
        --timing-only --no-export-dag \
        --out-dir "${OUT_DIR}"
  done
done

echo ""
for TOL in ${TOLS}; do
  echo "[$(date)] weak-scaling efficiency, ${FAMILY}/${TOL}:"
  python3 scripts/scaling/collect_scaling.py --root "${RUN_ROOT}/${TOL}" \
    --label "${FAMILY}_${TOL}" --weak
done
