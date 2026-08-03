#!/bin/bash
# SLURM script for one 8-rank segment-correction diagnostic run.
# It enables the buffered diagnostic check. The Python driver runs:
#   1. production DAG pass
#   2. stricter buffered DAG pass
#   3. snapshot rel-L2 comparison
#
# Usage:
#   sbatch run_mpi_diagnostic_8rank_sbatch.sh

#SBATCH -J segcorr_diag8
#SBATCH -N 8
#SBATCH -n 8
#SBATCH -t 0:30:00
#SBATCH -o logs/segcorr_diag8_%j.out
#SBATCH -e logs/segcorr_diag8_%j.err
#SBATCH -p gh
#SBATCH -A ASC21034

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scratch/10226/shawnraul/work/Jiffy}
cd "${PROJECT_DIR}"
mkdir -p logs outputs .tmp_wrappers

module purge
module load cuda
module load gcc/13.2
module load openmpi

export CUDA_HOME=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/12.5/targets/sbsa-linux
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export NUMBA_CUDA_DRIVER=/usr/lib64/libcuda.so
export CUDA_HOME="${TACC_CUDA_DIR:-${CUDA_HOME}}"
export LD_LIBRARY_PATH="/usr/lib64/:${LD_LIBRARY_PATH:-}"

source /work/10226/shawnraul/vista/anaconda3/bin/activate
conda activate hermes

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

CONFIG=${CONFIG:-configs/examples/sim_ex1.ini}
PATH_CONFIG=${PATH_CONFIG:-configs/examples/hybrid_spiral_raster.ini}
DIAGNOSTIC_CONFIG=${DIAGNOSTIC_CONFIG:-configs/diagnostic_check.ini}
DT_US=${DT_US:-10}
PLANNER_MODE=${PLANNER_MODE:-uniform}
CORRECTION_WEIGHT=${CORRECTION_WEIGHT:-0.25}
PAR_SOLVER_MODE=${PAR_SOLVER_MODE:-fused}
N_RANKS=8
MAX_RANKS=${SLURM_NTASKS:-8}
RUN_ROOT=${RUN_ROOT:-outputs/segment_correction_diagnostic_8rank_${SLURM_JOB_ID:-manual}}

if [ "${N_RANKS}" -gt "${MAX_RANKS}" ]; then
  echo "[error] requested ${N_RANKS} ranks but SLURM_NTASKS=${MAX_RANKS}"
  exit 1
fi

for REQUIRED_FILE in "${CONFIG}" "${PATH_CONFIG}" "${DIAGNOSTIC_CONFIG}"; do
  if [ ! -f "${REQUIRED_FILE}" ]; then
    echo "[error] required config not found: ${REQUIRED_FILE}"
    exit 1
  fi
done

if [ "${PLANNER_MODE}" != "uniform" ] && [ "${PLANNER_MODE}" != "exact_dp" ] && [ "${PLANNER_MODE}" != "dp_monotonicity" ]; then
  echo "[error] PLANNER_MODE must be uniform, exact_dp, or dp_monotonicity; got ${PLANNER_MODE}"
  exit 1
fi

WRAPPER=$(mktemp "${PROJECT_DIR}/.tmp_wrappers/segcorr_gpu_bind_XXXX.sh")
cleanup() {
  rm -f "${WRAPPER}" || true
}
trap cleanup EXIT

cat > "${WRAPPER}" <<'EOFWRAP'
#!/bin/bash
set -euo pipefail
LOCAL_RANK=${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-${PMI_LOCAL_RANK:-0}}}
ORIG_CVD=${CUDA_VISIBLE_DEVICES:-}
if [ -n "${ORIG_CVD}" ]; then
  IFS=',' read -r -a DEV_ARR <<< "${ORIG_CVD}"
  NDEV=${#DEV_ARR[@]}
  if [ "${NDEV}" -le 0 ]; then
    echo "[gpu_bind] invalid CUDA_VISIBLE_DEVICES='${ORIG_CVD}'" >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES=${DEV_ARR[$((LOCAL_RANK % NDEV))]}
else
  export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
fi
exec "$@"
EOFWRAP
chmod +x "${WRAPPER}"

mkdir -p "${RUN_ROOT}"

echo "======================================================"
echo " Segment Correction 8-Rank Diagnostic Run"
echo "  job_id                  : ${SLURM_JOB_ID:-manual}"
echo "  project_dir             : ${PROJECT_DIR}"
echo "  config                  : ${CONFIG}"
echo "  path_config             : ${PATH_CONFIG}"
echo "  diagnostic_config       : ${DIAGNOSTIC_CONFIG}"
echo "  dt_us                   : ${DT_US}"
echo "  planner_mode            : ${PLANNER_MODE}"
echo "  correction_weight       : ${CORRECTION_WEIGHT}"
echo "  solver_mode             : ${PAR_SOLVER_MODE}"
echo "  ranks                   : ${N_RANKS}"
echo "  run_root                : ${RUN_ROOT}"
echo "======================================================"

echo ""
echo "[$(date)] Starting diagnostic MPI run."
mpirun -n "${N_RANKS}" --bind-to none "${WRAPPER}" \
  python src/hermes/scripts/segment_correction/main.py \
    --config "${CONFIG}" \
    --path-config "${PATH_CONFIG}" \
    --dt-us "${DT_US}" \
    --solver-mode "${PAR_SOLVER_MODE}" \
    --planner-mode "${PLANNER_MODE}" \
    --correction-weight "${CORRECTION_WEIGHT}" \
    --no-export-dag \
    --diagnostic-check \
    --diagnostic-config "${DIAGNOSTIC_CONFIG}" \
    --out-dir "${RUN_ROOT}"

test -f "${RUN_ROOT}/diagnostic_summary.json"

echo ""
echo "[$(date)] Diagnostic run complete."
echo "  run_root           : ${RUN_ROOT}"
echo "  diagnostic summary : ${RUN_ROOT}/diagnostic_summary.json"
echo "  normal pass        : ${RUN_ROOT}/diagnostic_normal"
echo "  buffer pass        : ${RUN_ROOT}/diagnostic_buffer"
echo "  comparison         : ${RUN_ROOT}/diagnostic_compare"
