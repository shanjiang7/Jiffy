#!/bin/bash
# SLURM strong-scaling script for segment correction.
# Runs parallel MPI timing-only jobs for 64 and 32 ranks on fast_heat and texas.
#
# Usage:
#   sbatch run_mpi_strong_scaling_64_32_sbatch.sh

#SBATCH -J segcorr_64_32
#SBATCH -N 64
#SBATCH -n 64
#SBATCH -t 02:00:00
#SBATCH -o logs/segcorr_64_32_%j.out
#SBATCH -e logs/segcorr_64_32_%j.err
#SBATCH -p gh
#SBATCH -A ASC21034

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

CONFIG=${CONFIG:-configs/sim_ex1.ini}
PATH_CONFIG_SWEEP=${PATH_CONFIG_SWEEP:-"configs/fast_heat.ini configs/texas.ini"}
RANK_SWEEP=${RANK_SWEEP:-"64 32"}
DT_US=${DT_US:-10}
PLANNER_MODE=${PLANNER_MODE:-uniform}
CORRECTION_WEIGHT=${CORRECTION_WEIGHT:-0.25}
PAR_SOLVER_MODE=${PAR_SOLVER_MODE:-fused}
MAX_RANKS=${SLURM_NTASKS:-64}
RUN_ROOT=${RUN_ROOT:-outputs/segment_correction_strong_scaling_64_32_${SLURM_JOB_ID:-manual}}

if [ "${PLANNER_MODE}" != "uniform" ]; then
  echo "[error] this script is fixed to PLANNER_MODE=uniform"
  exit 1
fi

declare -a PATH_CONFIG_LIST=()
declare -A PATH_CONFIG_SEEN=()
for PATH_CFG in ${PATH_CONFIG_SWEEP//,/ }; do
  [ -z "${PATH_CFG}" ] && continue
  if [ ! -f "${PATH_CFG}" ]; then
    echo "[error] path config not found: '${PATH_CFG}'"
    exit 1
  fi
  if [[ -z "${PATH_CONFIG_SEEN[${PATH_CFG}]+x}" ]]; then
    PATH_CONFIG_LIST+=("${PATH_CFG}")
    PATH_CONFIG_SEEN["${PATH_CFG}"]=1
  fi
done

if [ "${#PATH_CONFIG_LIST[@]}" -eq 0 ]; then
  echo "[error] PATH_CONFIG_SWEEP resolved to an empty list."
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

echo "======================================================"
echo " Segment Correction Strong Scaling: 64/32 ranks"
echo "  job_id             : ${SLURM_JOB_ID:-manual}"
echo "  project_dir        : ${PROJECT_DIR}"
echo "  config             : ${CONFIG}"
echo "  path_config_sweep  : ${PATH_CONFIG_LIST[*]}"
echo "  dt_us              : ${DT_US}"
echo "  planner_mode       : ${PLANNER_MODE}"
echo "  correction_weight  : ${CORRECTION_WEIGHT}"
echo "  parallel solver    : ${PAR_SOLVER_MODE}"
echo "  rank_sweep         : ${RANK_SWEEP}"
echo "  max_alloc_ranks    : ${MAX_RANKS}"
echo "  run_root           : ${RUN_ROOT}"
echo "======================================================"

mkdir -p "${RUN_ROOT}"

for PATH_CFG in "${PATH_CONFIG_LIST[@]}"; do
  PATH_TAG=$(basename "${PATH_CFG}")
  PATH_TAG="${PATH_TAG%.ini}"
  PATH_RUN_ROOT="${RUN_ROOT}/${PATH_TAG}/${PLANNER_MODE}"
  mkdir -p "${PATH_RUN_ROOT}"

  for N_RANKS in ${RANK_SWEEP}; do
    if ! [[ "${N_RANKS}" =~ ^[0-9]+$ ]] || [ "${N_RANKS}" -lt 1 ]; then
      echo "[error] each rank count in RANK_SWEEP must be a positive integer, got '${N_RANKS}'"
      exit 1
    fi
    if [ "${N_RANKS}" -gt "${MAX_RANKS}" ]; then
      echo "[error] requested ${N_RANKS} ranks but SLURM_NTASKS=${MAX_RANKS}"
      exit 1
    fi

    PAR_OUT_DIR="${PATH_RUN_ROOT}/parallel_${N_RANKS}r"
    mkdir -p "${PAR_OUT_DIR}"

    echo ""
    echo "[$(date)] Parallel timing-only run: path_config=${PATH_CFG}, planner=${PLANNER_MODE}, ranks=${N_RANKS}"
    mpirun -n "${N_RANKS}" --bind-to none "${WRAPPER}" \
      python src/hermes/scripts/segment_correction/main.py \
        --config "${CONFIG}" \
        --path-config "${PATH_CFG}" \
        --dt-us "${DT_US}" \
        --solver-mode "${PAR_SOLVER_MODE}" \
        --planner-mode "${PLANNER_MODE}" \
        --correction-weight "${CORRECTION_WEIGHT}" \
        --no-export-dag \
        --timing-only \
        --out-dir "${PAR_OUT_DIR}"
  done
done

echo ""
echo "[$(date)] 64/32 strong-scaling sweep complete."
echo "  run_root : ${RUN_ROOT}"
