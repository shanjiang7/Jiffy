#!/bin/bash
# SLURM MPS strong-scaling script for segment correction.
# Runs timing-only MPI jobs with 2 ranks per GPU/node under CUDA MPS and skips
# serial reference + snapshot comparison.
#
# Usage:
#   sbatch run_mps_sbatch.sh
#   GPU_SWEEP="8 6 4 2" MPS_SPLIT=50/50 PATH_CONFIG=configs/fast_heat.ini sbatch run_mps_sbatch.sh
#   PATH_CONFIG_SWEEP="configs/fast_heat.ini configs/texas.ini" GPU_SWEEP="8 4 2" sbatch run_mps_sbatch.sh
#   N_GPU="8,6,4,2" PLANNER_MODE="exact_dp,uniform" sbatch run_mps_sbatch.sh

#SBATCH -J segcorr_mps_scale
#SBATCH -N 8
#SBATCH -n 16
#SBATCH -t 01:00:00
#SBATCH -o logs/segcorr_mps_scale_%j.out
#SBATCH -e logs/segcorr_mps_scale_%j.err
#SBATCH -p gh-dev
#SBATCH -A ASC21034


PROJECT_DIR=/scratch/10226/shawnraul/Parallel_Hermes
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

# Avoid PMIx trying unavailable psec/munge component.
export PMIX_MCA_psec=native
export PRTE_MCA_psec=native

source /work/10226/shawnraul/vista/anaconda3/bin/activate
conda activate hermes

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

CONFIG=${CONFIG:-configs/sim_ex3.ini}
PATH_CONFIG=${PATH_CONFIG:-configs/fast_heat.ini}
PATH_CONFIG_SWEEP=${PATH_CONFIG_SWEEP:-${PATH_CONFIG}}
DT_US=${DT_US:-10}
PLANNER_MODE=${PLANNER_MODE:-"exact_dp uniform"}
CORRECTION_WEIGHT=${CORRECTION_WEIGHT:-0.25}
PAR_SOLVER_MODE=${PAR_SOLVER_MODE:-fused}
PROCS_PER_GPU=2
GPU_SWEEP=${GPU_SWEEP:-${N_GPU:-"8 7 6 5 4 3 2 1"}}
MPS_SPLIT=${MPS_SPLIT:-95/95}
MAX_GPUS=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-8}}
MAX_RANKS=${SLURM_NTASKS:-$((MAX_GPUS * PROCS_PER_GPU))}
RUN_ROOT=${RUN_ROOT:-outputs/segment_correction_mps_strong_scaling_${SLURM_JOB_ID:-manual}}

declare -a PLANNER_MODES=()
declare -A PLANNER_MODE_SEEN=()
for MODE in ${PLANNER_MODE//,/ }; do
  [ -z "${MODE}" ] && continue
  if ! [[ "${MODE}" =~ ^(uniform|exact_dp)$ ]]; then
    echo "[error] PLANNER_MODE entries must be 'uniform' or 'exact_dp', got '${MODE}' from '${PLANNER_MODE}'"
    exit 1
  fi
  if [[ -z "${PLANNER_MODE_SEEN[${MODE}]+x}" ]]; then
    PLANNER_MODES+=("${MODE}")
    PLANNER_MODE_SEEN["${MODE}"]=1
  fi
done

if [ "${#PLANNER_MODES[@]}" -eq 0 ]; then
  echo "[error] PLANNER_MODE resolved to an empty list."
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

IFS='/' read -r MPS_A MPS_B <<< "${MPS_SPLIT}"
if ! [[ "${MPS_A:-}" =~ ^[0-9]+$ ]] || ! [[ "${MPS_B:-}" =~ ^[0-9]+$ ]]; then
  echo "[error] MPS_SPLIT must be A/B with integer A,B in [1,100], got '${MPS_SPLIT}'"
  exit 1
fi
if [ "${MPS_A}" -lt 1 ] || [ "${MPS_A}" -gt 100 ] || [ "${MPS_B}" -lt 1 ] || [ "${MPS_B}" -gt 100 ]; then
  echo "[error] MPS_SPLIT must be A/B with integer A,B in [1,100], got '${MPS_SPLIT}'"
  exit 1
fi

declare -a GPU_COUNTS=()
declare -A GPU_SEEN=()
for tok in ${GPU_SWEEP//,/ }; do
  [ -z "${tok}" ] && continue
  if ! [[ "${tok}" =~ ^[0-9]+$ ]] || [ "${tok}" -lt 1 ]; then
    echo "[error] each GPU count in GPU_SWEEP/N_GPU must be a positive integer, got '${tok}'"
    exit 1
  fi
  if [ "${tok}" -gt "${MAX_GPUS}" ]; then
    echo "[error] requested ${tok} GPU nodes but allocated MAX_GPUS=${MAX_GPUS}"
    exit 1
  fi
  if [[ -z "${GPU_SEEN[${tok}]+x}" ]]; then
    GPU_COUNTS+=("${tok}")
    GPU_SEEN["${tok}"]=1
  fi
done

if [ "${#GPU_COUNTS[@]}" -eq 0 ]; then
  echo "[error] GPU_SWEEP/N_GPU resolved to an empty list."
  exit 1
fi

select_single_visible_gpu_and_mps_dirs='
set -euo pipefail
ORIG_CVD=${CUDA_VISIBLE_DEVICES:-}
if [ -n "${ORIG_CVD}" ]; then
  IFS="," read -r FIRST_DEV _ <<< "${ORIG_CVD}"
  export CUDA_VISIBLE_DEVICES="${FIRST_DEV}"
else
  export CUDA_VISIBLE_DEVICES=0
fi
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps_${USER}/segcorr_${SLURM_JOB_ID:-manual}/pipe"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps_${USER}/segcorr_${SLURM_JOB_ID:-manual}/log"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
'

start_mps_servers() {
  local num_nodes="$1"
  srun -N "${num_nodes}" -n "${num_nodes}" --ntasks-per-node=1 --cpu-bind=none --exact --export=ALL \
    bash -lc "${select_single_visible_gpu_and_mps_dirs}
echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
nvidia-cuda-mps-control -d
echo \"[mps-start] host=\$(hostname) cuda_visible=\${CUDA_VISIBLE_DEVICES} pipe=\${CUDA_MPS_PIPE_DIRECTORY}\""
}

stop_mps_servers() {
  local num_nodes="$1"
  srun -N "${num_nodes}" -n "${num_nodes}" --ntasks-per-node=1 --cpu-bind=none --exact --export=ALL \
    bash -lc "${select_single_visible_gpu_and_mps_dirs}
echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
echo \"[mps-stop] host=\$(hostname) cuda_visible=\${CUDA_VISIBLE_DEVICES} pipe=\${CUDA_MPS_PIPE_DIRECTORY}\""
}

ACTIVE_MPS_NODES=0
WRAPPER=$(mktemp "${PROJECT_DIR}/.tmp_wrappers/segcorr_gpu_bind_mps_XXXX.sh")
cleanup() {
  if [ "${ACTIVE_MPS_NODES}" -gt 0 ]; then
    stop_mps_servers "${ACTIVE_MPS_NODES}" || true
    ACTIVE_MPS_NODES=0
  fi
  rm -f "${WRAPPER}" || true
}
trap cleanup EXIT INT TERM

cat > "${WRAPPER}" <<EOFWRAP
#!/bin/bash
set -euo pipefail

LOCAL_RANK=\${SLURM_LOCALID:-\${OMPI_COMM_WORLD_LOCAL_RANK:-\${PMI_LOCAL_RANK:-0}}}
if [ "\${LOCAL_RANK}" -eq 0 ]; then
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${MPS_A}"
elif [ "\${LOCAL_RANK}" -eq 1 ]; then
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${MPS_B}"
else
  echo "[gpu_bind_mps] expected local rank 0/1 for PROCS_PER_GPU=2, got \${LOCAL_RANK}" >&2
  exit 1
fi

ORIG_CVD=\${CUDA_VISIBLE_DEVICES:-}
if [ -n "\${ORIG_CVD}" ]; then
  IFS=',' read -r FIRST_DEV _ <<< "\${ORIG_CVD}"
  export CUDA_VISIBLE_DEVICES="\${FIRST_DEV}"
else
  export CUDA_VISIBLE_DEVICES=0
fi

export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps_\${USER}/segcorr_\${SLURM_JOB_ID:-manual}/pipe"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps_\${USER}/segcorr_\${SLURM_JOB_ID:-manual}/log"
mkdir -p "\${CUDA_MPS_PIPE_DIRECTORY}" "\${CUDA_MPS_LOG_DIRECTORY}"

exec "\$@"
EOFWRAP
chmod +x "${WRAPPER}"

echo "======================================================"
echo " Segment Correction MPS Strong Scaling"
echo "  job_id             : ${SLURM_JOB_ID:-manual}"
echo "  config             : ${CONFIG}"
echo "  path_config_sweep  : ${PATH_CONFIG_LIST[*]}"
echo "  dt_us              : ${DT_US}"
echo "  planner_mode_sweep : ${PLANNER_MODES[*]}"
echo "  correction_weight  : ${CORRECTION_WEIGHT}"
echo "  parallel solver    : ${PAR_SOLVER_MODE}"
echo "  gpu_sweep          : ${GPU_COUNTS[*]}"
echo "  procs_per_gpu      : ${PROCS_PER_GPU}"
echo "  mps_split          : ${MPS_A}/${MPS_B}"
echo "  max_alloc_gpus     : ${MAX_GPUS}"
echo "  max_alloc_ranks    : ${MAX_RANKS}"
echo "  run_root           : ${RUN_ROOT}"
echo "======================================================"

mkdir -p "${RUN_ROOT}"

for PATH_CFG in "${PATH_CONFIG_LIST[@]}"; do
  PATH_TAG=$(basename "${PATH_CFG}")
  PATH_TAG="${PATH_TAG%.ini}"
  PATH_RUN_ROOT="${RUN_ROOT}/${PATH_TAG}"
  mkdir -p "${PATH_RUN_ROOT}"

  for MODE in "${PLANNER_MODES[@]}"; do
    MODE_RUN_ROOT="${PATH_RUN_ROOT}/${MODE}"
    mkdir -p "${MODE_RUN_ROOT}"

    for N_GPUS in "${GPU_COUNTS[@]}"; do
      N_RANKS=$((N_GPUS * PROCS_PER_GPU))
      if [ "${N_RANKS}" -gt "${MAX_RANKS}" ]; then
        echo "[error] requested ${N_RANKS} ranks from ${N_GPUS} GPUs but allocated MAX_RANKS=${MAX_RANKS}"
        exit 1
      fi

      PAR_OUT_DIR="${MODE_RUN_ROOT}/mps_${N_GPUS}g_${N_RANKS}r"
      mkdir -p "${PAR_OUT_DIR}"

      echo ""
      echo "[$(date)] MPS timing-only run: path_config=${PATH_CFG}, planner=${MODE}, ${N_GPUS} GPU node(s), ${N_RANKS} rank(s), split=${MPS_A}/${MPS_B}"

      start_mps_servers "${N_GPUS}"
      ACTIVE_MPS_NODES="${N_GPUS}"

      srun -N "${N_GPUS}" -n "${N_RANKS}" --ntasks-per-node="${PROCS_PER_GPU}" --cpu-bind=none --exact --export=ALL \
        "${WRAPPER}" \
        python src/hermes/scripts/segment_correction/main.py \
          --config "${CONFIG}" \
          --path-config "${PATH_CFG}" \
          --dt-us "${DT_US}" \
          --solver-mode "${PAR_SOLVER_MODE}" \
          --planner-mode "${MODE}" \
          --correction-weight "${CORRECTION_WEIGHT}" \
          --timing-only \
          --out-dir "${PAR_OUT_DIR}"

      stop_mps_servers "${N_GPUS}"
      ACTIVE_MPS_NODES=0
    done
  done
done

echo ""
echo "[$(date)] MPS strong-scaling sweep complete."
echo "  run_root : ${RUN_ROOT}"
