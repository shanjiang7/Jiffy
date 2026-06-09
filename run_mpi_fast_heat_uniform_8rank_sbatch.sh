#!/bin/bash
# SLURM test script for one 8-rank segment-correction MPI run,
# followed by the matching serial run and snapshot comparison.
#
# Usage:
#   sbatch run_mpi_fast_heat_uniform_8rank_sbatch.sh

#SBATCH -J segcorr_fh_u8
#SBATCH -N 8
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o logs/segcorr_fh_u8_%j.out
#SBATCH -e logs/segcorr_fh_u8_%j.err
#SBATCH -p gh-dev
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
PATH_CONFIG=${PATH_CONFIG:-configs/fast_heat.ini}
DT_US=${DT_US:-10}
PLANNER_MODE=${PLANNER_MODE:-uniform}
CORRECTION_WEIGHT=${CORRECTION_WEIGHT:-0.25}
PAR_SOLVER_MODE=${PAR_SOLVER_MODE:-fused}
SNAP_EVERY_STEPS=${SNAP_EVERY_STEPS:-100}
N_RANKS=8
MAX_RANKS=${SLURM_NTASKS:-8}
RUN_ROOT=${RUN_ROOT:-outputs/segment_correction_fast_heat_uniform_8rank_${SLURM_JOB_ID:-manual}}

if [ "${N_RANKS}" -gt "${MAX_RANKS}" ]; then
  echo "[error] requested ${N_RANKS} ranks but SLURM_NTASKS=${MAX_RANKS}"
  exit 1
fi

if [ ! -f "${PATH_CONFIG}" ]; then
  echo "[error] path config not found: '${PATH_CONFIG}'"
  exit 1
fi

if [ "${PLANNER_MODE}" != "uniform" ]; then
  echo "[error] this test script is fixed to PLANNER_MODE=uniform"
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

PAR_OUT_DIR="${RUN_ROOT}/parallel_8r"
SER_OUT_DIR="${RUN_ROOT}/serial_8r_plan"
CMP_OUT_DIR="${RUN_ROOT}/comparison_parallel_vs_serial"
mkdir -p "${PAR_OUT_DIR}" "${SER_OUT_DIR}" "${CMP_OUT_DIR}"

echo "======================================================"
echo " Segment Correction 8-Rank Test"
echo "  job_id             : ${SLURM_JOB_ID:-manual}"
echo "  project_dir        : ${PROJECT_DIR}"
echo "  config             : ${CONFIG}"
echo "  path_config        : ${PATH_CONFIG}"
echo "  dt_us              : ${DT_US}"
echo "  planner_mode       : ${PLANNER_MODE}"
echo "  correction_weight  : ${CORRECTION_WEIGHT}"
echo "  parallel solver    : ${PAR_SOLVER_MODE}"
echo "  snap_every_steps   : ${SNAP_EVERY_STEPS}"
echo "  ranks              : ${N_RANKS}"
echo "  run_root           : ${RUN_ROOT}"
echo "======================================================"

echo ""
echo "[$(date)] Parallel snapshot run: path_config=${PATH_CONFIG}, planner=${PLANNER_MODE}, ranks=${N_RANKS}"
mpirun -n "${N_RANKS}" --bind-to none "${WRAPPER}" \
  python src/hermes/scripts/segment_correction/main.py \
    --config "${CONFIG}" \
    --path-config "${PATH_CONFIG}" \
    --dt-us "${DT_US}" \
    --solver-mode "${PAR_SOLVER_MODE}" \
    --planner-mode "${PLANNER_MODE}" \
    --correction-weight "${CORRECTION_WEIGHT}" \
    --no-export-dag \
    --snap-every-steps "${SNAP_EVERY_STEPS}" \
    --out-dir "${PAR_OUT_DIR}"

echo ""
echo "[$(date)] Serial reference run with matching runtime plan."
python src/hermes/scripts/segment_correction/serial_run.py \
  --config "${CONFIG}" \
  --path-config "${PATH_CONFIG}" \
  --dt-us "${DT_US}" \
  --solver-mode "${PAR_SOLVER_MODE}" \
  --world-size "${N_RANKS}" \
  --planner-mode "${PLANNER_MODE}" \
  --correction-weight "${CORRECTION_WEIGHT}" \
  --snap-every-steps "${SNAP_EVERY_STEPS}" \
  --out-dir "${SER_OUT_DIR}"

echo ""
echo "[$(date)] Comparing parallel and serial snapshots."
python src/hermes/scripts/segment_correction/compare_runs.py \
  --par-snap-dir "${PAR_OUT_DIR}/snapshots_par" \
  --ser-snap-dir "${SER_OUT_DIR}/snapshots_ser" \
  --out-dir "${CMP_OUT_DIR}"

echo ""
echo "[$(date)] 8-rank fast_heat uniform validation complete."
echo "  parallel   : ${PAR_OUT_DIR}"
echo "  serial     : ${SER_OUT_DIR}"
echo "  comparison : ${CMP_OUT_DIR}"
