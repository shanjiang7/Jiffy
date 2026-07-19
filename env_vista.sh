# Environment setup for JIFFY on TACC Vista (GH200 nodes).
# Usage: source env_vista.sh
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

export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" && pwd)/src:${PYTHONPATH:-}"
