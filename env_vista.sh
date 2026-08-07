# Environment setup for JIFFY on TACC Vista (GH200 nodes).
# Usage: source env_vista.sh
module purge
module load gcc/13.2.0
module load openmpi/5.0.5
module load cuda/12.5

export CUDA_HOME=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/12.5/targets/sbsa-linux
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export NUMBA_CUDA_DRIVER=/usr/lib64/libcuda.so
export CUDA_HOME="${TACC_CUDA_DIR:-${CUDA_HOME}}"
export LD_LIBRARY_PATH="/usr/lib64/:${LD_LIBRARY_PATH:-}"

# Activate the hermes conda environment. Set CONDA_ROOT to your conda
# installation; on TACC it defaults to $WORK/anaconda3.
CONDA_ROOT="${CONDA_ROOT:-${WORK:-$HOME}/anaconda3}"
if [ ! -f "${CONDA_ROOT}/bin/activate" ]; then
  echo "env_vista.sh: no conda at CONDA_ROOT=${CONDA_ROOT} — export CONDA_ROOT=/path/to/your/conda (e.g. \$HOME/miniforge3)" >&2
  return 1 2>/dev/null || exit 1
fi
source "${CONDA_ROOT}/bin/activate"
conda activate hermes

export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" && pwd)/src:${PYTHONPATH:-}"
