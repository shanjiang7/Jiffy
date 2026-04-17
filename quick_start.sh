ml gcc
ml openmpi
ml cuda
export CUDA_HOME=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/12.5/targets/sbsa-linux
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export NUMBA_CUDA_DRIVER=/usr/lib64/libcuda.so
export CUDA_HOME=$TACC_CUDA_DIR
export LD_LIBRARY_PATH=/usr/lib64/:$LD_LIBRARY_PATH

export PYTHONPATH=/scratch/10226/shawnraul/Jiffy/src:$PYTHONPATH