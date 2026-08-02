#!/bin/bash
# Render the melt-pool prototype scenes with ParaView (headless pvbatch).
# Inputs are pre-exported by dev/proto_melt_pools.py into
# outputs/proto_melt_pools/ (rank_*.vtk, path.vtk, ranks.json).
#
# Usage:  sbatch dev/run_render_melt_pools_sbatch.sh

#SBATCH -J pv_pools
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -o logs/pv_pools_%j.out
#SBATCH -e logs/pv_pools_%j.err
#SBATCH -p gh
#SBATCH -A ASC21034

set -uo pipefail
PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
while [ ! -f "${PROJECT_DIR}/env_vista.sh" ] && [ "${PROJECT_DIR}" != "/" ]; do
  PROJECT_DIR=$(dirname "${PROJECT_DIR}")
done
cd "${PROJECT_DIR}" || exit 1

module purge
module load gcc/15.1.0 cuda/13.0 openmpi/5.0.9 paraview_osmesa/5.13.3

OMP_NUM_THREADS=1 pvbatch dev/render_melt_pools_pv.py
