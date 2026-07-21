#!/bin/bash
# Measure the source-off/source-on per-step cost ratio (= correction_weight of
# the partitioner's workload model) on both grids used by the paper.
#
# Usage:  sbatch dev/run_bench_source_on_off_sbatch.sh

#SBATCH -J bench_onoff
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o logs/bench_onoff_%j.out
#SBATCH -e logs/bench_onoff_%j.err
#SBATCH -p gh-dev
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
mkdir -p logs

source "${PROJECT_DIR}/env_vista.sh"

for CFG in configs/examples/sim_ex1.ini configs/examples/sim_calibration.ini configs/sim_ex3.ini; do
  [ -f "${CFG}" ] || continue
  echo ""
  echo "######################################################"
  srun -N 1 -n 1 python dev/bench_source_on_off.py --config "${CFG}" --steps 1000 --repeats 3
done
