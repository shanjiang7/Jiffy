#!/bin/bash
# Rank-breakdown comparison on the Spiral-Raster path: uniform vs DP with the
# recalibrated model (w=0.21, a0=7.9), eps=0.01K, R in {8,32,64}, one GPU per
# rank, all on one allocation. Mirrors the Bull gate3264 experiment.

#SBATCH -J gate_hyb
#SBATCH -N 64
#SBATCH -n 64
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:30:00
#SBATCH -o logs/gate_hyb_%j.out
#SBATCH -e logs/gate_hyb_%j.err
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

run_case () {
  local R=$1 MODE=$2 W=$3 NAME=$4
  local OUT=outputs/gate_hybrid/r${R}_${NAME}
  if [ -f "${OUT}/timing_summary.json" ]; then
    echo " [$(date)] r${R}/${NAME}: done, skipping"; return
  fi
  echo "###### [$(date)] hybrid r${R} ${NAME} ######"
  srun -N "${R}" -n "${R}" --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config configs/examples/sim_ex1.ini \
      --path-config configs/examples/hybrid_spiral_raster.ini \
      --dt-us 10 --planner-mode "${MODE}" --correction-weight "${W}" \
      --timing-only --no-export-dag \
      --out-dir "${OUT}"
}

for R in 64 32 8; do
  run_case "${R}" uniform  0.25 uniform
  run_case "${R}" exact_dp 0.21 dp_new
done

echo ""
echo "===== [$(date)] Spiral-Raster DP vs uniform summary ====="
python3 - <<'PY'
import json, pathlib
def stats(p):
    f = pathlib.Path(f"outputs/gate_hybrid/{p}/timing_summary.json")
    if not f.exists(): return None
    d = json.loads(f.read_text())
    rb = d["rank_timing_breakdown"]
    busy = [v["base_solve_seconds"]+v["tracer_solve_seconds"]+v["local_superpose_seconds"] for v in rb.values()]
    return d["parallel_total_seconds"], max(busy)/(sum(busy)/len(busy)), max(busy)/min(busy)
for R in (8, 32, 64):
    print(f"\nR={R}:")
    vals = {}
    for name in ("uniform", "dp_new"):
        s = stats(f"r{R}_{name}")
        vals[name] = s
        if s: print(f"  {name:8s} wall={s[0]:7.1f}s  busy skew={s[1]:.3f}  max/min={s[2]:.3f}")
    if all(vals.values()):
        print(f"  uniform/dp = {vals['uniform'][0]/vals['dp_new'][0]:.3f}x")
PY
