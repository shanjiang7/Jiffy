#!/bin/bash
# Per-rank breakdown, DP (affine model) vs uniform, Bull eps=0.01K (1e-7
# regime, same config as the 8-rank figC pair), at 32 and 64 ranks -- one GPU
# per rank. All four runs share one allocation so node-draw offsets cancel.
# Resumable: existing timing_summary.json points are skipped.

#SBATCH -J brk3264
#SBATCH -N 64
#SBATCH -n 64
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:30:00
#SBATCH -o logs/brk3264_%j.out
#SBATCH -e logs/brk3264_%j.err
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
  local R=$1 MODE=$2
  local OUT=outputs/breakdown_scaling/r${R}_${MODE}
  if [ -f "${OUT}/timing_summary.json" ]; then
    echo " [$(date)] r${R}/${MODE}: already done, skipping"
    return
  fi
  echo "###### [$(date)] Bull eps=0.01, ranks=${R}, planner=${MODE} ######"
  srun -N "${R}" -n "${R}" --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config configs/examples/sim_ex1.ini \
      --path-config configs/examples/fast_heat.ini \
      --dt-us 10 --planner-mode "${MODE}" --correction-weight 0.25 \
      --timing-only --no-export-dag \
      --out-dir "${OUT}"
}

for R in 64 32; do
  run_case "${R}" uniform
  run_case "${R}" exact_dp
done

echo ""
echo "===== [$(date)] breakdown scaling summary (Bull eps=0.01) ====="
python3 - <<'PY'
import json, pathlib
print(f"{'ranks':>6} {'mode':>9} {'wall(s)':>8} {'busy max':>9} {'busy min':>9} {'busy max/min':>12}")
for R in (32, 64):
    for mode in ("uniform", "exact_dp"):
        f = pathlib.Path(f"outputs/breakdown_scaling/r{R}_{mode}/timing_summary.json")
        if not f.exists():
            print(f"{R:>6} {mode:>9}  MISSING"); continue
        d = json.loads(f.read_text())
        rb = d["rank_timing_breakdown"]
        busy = [v["base_solve_seconds"] + v["tracer_solve_seconds"] + v["local_superpose_seconds"]
                for v in rb.values()]
        print(f"{R:>6} {mode:>9} {d['parallel_total_seconds']:>8.1f} "
              f"{max(busy):>9.1f} {min(busy):>9.1f} {max(busy)/min(busy):>12.3f}")
PY
