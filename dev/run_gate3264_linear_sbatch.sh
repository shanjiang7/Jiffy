#!/bin/bash
# Gate B addendum: linear cost model (w=0.25, NO fixed term — the submitted
# paper's model, via HERMES_CORRECTION_FIXED_COST_SS=0) at 32/64 ranks,
# Bull eps=0.01K. Completes the ladder: uniform | linear | affine old | new.

#SBATCH -J gate_lin
#SBATCH -N 64
#SBATCH -n 64
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:45:00
#SBATCH -o logs/gate_lin_%j.out
#SBATCH -e logs/gate_lin_%j.err
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

for R in 64 32; do
  OUT=outputs/gate3264/r${R}_dp_linear
  if [ -f "${OUT}/timing_summary.json" ]; then
    echo " [$(date)] r${R}/dp_linear: done, skipping"; continue
  fi
  echo "###### [$(date)] gate3264 r${R} dp_linear (w=0.25 a=0) ######"
  HERMES_CORRECTION_FIXED_COST_SS=0 \
  srun -N "${R}" -n "${R}" --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config configs/examples/sim_ex1.ini \
      --path-config configs/examples/fast_heat.ini \
      --dt-us 10 --planner-mode exact_dp --correction-weight 0.25 \
      --timing-only --no-export-dag \
      --out-dir "${OUT}"
done

echo ""
echo "===== [$(date)] full model ladder (Bull eps=0.01) ====="
python3 - <<'PY'
import json, pathlib
def stats(p):
    f = pathlib.Path(f"outputs/gate3264/{p}/timing_summary.json")
    if not f.exists(): return None
    d = json.loads(f.read_text())
    rb = d["rank_timing_breakdown"]
    busy = [v["base_solve_seconds"]+v["tracer_solve_seconds"]+v["local_superpose_seconds"] for v in rb.values()]
    return d["parallel_total_seconds"], max(busy)/(sum(busy)/len(busy)), max(busy)/min(busy)
for R in (32, 64):
    print(f"\nR={R}:")
    for name in ("uniform", "dp_linear", "dp_old", "dp_new"):
        s = stats(f"r{R}_{name}")
        if s: print(f"  {name:10s} wall={s[0]:7.1f}s  busy skew={s[1]:.3f}  max/min={s[2]:.3f}")
        else: print(f"  {name:10s} (pending)")
PY
