#!/bin/bash
# Gate B: 32/64-rank comparison on one allocation. Bull eps=0.01K.
# Variants: uniform | DP old affine (a=5.5,w=0.25) | DP new affine (a=7.9,w=0.21).

#SBATCH -J gate3264
#SBATCH -N 64
#SBATCH -n 64
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:30:00
#SBATCH -o logs/gate3264_%j.out
#SBATCH -e logs/gate3264_%j.err
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
  local R=$1 MODE=$2 W=$3 A=$4 NAME=$5
  local OUT=outputs/gate3264/r${R}_${NAME}
  if [ -f "${OUT}/timing_summary.json" ]; then
    echo " [$(date)] r${R}/${NAME}: done, skipping"; return
  fi
  echo "###### [$(date)] gate3264 r${R} ${NAME} (w=${W} a=${A}) ######"
  HERMES_CORRECTION_FIXED_COST_SS=${A} \
  srun -N "${R}" -n "${R}" --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config configs/examples/sim_ex1.ini \
      --path-config configs/examples/fast_heat.ini \
      --dt-us 10 --planner-mode "${MODE}" --correction-weight "${W}" \
      --timing-only --no-export-dag \
      --out-dir "${OUT}"
}

for R in 64 32; do
  run_case "${R}" uniform  0.25 7.9 uniform
  run_case "${R}" exact_dp 0.25 5.5 dp_old
  run_case "${R}" exact_dp 0.21 7.9 dp_new
done

echo ""
echo "===== [$(date)] GATE-32/64 summary ====="
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
    vals = {}
    for name in ("uniform", "dp_old", "dp_new"):
        s = stats(f"r{R}_{name}")
        vals[name] = s
        if s: print(f"  {name:8s} wall={s[0]:7.1f}s  busy skew={s[1]:.3f}  max/min={s[2]:.3f}")
    if all(vals.values()):
        u, o, n = vals["uniform"][0], vals["dp_old"][0], vals["dp_new"][0]
        verdict = "PASS" if n <= o*1.02 else "FAIL (new slower)"
        print(f"  new vs old: {o/n:.3f}x   new vs uniform: {u/n:.3f}x   -> {verdict}")
PY
