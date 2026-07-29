#!/bin/bash
# Gate A: 8-rank model ladder on one allocation. Bull, eps in {0.2, 0.01} K.
# Variants per eps: uniform | DP old affine (a=5.5, w=0.25) | DP new affine
# (a=7.9, w=0.21, the pooled-fit constants). Pass: new >= old (within noise).

#SBATCH -J gate8
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -o logs/gate8_%j.out
#SBATCH -e logs/gate8_%j.err
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
  local TAG=$1 CFG=$2 MODE=$3 W=$4 A=$5 NAME=$6
  local OUT=outputs/gate8/${TAG}_${NAME}
  if [ -f "${OUT}/timing_summary.json" ]; then
    echo " [$(date)] ${TAG}/${NAME}: done, skipping"; return
  fi
  echo "###### [$(date)] gate8 ${TAG} ${NAME} (w=${W} a=${A}) ######"
  HERMES_CORRECTION_FIXED_COST_SS=${A} \
  srun -N 8 -n 8 --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config configs/examples/sim_ex1.ini \
      --path-config "${CFG}" \
      --dt-us 10 --planner-mode "${MODE}" --correction-weight "${W}" \
      --timing-only --no-export-dag \
      --out-dir "${OUT}"
}

for SPEC in "eps0p2:configs/dev/bull_eps0p2.ini" "eps0p01:configs/examples/fast_heat.ini"; do
  TAG=${SPEC%%:*}; CFG=${SPEC#*:}
  run_case "${TAG}" "${CFG}" uniform  0.25 7.9 uniform
  run_case "${TAG}" "${CFG}" exact_dp 0.25 5.5 dp_old
  run_case "${TAG}" "${CFG}" exact_dp 0.21 7.9 dp_new
done

echo ""
echo "===== [$(date)] GATE-8 summary ====="
python3 - <<'PY'
import json, pathlib
def stats(p):
    f = pathlib.Path(f"outputs/gate8/{p}/timing_summary.json")
    if not f.exists(): return None
    d = json.loads(f.read_text())
    rb = d["rank_timing_breakdown"]
    busy = [v["base_solve_seconds"]+v["tracer_solve_seconds"]+v["local_superpose_seconds"] for v in rb.values()]
    return d["parallel_total_seconds"], max(busy)/(sum(busy)/len(busy)), max(busy)/min(busy)
for tag in ("eps0p2", "eps0p01"):
    print(f"\n{tag}:")
    vals = {}
    for name in ("uniform", "dp_old", "dp_new"):
        s = stats(f"{tag}_{name}")
        vals[name] = s
        if s: print(f"  {name:8s} wall={s[0]:7.1f}s  busy skew={s[1]:.3f}  max/min={s[2]:.3f}")
    if all(vals.values()):
        u, o, n = vals["uniform"][0], vals["dp_old"][0], vals["dp_new"][0]
        verdict = "PASS (new<=old within 2%)" if n <= o*1.02 else "FAIL (new slower than old)"
        print(f"  new vs old: {o/n:.3f}x   new vs uniform: {u/n:.3f}x   -> {verdict}")
PY
