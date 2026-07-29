#!/bin/bash
# Can the recalibrated model (w=0.21, a0=7.9) beat uniform at eps=5K, 8 ranks?
# The expected effect (<=1%) is below single-run noise (+-1.5%), so this runs
# median-of-3 for both modes on ONE allocation, alternating to cancel drift.

#SBATCH -J gate8e5
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:30:00
#SBATCH -o logs/gate8e5_%j.out
#SBATCH -e logs/gate8e5_%j.err
#SBATCH -p gh-dev
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
  local REP=$1 MODE=$2 W=$3 NAME=$4
  local OUT=outputs/gate8_eps5/rep${REP}_${NAME}
  if [ -f "${OUT}/timing_summary.json" ]; then
    echo " [$(date)] rep${REP}/${NAME}: done, skipping"; return
  fi
  echo "###### [$(date)] eps5 rep${REP} ${NAME} ######"
  srun -N 8 -n 8 --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config configs/examples/sim_ex1.ini \
      --path-config configs/dev/bull_eps5.ini \
      --dt-us 10 --planner-mode "${MODE}" --correction-weight "${W}" \
      --timing-only --no-export-dag \
      --out-dir "${OUT}"
}

# alternate modes within the allocation to cancel slow node/thermal drift
for REP in 1 2 3; do
  run_case "${REP}" uniform  0.25 uniform
  run_case "${REP}" exact_dp 0.21 dp_new
done

echo ""
echo "===== [$(date)] eps=5K median-of-3 verdict ====="
python3 - <<'PY'
import json, pathlib, statistics
def wall(p):
    f = pathlib.Path(f"outputs/gate8_eps5/{p}/timing_summary.json")
    return float(json.loads(f.read_text())["parallel_total_seconds"]) if f.exists() else None
res = {}
for name in ("uniform", "dp_new"):
    walls = [wall(f"rep{r}_{name}") for r in (1, 2, 3)]
    walls = [w for w in walls if w is not None]
    res[name] = walls
    if walls:
        print(f"{name:8s}: runs {['%.1f'%w for w in walls]}  median {statistics.median(walls):.1f}s")
if all(len(v) == 3 for v in res.values()):
    mu, md = statistics.median(res["uniform"]), statistics.median(res["dp_new"])
    spread = max(max(v)-min(v) for v in res.values())
    print(f"median uniform/dp_new = {mu/md:.3f}x   (worst within-mode spread {spread:.1f}s)")
    if md < mu - spread/2: print("VERDICT: DP beats uniform beyond noise")
    elif mu < md - spread/2: print("VERDICT: uniform faster beyond noise")
    else: print("VERDICT: statistical tie (difference within run-to-run spread)")
PY
