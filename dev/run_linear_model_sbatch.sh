#!/bin/bash
# Linear cost model (w=0.25, no fixed term) on the updated source-charged
# span: DP runs for Bull + Spiral-Raster at 8/32/64 ranks, eps=0.01K.
# Outputs under outputs/linear_model/; uniform baselines reused from gates.

#SBATCH -J lin_model
#SBATCH -N 64
#SBATCH -n 64
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:30:00
#SBATCH -o logs/lin_model_%j.out
#SBATCH -e logs/lin_model_%j.err
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
  local R=$1 CFG=$2 OUT=$3
  if [ -f "${OUT}/timing_summary.json" ]; then
    echo " [$(date)] ${OUT}: done, skipping"; return
  fi
  echo "###### [$(date)] linear R=${R} ${CFG} ######"
  HERMES_CORRECTION_FIXED_COST_SS=0 \
  srun -N "${R}" -n "${R}" --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config configs/examples/sim_ex1.ini \
      --path-config "${CFG}" \
      --dt-us 10 --planner-mode exact_dp --correction-weight 0.25 \
      --timing-only --no-export-dag \
      --out-dir "${OUT}"
}

BULL=configs/examples/fast_heat.ini
HYB=configs/examples/hybrid_spiral_raster.ini

for R in 64 32 8; do
  run_case "${R}" "${BULL}" outputs/linear_model/bull/r${R}_dp_linear
  run_case "${R}" "${HYB}"  outputs/linear_model/hyb/r${R}_dp_linear
done

echo ""
echo "===== [$(date)] linear model (updated span) vs final model ====="
python3 - <<'PY'
import json, pathlib
def stats(p):
    f = pathlib.Path(f"outputs/{p}/timing_summary.json")
    if not f.exists(): return None
    d = json.loads(f.read_text())
    rb = d["rank_timing_breakdown"]
    busy = [v["base_solve_seconds"]+v["tracer_solve_seconds"]+v["local_superpose_seconds"] for v in rb.values()]
    return d["parallel_total_seconds"], max(busy)/(sum(busy)/len(busy))
CASES = [
    ("Bull", "gate3264" , "gate8/eps0p01", "linear_model/bull"),
    ("Hyb ", "gate_hybrid", "gate_hybrid/r8", "linear_model/hyb"),
]
for R in (8, 32, 64):
    for name, groot, r8root, lroot in CASES:
        if R == 8:
            u = stats(f"{r8root}_uniform"); s = stats(f"{r8root}_dp_srcfix")
        else:
            u = stats(f"{groot}/r{R}_uniform"); s = stats(f"{groot}/r{R}_dp_srcfix")
        l = stats(f"{lroot}/r{R}_dp_linear")
        fu = f"{u[0]:6.1f}s" if u else "  --  "
        fl = f"{l[0]:6.1f}s k={l[1]:.3f}" if l else "     --      "
        fs = f"{s[0]:6.1f}s k={s[1]:.3f}" if s else "     --      "
        print(f"{name} R={R:2d}: uniform {fu} | linear {fl} | final {fs}")
PY
