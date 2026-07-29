#!/bin/bash
# Validate the affine correction cost model (fixed 5.5 SS per boundary tracer,
# _grouping.CORRECTION_FIXED_COST_SS): rerun Bull 8-rank exact_dp at three
# thresholds and compare against the measured uniform and linear-model DP
# baselines in outputs/dp_vs_uniform. Success criterion: affine DP <= uniform
# everywhere (in particular eps=5K, where linear DP measured 0.994x).

#SBATCH -J affine_val
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:45:00
#SBATCH -o logs/affine_val_%j.out
#SBATCH -e logs/affine_val_%j.err
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
  local TAG=$1 CFG=$2
  local OUT=outputs/dp_vs_uniform/${TAG}_exact_dp_affine
  if [ -f "${OUT}/timing_summary.json" ]; then
    echo " [$(date)] ${TAG}/affine: already done, skipping"
    return
  fi
  echo "###### [$(date)] Bull 8 ranks, eps=${TAG}, exact_dp + affine model ######"
  srun -N 8 -n 8 --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config configs/examples/sim_ex1.ini \
      --path-config "${CFG}" \
      --dt-us 10 --planner-mode exact_dp --correction-weight 0.25 \
      --timing-only --no-export-dag \
      --out-dir "${OUT}"
}

run_case eps5    configs/dev/bull_eps5.ini
run_case eps0p2  configs/dev/bull_eps0p2.ini
run_case eps0p01 configs/examples/fast_heat.ini

echo ""
echo "===== [$(date)] affine model validation ====="
python3 - <<'PY'
import json, pathlib
def T(p):
    f = pathlib.Path(f"outputs/dp_vs_uniform/{p}/timing_summary.json")
    return float(json.loads(f.read_text())["parallel_total_seconds"]) if f.exists() else None
print(f"{'eps':>8} {'uniform':>9} {'dp linear':>10} {'dp affine':>10} {'affine vs uniform':>18}")
for tag in ("eps5", "eps0p2", "eps0p01"):
    u, dl, da = T(f"{tag}_uniform"), T(f"{tag}_exact_dp"), T(f"{tag}_exact_dp_affine")
    if None in (u, da):
        print(f"{tag:>8}  MISSING"); continue
    verdict = "DP wins/ties" if da <= u * 1.005 else "DP STILL LOSES"
    print(f"{tag:>8} {u:>8.1f}s {dl or 0:>9.1f}s {da:>9.1f}s {u/da:>13.3f}x  {verdict}")
PY
