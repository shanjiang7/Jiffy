#!/bin/bash
# DP (exact_dp) vs uniform partitioning, measured head-to-head on Bull, 8 ranks,
# h = 18 um, weight 0.25 -- at both ends of the epsilon range.
#
# Planner prediction (validated cost model, dev scan 2026-07-21):
#   eps = 5 K   : uniform/dp max-load ratio 1.009  (DP ~0.9% faster)
#   eps = 0.01 K: uniform/dp max-load ratio 1.117  (DP ~11.7% faster)
# Tight epsilon is where corrections are large enough for boundary placement to
# matter; loose epsilon makes corrections negligible so DP ~ uniform.
# No 1-rank baseline needed: the comparison is T_uniform(8) vs T_dp(8).

#SBATCH -J dp_vs_uni
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -o logs/dp_vs_uni_%j.out
#SBATCH -e logs/dp_vs_uni_%j.err
#SBATCH -p gh
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
mkdir -p logs outputs
source "${PROJECT_DIR}/env_vista.sh"

run_case () {
  local TAG=$1 CFG=$2 MODE=$3
  local OUT=outputs/dp_vs_uniform/${TAG}_${MODE}
  if [ -f "${OUT}/timing_summary.json" ]; then
    echo " [$(date)] ${TAG}/${MODE}: already done, skipping"
    return
  fi
  echo ""
  echo "###### [$(date)] Bull 8 ranks, eps=${TAG}, planner=${MODE} ######"
  srun -N 8 -n 8 --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config configs/examples/sim_ex1.ini \
      --path-config "${CFG}" \
      --dt-us 10 --planner-mode "${MODE}" --correction-weight 0.25 \
      --timing-only --no-export-dag \
      --out-dir "${OUT}"
}

run_case eps0p01 configs/examples/fast_heat.ini exact_dp
run_case eps0p01 configs/examples/fast_heat.ini uniform
run_case eps5    configs/dev/bull_eps5.ini      exact_dp
run_case eps5    configs/dev/bull_eps5.ini      uniform

echo ""
echo "===== [$(date)] DP vs uniform, measured ====="
python3 - <<'PY'
import json, pathlib
PRED = {"eps0p01": 1.117, "eps5": 1.009}
print(f"{'eps':>8} {'uniform T(8)':>13} {'dp T(8)':>9} {'measured':>9} {'predicted':>10}")
for tag in ("eps5", "eps0p01"):
    t = {}
    for mode in ("uniform", "exact_dp"):
        f = pathlib.Path(f"outputs/dp_vs_uniform/{tag}_{mode}/timing_summary.json")
        t[mode] = float(json.loads(f.read_text())["parallel_total_seconds"]) if f.exists() else None
    if None in t.values():
        print(f"{tag:>8}  MISSING RUN(S)")
        continue
    ratio = t["uniform"] / t["exact_dp"]
    print(f"{tag:>8} {t['uniform']:>12.1f}s {t['exact_dp']:>8.1f}s {ratio:>8.3f}x {PRED[tag]:>9.3f}x")
print()
print("(ratio = uniform/dp makespan; >1 means DP is faster)")
PY
