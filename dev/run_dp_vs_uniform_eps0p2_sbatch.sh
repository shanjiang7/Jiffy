#!/bin/bash
# DP vs uniform at eps = 0.2 K (1e-5 ladder threshold), Bull, 8 ranks, h = 18 um.
# Fills the unmeasured cell between the measured eps=1K (+0.5%) and
# eps=0.01K (+8.7%) DP advantages. Same conditions as outputs/dp_vs_uniform.

#SBATCH -J dp_uni_0p2
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:45:00
#SBATCH -o logs/dp_uni_0p2_%j.out
#SBATCH -e logs/dp_uni_0p2_%j.err
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

for MODE in exact_dp uniform; do
  OUT=outputs/dp_vs_uniform/eps0p2_${MODE}
  if [ -f "${OUT}/timing_summary.json" ]; then
    echo " [$(date)] eps0p2/${MODE}: already done, skipping"
    continue
  fi
  echo "###### [$(date)] Bull 8 ranks, eps=0.2, planner=${MODE} ######"
  srun -N 8 -n 8 --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config configs/examples/sim_ex1.ini \
      --path-config configs/dev/bull_eps0p2.ini \
      --dt-us 10 --planner-mode "${MODE}" --correction-weight 0.25 \
      --timing-only --no-export-dag \
      --out-dir "${OUT}"
done

echo ""
echo "===== [$(date)] DP vs uniform, eps sweep summary ====="
python3 - <<'PY'
import json, pathlib
print(f"{'eps':>8} {'uniform T(8)':>13} {'dp T(8)':>9} {'uniform/dp':>10}")
for tag in ("eps5", "eps1", "eps0p2", "eps0p01"):
    t = {}
    for mode in ("uniform", "exact_dp"):
        f = pathlib.Path(f"outputs/dp_vs_uniform/{tag}_{mode}/timing_summary.json")
        t[mode] = float(json.loads(f.read_text())["parallel_total_seconds"]) if f.exists() else None
    if None in t.values():
        print(f"{tag:>8}  MISSING RUN(S)")
        continue
    print(f"{tag:>8} {t['uniform']:>12.1f}s {t['exact_dp']:>8.1f}s {t['uniform']/t['exact_dp']:>9.3f}x")
PY
