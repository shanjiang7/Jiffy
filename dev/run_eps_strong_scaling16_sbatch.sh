#!/bin/bash
# Strong scaling to 16 ranks (one GPU each) on Bull, h = 18 um, exact_dp,
# timing-only. Sampled at ranks 2,4,...,16 for three thresholds:
#   eps = 5 K    -> 1e-4 target   (configs/dev/bull_eps5.ini)
#   eps = 0.2 K  -> 1e-5 target   (configs/dev/bull_eps0p2.ini)
#   eps = 0.01 K -> 1e-7 target   (configs/examples/fast_heat.ini default)
# 1-rank baseline reused from outputs/eps_probe/bull_1r (eps-independent).
# Resumable: any (eps, R) with an existing timing_summary.json is skipped.

#SBATCH -J eps_scale16
#SBATCH -N 16
#SBATCH -n 16
#SBATCH --ntasks-per-node=1
#SBATCH -t 03:00:00
#SBATCH -o logs/eps_scale16_%j.out
#SBATCH -e logs/eps_scale16_%j.err
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

ROOT=outputs/eps_strong_scaling16

run_case () {
  local TAG=$1 CFG=$2 R=$3
  local OUT=${ROOT}/${TAG}_r${R}
  if [ -f "${OUT}/timing_summary.json" ]; then
    echo " [$(date)] ${TAG} R=${R}: already done, skipping"
    return
  fi
  echo ""
  echo "###### [$(date)] Bull eps=${TAG} ranks=${R} ######"
  srun -N "${R}" -n "${R}" --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config configs/examples/sim_ex1.ini \
      --path-config "${CFG}" \
      --dt-us 10 --planner-mode exact_dp --correction-weight 0.25 \
      --timing-only --no-export-dag \
      --out-dir "${OUT}"
}

for SPEC in "eps5:configs/dev/bull_eps5.ini" \
            "eps0p2:configs/dev/bull_eps0p2.ini" \
            "eps0p01:configs/examples/fast_heat.ini"; do
  TAG=${SPEC%%:*}; CFG=${SPEC#*:}
  for R in 16 14 12 10 8 6 4 2; do
    run_case "${TAG}" "${CFG}" "${R}"
  done
done

echo ""
echo "===== [$(date)] 16-rank eps strong-scaling summary ====="
python3 - <<'PY'
import json, pathlib
T1 = json.loads(pathlib.Path("outputs/eps_probe/bull_1r/timing_summary.json").read_text())["parallel_total_seconds"]
print(f"1-rank baseline (eps-independent): {T1:.1f}s")
ranks = list(range(2, 17, 2))
print(f"{'eps':>8} " + " ".join(f"{f'R={r}':>9}" for r in ranks))
for tag in ("eps5", "eps0p2", "eps0p01"):
    cells = []
    for r in ranks:
        f = pathlib.Path(f"outputs/eps_strong_scaling16/{tag}_r{r}/timing_summary.json")
        cells.append(f"{T1/json.loads(f.read_text())['parallel_total_seconds']:>8.2f}x" if f.exists() else f"{'--':>9}")
    print(f"{tag:>8} " + " ".join(cells))
PY
