#!/bin/bash
# Strong scaling (ranks 2..8) at four calibration-ladder thresholds on Bull,
# h = 18 um, exact_dp, timing-only. One curve per error target:
#   eps = 5 K    -> 1e-4 target   (configs/dev/bull_eps5.ini)
#   eps = 0.2 K  -> 1e-5 target   (configs/dev/bull_eps0p2.ini)
#   eps = 0.03 K -> 1e-6 target   (configs/dev/bull_eps0p03.ini)
#   eps = 0.01 K -> 1e-7 target   (configs/examples/fast_heat.ini default)
# The 1-rank baseline is epsilon-independent (no cuts -> no corrections):
# reuse outputs/eps_probe/bull_1r (841.5 s).
# Resumable: any (eps, R) with an existing timing_summary.json is skipped.

#SBATCH -J eps_scaling
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 03:00:00
#SBATCH -o logs/eps_scaling_%j.out
#SBATCH -e logs/eps_scaling_%j.err
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

ROOT=outputs/eps_strong_scaling

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

# Loosest first (cheap runs surface script problems early); within each eps,
# descending R so the expensive 2-rank runs are last and resumable.
for SPEC in "eps5:configs/dev/bull_eps5.ini" \
            "eps0p2:configs/dev/bull_eps0p2.ini" \
            "eps0p03:configs/dev/bull_eps0p03.ini" \
            "eps0p01:configs/examples/fast_heat.ini"; do
  TAG=${SPEC%%:*}; CFG=${SPEC#*:}
  for R in 8 7 6 5 4 3 2; do
    run_case "${TAG}" "${CFG}" "${R}"
  done
done

echo ""
echo "===== [$(date)] eps strong-scaling summary ====="
python3 - <<'PY'
import json, pathlib
T1 = json.loads(pathlib.Path("outputs/eps_probe/bull_1r/timing_summary.json").read_text())["parallel_total_seconds"]
print(f"1-rank baseline (eps-independent): {T1:.1f}s")
print(f"{'eps':>8} " + " ".join(f"{f'R={r}':>9}" for r in range(2, 9)))
for tag in ("eps5", "eps0p2", "eps0p03", "eps0p01"):
    cells = []
    for r in range(2, 9):
        f = pathlib.Path(f"outputs/eps_strong_scaling/{tag}_r{r}/timing_summary.json")
        cells.append(f"{T1/json.loads(f.read_text())['parallel_total_seconds']:>8.2f}x" if f.exists() else f"{'--':>9}")
    print(f"{tag:>8} " + " ".join(cells))
PY
