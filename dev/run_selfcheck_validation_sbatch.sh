#!/bin/bash
# Self-convergence ladder study at 31 cuts (32 ranks), tol1e4, hybrid + Bull:
# 6 refinement iterations (level_K/2^k, horizons +1 SS per rung), saving every
# iterate, then comparing each against the serial ground truth.
#
# Final table per path and iteration k:
#   estimate_k  = self-check shift at iteration k  (predicts error of iterate k-1)
#   truth_{k-1} = rel-L2 of iterate k-1 vs the serial reference
# The estimator is validated if estimate_k tracks truth_{k-1} down the ladder,
# and the ladder itself demonstrates error CONTROL: truth should fall below the
# 1e-4 target within a few iterations.
#
# Usage:  sbatch dev/run_selfcheck_validation_sbatch.sh

#SBATCH -J selfcheck_val
#SBATCH -N 8
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH -t 2:00:00
#SBATCH -o logs/selfcheck_val_%j.out
#SBATCH -e logs/selfcheck_val_%j.err
#SBATCH -p gh-dev
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

SIM_CONFIG=configs/examples/sim_calibration.ini
SNAP_EVERY=25
ITERS=6

for P in hybrid bull; do
  if [ "$P" = "hybrid" ]; then
    CFG=configs/accuracy/hybrid_spiral_raster_tol1e4.ini
  else
    CFG=configs/accuracy/bull_tol1e4.ini
  fi
  ROOT=outputs/accuracy_${P}_tol1e4_h30
  if [ ! -d "${ROOT}/serial/snapshots_ser" ]; then
    echo "ERROR: serial reference ${ROOT}/serial missing" >&2
    exit 1
  fi

  echo "======================================================"
  echo " [$(date)] ${P}/tol1e4: 32 ranks (31 cuts), self-check ladder x${ITERS}"
  echo "======================================================"
  srun -N 8 -n 32 --ntasks-per-node=4 python src/hermes/scripts/segment_correction/main.py \
    --config "${SIM_CONFIG}" --path-config "${CFG}" \
    --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
    --planner-mode exact_dp --no-export-dag \
    --self-check --self-check-gamma 2.0 \
    --self-check-iters "${ITERS}" --self-check-save-iters \
    --out-dir "${ROOT}/par32_ladder"

  echo "------ [$(date)] ${P}: truth comparisons (production + each iterate) ------"
  for K in "" $(seq -f "_iter%g" 1 ${ITERS}); do
    srun -N 1 -n 1 --ntasks-per-node=1 python src/hermes/scripts/segment_correction/compare_runs.py \
      --par-snap-dir "${ROOT}/par32_ladder/snapshots_par${K}" \
      --ser-snap-dir "${ROOT}/serial/snapshots_ser" \
      --out-dir "${ROOT}/compare_par32_ladder${K}" \
      --source-on-only \
      --config "${SIM_CONFIG}" --path-config "${CFG}" --dt-us 10 | tail -3
    if [ -f "${ROOT}/compare_par32_ladder${K}/comparison_summary.json" ]; then
      rm -rf "${ROOT}/par32_ladder/snapshots_par${K}" "${ROOT}/par32_ladder/snapshots_par${K}_meta"
    fi
  done
done

echo ""
echo "[$(date)] ===== self-convergence estimate vs true rel-L2 (31 cuts, tol1e4, gamma=2) ====="
L=$(ls -t logs/selfcheck_val_*.out | head -1)
python3 - "$L" "${ITERS}" <<'PYEOF'
import json, re, sys

log = open(sys.argv[1]).read()
iters = int(sys.argv[2])
# per-path estimate lists, in run order (hybrid then bull)
blocks = re.findall(
    r"\[self-check\] iter (\d+): estimated rel-L2 of previous iterate.*?max=([0-9.e+-]+)\s+rms=([0-9.e+-]+)",
    log,
)
paths = ["hybrid", "bull"]
per_path = {p: blocks[i * iters : (i + 1) * iters] for i, p in enumerate(paths)}

def truth(p, suffix):
    try:
        d = json.load(open(f"outputs/accuracy_{p}_tol1e4_h30/compare_par32_ladder{suffix}/comparison_summary.json"))
        return f"{d['max_rel_l2']:.4e}"
    except Exception:
        return "n/a"

for p in paths:
    print(f"\n{p} (target 1e-4):")
    print(f"  {'iterate':<10} {'true max rel-L2':>16} {'self-check estimate':>20}")
    print(f"  {'u_0 (prod)':<10} {truth(p, ''):>16} {per_path[p][0][1] if per_path[p] else 'n/a':>20}")
    for k in range(1, iters + 1):
        est = per_path[p][k][1] if k < len(per_path[p]) else "-"
        print(f"  {'u_' + str(k):<10} {truth(p, f'_iter{k}'):>16} {est:>20}")
PYEOF
