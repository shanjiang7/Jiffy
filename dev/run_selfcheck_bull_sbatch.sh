#!/bin/bash
# Self-convergence ladder study (horizon-only mode) on the bull path:
# 32 ranks (31 cuts), tol1e4, 6 refinement iterations, each extending every
# connected pair's correction window by one supersegment. No deep DAGs or
# lookup builds - the cheap production estimator. Saves every iterate and
# compares each against the serial ground truth; ends with the
# estimate-vs-truth table.
#
# Basis for horizon-only: in the full-mode study (job 844825) the new-pairs
# channel contributed only ~1e-13 on every rung of both paths - the measured
# error is truncated correction tails, which this mode refines directly.
#
# Usage:  sbatch dev/run_selfcheck_bull_sbatch.sh

#SBATCH -J selfchk_bull
#SBATCH -N 8
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH -t 2:00:00
#SBATCH -o logs/selfchk_bull_%j.out
#SBATCH -e logs/selfchk_bull_%j.err
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
CFG=configs/accuracy/bull_tol1e4.ini
ROOT=outputs/accuracy_bull_tol1e4_h30
SNAP_EVERY=25
ITERS=6

if [ ! -d "${ROOT}/serial/snapshots_ser" ]; then
  echo "ERROR: serial reference ${ROOT}/serial missing" >&2
  exit 1
fi

echo "====== [$(date)] bull/tol1e4: 32 ranks, horizon-only self-check ladder x${ITERS} ======"
srun -N 8 -n 32 --ntasks-per-node=4 python src/hermes/scripts/segment_correction/main.py \
  --config "${SIM_CONFIG}" --path-config "${CFG}" \
  --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
  --planner-mode exact_dp --no-export-dag \
  --self-check --self-check-mode horizon \
  --self-check-iters "${ITERS}" --self-check-save-iters \
  --out-dir "${ROOT}/par32_ladder"

echo "------ [$(date)] bull: truth comparisons (production + each iterate) ------"
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

echo ""
echo "[$(date)] ===== bull: estimate vs true rel-L2 (31 cuts, tol1e4, horizon-only) ====="
L=$(ls -t logs/selfchk_bull_*.out | head -1)
python3 - "$L" "${ITERS}" <<'PYEOF'
import json, re, sys

log = open(sys.argv[1]).read()
iters = int(sys.argv[2])
ests = re.findall(
    r"\[self-check\] iter (\d+): estimated rel-L2 of previous iterate.*?max=([0-9.e+-]+)\s+rms=([0-9.e+-]+)", log
)

def truth(suffix):
    try:
        d = json.load(open(f"outputs/accuracy_bull_tol1e4_h30/compare_par32_ladder{suffix}/comparison_summary.json"))
        return f"{d['max_rel_l2']:.4e}"
    except Exception:
        return "n/a"

print(f"  {'iterate':<10} {'true max rel-L2':>16} {'self-check estimate':>20}")
print(f"  {'u_0 (prod)':<10} {truth(''):>16} {ests[0][1] if ests else 'n/a':>20}")
for k in range(1, iters + 1):
    est = ests[k][1] if k < len(ests) else "-"
    print(f"  {'u_' + str(k):<10} {truth(f'_iter{k}'):>16} {est:>20}")
PYEOF
