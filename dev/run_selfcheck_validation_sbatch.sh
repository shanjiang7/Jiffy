#!/bin/bash
# Validate the --self-check error estimator at 31 cuts (32 ranks), where the
# plain tol1e4 configuration EXCEEDS its 1e-4 target:
#   hybrid tol1e4 par32: true error ~3.9e-4 (clear exceedance)
#   bull   tol1e4 par32: true error ~1.1e-4 (marginal exceedance)
# The job produces a table of the self-convergence estimate vs the reference
# rel-L2 (compared against the serial ground truth) for both paths.
#
# Smoke pre-validation (2-rank straight line): estimate 4.2800e-5 vs true
# 4.2800e-5, digit-exact.
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
  echo " [$(date)] ${P}/tol1e4: 32 ranks (31 cuts) with --self-check"
  echo "======================================================"
  srun -N 8 -n 32 --ntasks-per-node=4 python src/hermes/scripts/segment_correction/main.py \
    --config "${SIM_CONFIG}" --path-config "${CFG}" \
    --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
    --planner-mode exact_dp --no-export-dag \
    --self-check --self-check-gamma 2.0 \
    --out-dir "${ROOT}/par32_selfcheck"

  echo "------ [$(date)] ${P}: reference rel-L2 vs serial ground truth ------"
  srun -N 1 -n 1 --ntasks-per-node=1 python src/hermes/scripts/segment_correction/compare_runs.py \
    --par-snap-dir "${ROOT}/par32_selfcheck/snapshots_par" \
    --ser-snap-dir "${ROOT}/serial/snapshots_ser" \
    --out-dir "${ROOT}/compare_par32_selfcheck" \
    --source-on-only \
    --config "${SIM_CONFIG}" --path-config "${CFG}" --dt-us 10 | tail -6

  if [ -f "${ROOT}/compare_par32_selfcheck/comparison_summary.json" ]; then
    rm -rf "${ROOT}/par32_selfcheck/snapshots_par" "${ROOT}/par32_selfcheck/snapshots_par_meta"
  fi
done

echo ""
echo "[$(date)] ===== self-convergence estimate vs serial reference (31 cuts, tol1e4) ====="
L=$(ls -t logs/selfcheck_val_*.out | head -1)
python3 - "$L" <<'PYEOF'
import json, re, sys

log = open(sys.argv[1]).read()
estimates = re.findall(
    r"\[self-check\] estimated parallel-vs-serial rel-L2.*?max=([0-9.e+-]+)\s+rms=([0-9.e+-]+)", log
)
paths = ["hybrid", "bull"]
print(f"{'path':<8} {'self-check max':>15} {'self-check rms':>15} {'reference max':>15} {'reference mean':>15}")
for i, p in enumerate(paths):
    est_max, est_rms = (estimates[i] if i < len(estimates) else ("n/a", "n/a"))
    try:
        d = json.load(open(f"outputs/accuracy_{p}_tol1e4_h30/compare_par32_selfcheck/comparison_summary.json"))
        ref_max, ref_mean = f"{d['max_rel_l2']:.4e}", f"{d['mean_rel_l2']:.4e}"
    except Exception:
        ref_max = ref_mean = "n/a"
    print(f"{p:<8} {est_max:>15} {est_rms:>15} {ref_max:>15} {ref_mean:>15}")
PYEOF
