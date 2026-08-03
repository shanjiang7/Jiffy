#!/bin/bash
# Smoke validation of the tracer-core refactor (per-edge fused tracer as the
# only production logic). The refactor must be behavior-preserving, so two
# 32-rank runs are compared digit-for-digit against the pre-refactor w0.25
# results (job 855003):
#   bull/tol1e4  : expect max rel-L2 1.4062e-04  (shallow DAG, raster path)
#   hybrid/tol1e7: expect max rel-L2 2.2162e-08  (deep DAG, spiral path)
# Also runs dev/check_fused_bridge.py (CPU planning invariants; needs a GPU
# node only because the package imports the CUDA driver).
#
# Usage:  sbatch dev/run_refactor_smoke_sbatch.sh

#SBATCH -J refactor_smoke
#SBATCH -N 8
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH -t 1:00:00
#SBATCH -o logs/refactor_smoke_%j.out
#SBATCH -e logs/refactor_smoke_%j.err
#SBATCH -p gh
#SBATCH -A ASC21034

set -uo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scratch/10226/shawnraul/work/Jiffy}
cd "${PROJECT_DIR}"
mkdir -p logs outputs

source "${PROJECT_DIR}/env_vista.sh"

SIM_CONFIG=configs/examples/sim_calibration.ini
SNAP_EVERY=25

echo "====== [$(date)] fused-bridge planning invariants ======"
srun -N 1 -n 1 --ntasks-per-node=1 python dev/check_fused_bridge.py

run_case () {
  local PATHNAME=$1 TOL=$2 EXPECT=$3
  local ROOT=outputs/accuracy_${PATHNAME}_${TOL}_h30
  local CFG
  if [ "${PATHNAME}" = "bull" ]; then
    CFG=configs/accuracy/bull_${TOL}.ini
  else
    CFG=configs/accuracy/hybrid_spiral_raster_${TOL}.ini
  fi

  echo "====== [$(date)] ${PATHNAME}/${TOL}: 32 ranks / 31 cuts, w0.25, refactored core ======"
  srun -N 8 -n 32 --ntasks-per-node=4 python src/hermes/scripts/segment_correction/main.py \
    --config "${SIM_CONFIG}" --path-config "${CFG}" \
    --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
    --planner-mode exact_dp --correction-weight 0.25 --no-export-dag \
    --out-dir "${ROOT}/par32_refactor"

  srun -N 1 -n 1 --ntasks-per-node=1 python src/hermes/scripts/segment_correction/compare_runs.py \
    --par-snap-dir "${ROOT}/par32_refactor/snapshots_par" \
    --ser-snap-dir "${ROOT}/serial/snapshots_ser" \
    --out-dir "${ROOT}/compare_par32_refactor" \
    --source-on-only \
    --config "${SIM_CONFIG}" --path-config "${CFG}" --dt-us 10 | tail -4

  if [ -f "${ROOT}/compare_par32_refactor/comparison_summary.json" ]; then
    rm -rf "${ROOT}/par32_refactor/snapshots_par" "${ROOT}/par32_refactor/snapshots_par_meta"
  fi

  python3 - "${PATHNAME}" "${TOL}" "${EXPECT}" <<'PY'
import json, sys
path, tol, expect = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = json.load(open(f"outputs/accuracy_{path}_{tol}_h30/compare_par32_refactor/comparison_summary.json"))
got = float(d["max_rel_l2"])
ok = abs(got - expect) <= 1e-4 * abs(expect)
print(f"RESULT {path}/{tol}: refactored max={got:.4e}  pre-refactor {expect:.4e}  "
      f"{'MATCH' if ok else 'MISMATCH'}")
PY
}

run_case bull tol1e4 1.4062e-4
run_case hybrid tol1e7 2.2162e-8
echo "[$(date)] smoke validation done"
