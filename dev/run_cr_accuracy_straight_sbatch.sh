#!/bin/bash
# Camera-ready accuracy tables: straight-line path at 32 ranks (31 cuts),
# stride-10 snapshots, final planner logic — the missing row for the
# updated tab:acc_1e4 / tab:acc_1e7 (bull and spiral-raster 32-rank values
# already measured by the campaign). Serial refs + par32 + compare for both
# tolerance targets. The straight path is tiny (112 / 77 SS), so everything
# fits in one short 8-node job (32 ranks at 4 ranks/GPU).
#
# Usage:  sbatch dev/run_cr_accuracy_straight_sbatch.sh

#SBATCH -J cr_acc_straight
#SBATCH -N 8
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH -t 02:00:00
#SBATCH -o logs/cr_acc_straight_%j.out
#SBATCH -e logs/cr_acc_straight_%j.err
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
unset HERMES_CORRECTION_FIXED_COST_SS HERMES_TRACER_PROFILE

SIM_CONFIG=configs/examples/sim_calibration.ini
SNAP_EVERY=10
CORRECTION_WEIGHT=0.21

run_tol () {
  local TOL=$1
  local ROOT=outputs/accuracy_straight_line_${TOL}_h30
  local CFG=configs/accuracy/straight_line_${TOL}.ini
  local TAG=par32_s10_w021

  if [ ! -d "${ROOT}/serial_s10/snapshots_ser" ]; then
    echo "====== [$(date)] straight_line/${TOL}: serial reference, stride ${SNAP_EVERY} ======"
    srun -N 1 -n 1 --ntasks-per-node=1 --exact \
      python src/hermes/scripts/segment_correction/serial_reference_run.py \
        --config "${SIM_CONFIG}" --path-config "${CFG}" \
        --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
        --out-dir "${ROOT}/serial_s10"
  fi

  if [ -f "${ROOT}/compare_${TAG}/comparison_summary.json" ]; then
    echo " [$(date)] straight_line/${TOL}: already done, skipping"
    return
  fi

  echo "====== [$(date)] straight_line/${TOL}: 32 ranks (31 cuts) ======"
  srun -N 8 -n 32 --ntasks-per-node=4 --kill-on-bad-exit=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config "${SIM_CONFIG}" --path-config "${CFG}" \
      --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
      --planner-mode exact_dp --correction-weight "${CORRECTION_WEIGHT}" \
      --no-export-dag \
      --out-dir "${ROOT}/${TAG}"

  echo "------ [$(date)] straight_line/${TOL}: rel-L2 comparison ------"
  srun -N 1 -n 1 --ntasks-per-node=1 --exact \
    python src/hermes/scripts/segment_correction/compare_runs.py \
      --par-snap-dir "${ROOT}/${TAG}/snapshots_par" \
      --ser-snap-dir "${ROOT}/serial_s10/snapshots_ser" \
      --out-dir "${ROOT}/compare_${TAG}" \
      --source-on-only \
      --config "${SIM_CONFIG}" --path-config "${CFG}" --dt-us 10 | tail -6

  if [ -f "${ROOT}/compare_${TAG}/comparison_summary.json" ]; then
    rm -rf "${ROOT}/${TAG}/snapshots_par" "${ROOT}/${TAG}/snapshots_par_meta"
  fi
}

run_tol tol1e4
run_tol tol1e7

echo ""
echo "===== [$(date)] straight line, 31 cuts (stride 10, w=0.21) ====="
python3 - <<'PY'
import json, pathlib
for tol, target in (("tol1e4", 1e-4), ("tol1e7", 1e-7)):
    f = pathlib.Path(f"outputs/accuracy_straight_line_{tol}_h30/compare_par32_s10_w021/comparison_summary.json")
    if not f.exists():
        print(f"  {tol}: MISSING"); continue
    d = json.loads(f.read_text())
    flag = "OK" if d["max_rel_l2"] <= target else "ABOVE"
    print(f"  {tol}: max={d['max_rel_l2']:.4e} mean={d['mean_rel_l2']:.4e} [{flag}]")
PY
