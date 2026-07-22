#!/bin/bash
# Re-validate 31-cut (32-rank) accuracy with the NEW multi-rank correction
# logic (fused bridge tracer 34cbf67 + per-edge horizon bc66e55).
#
# Real 32-rank runs on 8 GH200 nodes (4 ranks per GPU), chord lookup source
# (the shipping default), bull + hybrid paths, both tolerances, at TWO weights:
#   w075 : --correction-weight 0.75 pinned -> same partition as the pre-fix
#          par32 baselines (job 842066/842067), isolating the tracer change
#   w025 : --correction-weight 0.25 (shipping default) -> validates current
#          defaults end-to-end
# 2 paths x 2 tols x 2 weights = 8 runs. Compared vs the existing serial
# references; completed comparisons are skipped (resumable).
#
# Usage:  sbatch dev/run_acc_cuts31_fix_sbatch.sh

#SBATCH -J acc_cuts31_fix
#SBATCH -N 8
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH -t 2:00:00
#SBATCH -o logs/acc_cuts31_fix_%j.out
#SBATCH -e logs/acc_cuts31_fix_%j.err
#SBATCH -p gh-dev
#SBATCH -A ASC21034

set -uo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scratch/10226/shawnraul/work/Jiffy}
cd "${PROJECT_DIR}"
mkdir -p logs outputs

source "${PROJECT_DIR}/env_vista.sh"

SIM_CONFIG=configs/examples/sim_calibration.ini
SNAP_EVERY=25
PLANNER_MODE=exact_dp

run_case () {
  local PATHNAME=$1 TOL=$2 WTAG=$3 WEIGHT=$4
  local ROOT=outputs/accuracy_${PATHNAME}_${TOL}_h30
  local CFG
  if [ "${PATHNAME}" = "bull" ]; then
    CFG=configs/accuracy/bull_${TOL}.ini
  else
    CFG=configs/accuracy/hybrid_spiral_raster_${TOL}.ini
  fi
  if [ ! -d "${ROOT}/serial/snapshots_ser" ]; then
    echo "ERROR: serial reference ${ROOT}/serial missing" >&2
    exit 1
  fi
  if [ -f "${ROOT}/compare_par32_chord_fix_${WTAG}/comparison_summary.json" ]; then
    echo " [$(date)] ${PATHNAME}/${TOL}/${WTAG}: already done, skipping"
    return
  fi

  echo "======================================================"
  echo " [$(date)] ${PATHNAME}/${TOL}: 32 ranks / 31 cuts, weight ${WEIGHT} (post-fix tracer)"
  echo "======================================================"
  srun -N 8 -n 32 --ntasks-per-node=4 python src/hermes/scripts/segment_correction/main.py \
    --config "${SIM_CONFIG}" --path-config "${CFG}" \
    --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
    --planner-mode "${PLANNER_MODE}" --correction-weight "${WEIGHT}" \
    --no-export-dag \
    --out-dir "${ROOT}/par32_chord_fix_${WTAG}"

  echo "------ [$(date)] ${PATHNAME}/${TOL}/${WTAG}: rel-L2 comparison (source-on only) ------"
  srun -N 1 -n 1 --ntasks-per-node=1 python src/hermes/scripts/segment_correction/compare_runs.py \
    --par-snap-dir "${ROOT}/par32_chord_fix_${WTAG}/snapshots_par" \
    --ser-snap-dir "${ROOT}/serial/snapshots_ser" \
    --out-dir "${ROOT}/compare_par32_chord_fix_${WTAG}" \
    --source-on-only \
    --config "${SIM_CONFIG}" \
    --path-config "${CFG}" \
    --dt-us 10 | tail -6

  if [ -f "${ROOT}/compare_par32_chord_fix_${WTAG}/comparison_summary.json" ]; then
    echo " [$(date)] comparison saved - deleting par32 snapshots to save space"
    rm -rf "${ROOT}/par32_chord_fix_${WTAG}/snapshots_par" \
           "${ROOT}/par32_chord_fix_${WTAG}/snapshots_par_meta"
  fi
}

# w075 first (direct old-vs-new comparison), then shipping default.
for PATHNAME in bull hybrid; do
  for TOL in tol1e4 tol1e7; do
    run_case "${PATHNAME}" "${TOL}" w075 0.75
    run_case "${PATHNAME}" "${TOL}" w025 0.25
  done
done

echo ""
echo "[$(date)] all done. 31-cut accuracy, pre-fix vs post-fix tracer:"
python3 - <<'PY'
import json, pathlib
TARGET = {"tol1e4": 1e-4, "tol1e7": 1e-7}
for path in ("bull", "hybrid"):
    for tol in ("tol1e4", "tol1e7"):
        root = pathlib.Path(f"outputs/accuracy_{path}_{tol}_h30")
        rows = [("pre-fix  w075", root / "compare_par32_chord/comparison_summary.json"),
                ("post-fix w075", root / "compare_par32_chord_fix_w075/comparison_summary.json"),
                ("post-fix w025", root / "compare_par32_chord_fix_w025/comparison_summary.json")]
        for label, f in rows:
            if not f.exists():
                print(f"RESULT {path}/{tol} {label}: MISSING")
                continue
            d = json.loads(f.read_text())
            ok = "PASS" if d["max_rel_l2"] <= TARGET[tol] else "FAIL"
            print(f"RESULT {path}/{tol} {label}: max={d['max_rel_l2']:.4e} "
                  f"mean={d['mean_rel_l2']:.4e} n={d['num_compared']}  "
                  f"target={TARGET[tol]:.0e}  {ok}")
        print()
PY
