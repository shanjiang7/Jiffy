#!/bin/bash
# 31-cut accuracy at h = 30 um with the MEASURED correction weight for this
# grid: w = 0.368 (dev/bench_source_on_off.py, job 852855 - source-off /
# source-on per-step ratio on the 161x161x81 calibration grid). The earlier
# legs used 0.75 (historical default) and 0.25 (h=18 measured value); this is
# the partition the true h30 cost model picks.
#
# bull + hybrid x tol1e4 + tol1e7, 32 ranks / 31 cuts, chord lookup,
# vs the existing h30 serial references. Resumable.
#
# Usage:  sbatch dev/run_acc_cuts31_w0368_sbatch.sh

#SBATCH -J acc_cuts31_w0368
#SBATCH -N 8
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH -t 1:30:00
#SBATCH -o logs/acc_cuts31_w0368_%j.out
#SBATCH -e logs/acc_cuts31_w0368_%j.err
#SBATCH -p gh
#SBATCH -A ASC21034

set -uo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scratch/10226/shawnraul/work/Jiffy}
cd "${PROJECT_DIR}"
mkdir -p logs outputs

source "${PROJECT_DIR}/env_vista.sh"

SIM_CONFIG=configs/examples/sim_calibration.ini
SNAP_EVERY=25
PLANNER_MODE=exact_dp
WEIGHT=0.368

for PATHNAME in bull hybrid; do
  for TOL in tol1e4 tol1e7; do
    ROOT=outputs/accuracy_${PATHNAME}_${TOL}_h30
    if [ "${PATHNAME}" = "bull" ]; then
      CFG=configs/accuracy/bull_${TOL}.ini
    else
      CFG=configs/accuracy/hybrid_spiral_raster_${TOL}.ini
    fi
    if [ ! -d "${ROOT}/serial/snapshots_ser" ]; then
      echo "ERROR: serial reference ${ROOT}/serial missing" >&2
      exit 1
    fi
    if [ -f "${ROOT}/compare_par32_chord_fix_w0368/comparison_summary.json" ]; then
      echo " [$(date)] ${PATHNAME}/${TOL}: already done, skipping"
      continue
    fi

    echo "====== [$(date)] ${PATHNAME}/${TOL}: 32 ranks / 31 cuts, weight ${WEIGHT} ======"
    srun -N 8 -n 32 --ntasks-per-node=4 python src/hermes/scripts/segment_correction/main.py \
      --config "${SIM_CONFIG}" --path-config "${CFG}" \
      --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
      --planner-mode "${PLANNER_MODE}" --correction-weight "${WEIGHT}" \
      --no-export-dag \
      --out-dir "${ROOT}/par32_chord_fix_w0368"

    echo "------ [$(date)] ${PATHNAME}/${TOL}: rel-L2 comparison (source-on only) ------"
    srun -N 1 -n 1 --ntasks-per-node=1 python src/hermes/scripts/segment_correction/compare_runs.py \
      --par-snap-dir "${ROOT}/par32_chord_fix_w0368/snapshots_par" \
      --ser-snap-dir "${ROOT}/serial/snapshots_ser" \
      --out-dir "${ROOT}/compare_par32_chord_fix_w0368" \
      --source-on-only \
      --config "${SIM_CONFIG}" --path-config "${CFG}" --dt-us 10 | tail -6

    if [ -f "${ROOT}/compare_par32_chord_fix_w0368/comparison_summary.json" ]; then
      echo " [$(date)] comparison saved - deleting par32 snapshots to save space"
      rm -rf "${ROOT}/par32_chord_fix_w0368/snapshots_par" "${ROOT}/par32_chord_fix_w0368/snapshots_par_meta"
    fi
  done
done

echo ""
echo "[$(date)] all done. 31-cut accuracy at h=30 across weights:"
python3 - <<'PY'
import json, pathlib
TARGET = {"tol1e4": 1e-4, "tol1e7": 1e-7}
LEGS = [("w075 (old default)", "compare_par32_chord_fix_w075"),
        ("w025 (h18 measured)", "compare_par32_chord_fix_w025"),
        ("w0368 (h30 measured)", "compare_par32_chord_fix_w0368")]
for path in ("bull", "hybrid"):
    for tol in ("tol1e4", "tol1e7"):
        root = pathlib.Path(f"outputs/accuracy_{path}_{tol}_h30")
        for label, sub in LEGS:
            f = root / sub / "comparison_summary.json"
            if not f.exists():
                print(f"RESULT {path}/{tol} {label}: MISSING")
                continue
            d = json.loads(f.read_text())
            ok = "PASS" if d["max_rel_l2"] <= TARGET[tol] else "FAIL"
            print(f"RESULT {path}/{tol} {label}: max={d['max_rel_l2']:.4e} "
                  f"mean={d['mean_rel_l2']:.4e}  target={TARGET[tol]:.0e}  {ok}")
        print()
PY
