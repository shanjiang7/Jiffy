#!/bin/bash
# Accuracy check for the NEW default planner logic (source-charged span_own
# cost model, w=0.21, a0=7.9): Bull path, 32 ranks / 31 cuts, both tolerance
# targets, rel-L2 vs the existing serial references. The runtime correction
# logic is unchanged — this validates that the NEW PARTITIONS (different
# boundary placement) still meet the accuracy targets.

#SBATCH -J acc_srcfix
#SBATCH -N 8
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH -t 1:30:00
#SBATCH -o logs/acc_srcfix_%j.out
#SBATCH -e logs/acc_srcfix_%j.err
#SBATCH -p gh-dev
#SBATCH -A ASC21034

set -uo pipefail
PROJECT_DIR=${PROJECT_DIR:-/scratch/10226/shawnraul/work/Jiffy}
cd "${PROJECT_DIR}"
mkdir -p logs outputs
source "${PROJECT_DIR}/env_vista.sh"

SIM_CONFIG=configs/examples/sim_calibration.ini
SNAP_EVERY=25

run_case () {
  local TOL=$1
  local ROOT=outputs/accuracy_bull_${TOL}_h30
  local CFG=configs/accuracy/bull_${TOL}.ini
  if [ ! -d "${ROOT}/serial/snapshots_ser" ]; then
    echo "ERROR: serial reference ${ROOT}/serial missing" >&2
    exit 1
  fi
  if [ -f "${ROOT}/compare_par32_srcfix_w021/comparison_summary.json" ]; then
    echo " [$(date)] bull/${TOL}: already done, skipping"
    return
  fi
  echo "====== [$(date)] bull/${TOL}: 32 ranks, srcfix model (w=0.21) ======"
  srun -N 8 -n 32 --ntasks-per-node=4 python src/hermes/scripts/segment_correction/main.py \
    --config "${SIM_CONFIG}" --path-config "${CFG}" \
    --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
    --planner-mode exact_dp --correction-weight 0.21 \
    --no-export-dag \
    --out-dir "${ROOT}/par32_srcfix_w021"

  echo "------ [$(date)] bull/${TOL}: rel-L2 comparison (source-on only) ------"
  srun -N 1 -n 1 --ntasks-per-node=1 python src/hermes/scripts/segment_correction/compare_runs.py \
    --par-snap-dir "${ROOT}/par32_srcfix_w021/snapshots_par" \
    --ser-snap-dir "${ROOT}/serial/snapshots_ser" \
    --out-dir "${ROOT}/compare_par32_srcfix_w021" \
    --source-on-only \
    --config "${SIM_CONFIG}" --path-config "${CFG}" --dt-us 10 | tail -6

  if [ -f "${ROOT}/compare_par32_srcfix_w021/comparison_summary.json" ]; then
    rm -rf "${ROOT}/par32_srcfix_w021/snapshots_par" \
           "${ROOT}/par32_srcfix_w021/snapshots_par_meta"
  fi
}

run_case tol1e4
run_case tol1e7

echo ""
echo "===== [$(date)] srcfix-model 31-cut accuracy vs previous partitions ====="
python3 - <<'PY'
import json, pathlib
TARGET = {"tol1e4": 1e-4, "tol1e7": 1e-7}
for tol in ("tol1e4", "tol1e7"):
    root = pathlib.Path(f"outputs/accuracy_bull_{tol}_h30")
    print(f"\nbull/{tol} (target {TARGET[tol]:.0e}):")
    for tag in ("chord_fix_w075", "chord_fix_w025", "chord_fix_w0368", "srcfix_w021"):
        f = root / f"compare_par32_{tag}" / "comparison_summary.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        flag = "OK " if d["max_rel_l2"] <= TARGET[tol] else "ABOVE"
        print(f"  {tag:16s} max={d['max_rel_l2']:.4e} mean={d['mean_rel_l2']:.4e}  [{flag}]")
PY
