#!/bin/bash
# Camera-ready campaign, experiment 4: accuracy vs cut count at snapshot
# stride 10. One job per path; per tolerance target runs 8/16/32 ranks
# (7/15/31 cuts) with the default planner logic (exact_dp, w = 0.21,
# a0 = 7.9 SS) and compares rel-L2 against the stride-10 serial references
# (dev/run_cr_serial_refs_s10_sbatch.sh — submit this job with
# --dependency=afterok:<that job>). Parallel snapshots are deleted after a
# successful comparison. Resumable: completed comparisons are skipped.
#
# Rank placement on the 8-node allocation: 8r -> 1/GPU, 16r -> 2/GPU,
# 32r -> 4/GPU (bind_local_gpu maps co-located ranks onto the shared device).
#
# Usage:  sbatch --dependency=afterok:<refjob> dev/run_cr_accuracy_cuts_s10_sbatch.sh <bull|hybrid>

#SBATCH -J cr_acc_cuts
#SBATCH -N 8
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH -t 06:00:00
#SBATCH -o logs/cr_acc_cuts_%j.out
#SBATCH -e logs/cr_acc_cuts_%j.err
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

PATHKEY=${1:-bull}
case "${PATHKEY}" in
  bull)              CFG_PREFIX=configs/accuracy/bull ;;
  continuous_hybrid) CFG_PREFIX=configs/accuracy/continuous_hybrid ;;
  *) echo "ERROR: unknown path '${PATHKEY}' (bull|continuous_hybrid)" >&2; exit 1 ;;
esac

SIM_CONFIG=configs/examples/sim_calibration.ini
SNAP_EVERY=10
CORRECTION_WEIGHT=0.21

run_case () {
  local TOL=$1 RANKS=$2
  local ROOT=outputs/accuracy_${PATHKEY}_${TOL}_h30
  local CFG=${CFG_PREFIX}_${TOL}.ini
  local TAG=par${RANKS}_s10_w021
  local NODES=$(( RANKS <= 8 ? RANKS : 8 ))
  local PER_NODE=$(( (RANKS + NODES - 1) / NODES ))

  if [ ! -d "${ROOT}/serial_s10/snapshots_ser" ]; then
    echo "ERROR: stride-10 serial reference ${ROOT}/serial_s10 missing" >&2
    exit 1
  fi
  if [ -f "${ROOT}/compare_${TAG}/comparison_summary.json" ]; then
    echo " [$(date)] ${PATHKEY}/${TOL}/${RANKS}r: already done, skipping"
    return
  fi

  echo "====== [$(date)] ${PATHKEY}/${TOL}: ${RANKS} ranks ($((RANKS-1)) cuts), stride ${SNAP_EVERY} ======"
  srun -N "${NODES}" -n "${RANKS}" --ntasks-per-node="${PER_NODE}" \
    python src/hermes/scripts/segment_correction/main.py \
      --config "${SIM_CONFIG}" --path-config "${CFG}" \
      --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
      --planner-mode exact_dp --correction-weight "${CORRECTION_WEIGHT}" \
      --no-export-dag \
      --out-dir "${ROOT}/${TAG}"

  echo "------ [$(date)] ${PATHKEY}/${TOL}/${RANKS}r: rel-L2 comparison ------"
  srun -N 1 -n 1 --ntasks-per-node=1 python src/hermes/scripts/segment_correction/compare_runs.py \
    --par-snap-dir "${ROOT}/${TAG}/snapshots_par" \
    --ser-snap-dir "${ROOT}/serial_s10/snapshots_ser" \
    --out-dir "${ROOT}/compare_${TAG}" \
    --source-on-only \
    --config "${SIM_CONFIG}" --path-config "${CFG}" --dt-us 10 | tail -6

  if [ -f "${ROOT}/compare_${TAG}/comparison_summary.json" ]; then
    rm -rf "${ROOT}/${TAG}/snapshots_par" "${ROOT}/${TAG}/snapshots_par_meta"
  fi
}

for TOL in tol1e4 tol1e7; do
  for RANKS in 8 16 32; do
    run_case "${TOL}" "${RANKS}"
  done
done

echo ""
echo "===== [$(date)] ${PATHKEY}: error vs cuts (stride 10, w=0.21) ====="
python3 - "${PATHKEY}" <<'PY'
import json, pathlib, sys
pathkey = sys.argv[1]
TARGET = {"tol1e4": 1e-4, "tol1e7": 1e-7}
for tol in ("tol1e4", "tol1e7"):
    root = pathlib.Path(f"outputs/accuracy_{pathkey}_{tol}_h30")
    print(f"\n{pathkey}/{tol} (target {TARGET[tol]:.0e}):")
    for ranks in (8, 16, 32):
        f = root / f"compare_par{ranks}_s10_w021" / "comparison_summary.json"
        if not f.exists():
            print(f"  {ranks:2d} ranks ({ranks-1:2d} cuts): MISSING")
            continue
        d = json.loads(f.read_text())
        flag = "OK " if d["max_rel_l2"] <= TARGET[tol] else "ABOVE"
        print(f"  {ranks:2d} ranks ({ranks-1:2d} cuts): max={d['max_rel_l2']:.4e} "
              f"mean={d['mean_rel_l2']:.4e}  [{flag}]")
PY
