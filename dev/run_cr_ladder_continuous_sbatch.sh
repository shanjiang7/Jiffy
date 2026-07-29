#!/bin/bash
# Self-convergence refinement ladder on the CONTINUOUS-HYBRID path (the
# fifth path, added precisely for this study). 32 ranks / 31 cuts, tol1e4,
# 10 horizon-only refinement iterations, default planner logic (exact_dp,
# w = 0.21, a0 = 7.9 SS). Hypothesis under test: on the discontinuous
# hybrid the max-error ladder plateaus and then collapses 3-4 orders in one
# iteration (deep-dependency site at the stranded spiral center); on a
# geometrically continuous path the information flow is chain-like and the
# max error should decay smoothly/geometrically per iteration.
#
# Snapshots at stride 20 (every step divisible by 20 also exists in the
# stride-10 serial reference, so comparisons hit every ladder snapshot
# while keeping the 11-iterate disk footprint moderate).
#
# Usage:  sbatch --dependency=afterok:<refs job> dev/run_cr_ladder_continuous_sbatch.sh

#SBATCH -J cr_ladder_ch
#SBATCH -N 8
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH -t 3:00:00
#SBATCH -o logs/cr_ladder_ch_%j.out
#SBATCH -e logs/cr_ladder_ch_%j.err
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
CFG=configs/accuracy/continuous_hybrid_tol1e4.ini
ROOT=outputs/accuracy_continuous_hybrid_tol1e4_h30
SNAP_EVERY=20
ITERS=10
WEIGHT=0.21
TAG=par32_ladder_s20_w021

if [ ! -d "${ROOT}/serial_s10/snapshots_ser" ]; then
  echo "ERROR: stride-10 serial reference ${ROOT}/serial_s10 missing" >&2
  exit 1
fi

if [ ! -f "${ROOT}/compare_${TAG}_iter${ITERS}/comparison_summary.json" ]; then
  echo "====== [$(date)] continuous_hybrid/tol1e4: 32 ranks, ladder x${ITERS}, w=${WEIGHT} ======"
  srun -N 8 -n 32 --ntasks-per-node=4 --kill-on-bad-exit=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config "${SIM_CONFIG}" --path-config "${CFG}" \
      --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
      --planner-mode exact_dp --correction-weight "${WEIGHT}" --no-export-dag \
      --self-check --self-check-mode horizon --self-check-horizon-step 1 \
      --self-check-iters "${ITERS}" --self-check-save-iters \
      --out-dir "${ROOT}/${TAG}"

  echo "------ [$(date)] truth comparisons (production + each iterate) ------"
  for K in "" $(seq -f "_iter%g" 1 ${ITERS}); do
    if [ -f "${ROOT}/compare_${TAG}${K}/comparison_summary.json" ]; then
      continue
    fi
    srun -N 1 -n 1 --ntasks-per-node=1 python src/hermes/scripts/segment_correction/compare_runs.py \
      --par-snap-dir "${ROOT}/${TAG}/snapshots_par${K}" \
      --ser-snap-dir "${ROOT}/serial_s10/snapshots_ser" \
      --out-dir "${ROOT}/compare_${TAG}${K}" \
      --source-on-only \
      --config "${SIM_CONFIG}" --path-config "${CFG}" --dt-us 10 | tail -3
    if [ -f "${ROOT}/compare_${TAG}${K}/comparison_summary.json" ]; then
      rm -rf "${ROOT}/${TAG}/snapshots_par${K}" "${ROOT}/${TAG}/snapshots_par${K}_meta"
    fi
  done
else
  echo " [$(date)] ladder already complete, printing summary only"
fi

echo ""
echo "===== [$(date)] continuous-hybrid ladder decay (max rel-L2 per iterate) ====="
python3 - "${ROOT}" "${TAG}" "${ITERS}" <<'PY'
import json, pathlib, sys
root, tag, iters = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
prev = None
for k in range(1, iters + 1):
    f = root / f"compare_{tag}_iter{k}" / "comparison_summary.json"
    if not f.exists():
        print(f"  iter {k:2d}: MISSING"); continue
    d = json.loads(f.read_text())
    ratio = f"  x{prev / d['max_rel_l2']:8.2f} vs prev" if prev else ""
    print(f"  iter {k:2d}: max={d['max_rel_l2']:.4e} mean={d['mean_rel_l2']:.4e}{ratio}")
    prev = d["max_rel_l2"]
f = root / f"compare_{tag}" / "comparison_summary.json"
if f.exists():
    d = json.loads(f.read_text())
    print(f"  production (no ladder): max={d['max_rel_l2']:.4e}")
print("\n  smooth decay = roughly constant per-iteration ratio; a plateau followed")
print("  by a >100x single-step drop reproduces the discontinuous-hybrid cliff.")
PY
