#!/bin/bash
# Self-convergence ladder at the +4 SS/iteration horizon step (explicit override;
# 6 iterations = +24 SS total, the protocol of the published
# tab:self_convergence table. Bull + continuous-hybrid, tol1e4, 32 ranks,
# stride-10 snapshots vs the stride-10 serial references (matches
# the accuracy-table sampling, so u0 equals the Table IV entry).
# Prints the paper-style table (tab:self_convergence format):
#   iterate | true max rel-L2 | ||d_k||   + the cumulative-estimate check.
#
# Usage:  sbatch experiments/accuracy/run_self_convergence_sbatch.sh

#SBATCH -J cr_selfchk4
#SBATCH -N 8
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH -t 4:00:00
#SBATCH -o logs/cr_selfchk4_%j.out
#SBATCH -e logs/cr_selfchk4_%j.err
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
ITERS=6
WEIGHT=0.21
TAG=par32_ladder_step4_s10_w021

run_ladder () {
  local PATHKEY=$1
  local ROOT=outputs/accuracy_${PATHKEY}_tol1e4_h30
  local CFG
  case "${PATHKEY}" in
    bull)              CFG=configs/accuracy/bull_tol1e4.ini ;;
    spiral_raster) CFG=configs/accuracy/spiral_raster_tol1e4.ini ;;
  esac
  if [ ! -d "${ROOT}/serial_s10/snapshots_ser" ]; then
    echo "ERROR: stride-10 serial reference ${ROOT}/serial_s10 missing" >&2
    exit 1
  fi
  if [ -f "${ROOT}/compare_${TAG}_iter${ITERS}/comparison_summary.json" ]; then
    echo " [$(date)] ${PATHKEY}: step-4 ladder already done, skipping"
    return
  fi

  echo "====== [$(date)] ${PATHKEY}/tol1e4: 32 ranks, ladder x${ITERS} @ +4 SS ======"
  srun -N 8 -n 32 --ntasks-per-node=4 --kill-on-bad-exit=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config "${SIM_CONFIG}" --path-config "${CFG}" \
      --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
      --planner-mode exact_dp --correction-weight "${WEIGHT}" --no-export-dag \
      --self-check --self-check-mode horizon --self-check-horizon-step 4 \
      --self-check-iters "${ITERS}" --self-check-save-iters \
      --out-dir "${ROOT}/${TAG}"

  echo "------ [$(date)] ${PATHKEY}: truth comparisons (production + each iterate) ------"
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
}

run_ladder bull
run_ladder spiral_raster

echo ""
echo "===== [$(date)] paper-style tables (step +4 SS/iter, 32 ranks, tol1e4) ====="
L=logs/cr_selfchk4_${SLURM_JOB_ID}.out
python3 - "$L" "${TAG}" "${ITERS}" <<'PY'
import json, pathlib, re, sys

log_text = pathlib.Path(sys.argv[1]).read_text()
tag, iters = sys.argv[2], int(sys.argv[3])

sections = {}
for path in ("bull", "spiral_raster"):
    m = re.search(rf"{path}/tol1e4: 32 ranks.*?(?======= \[|\Z)", log_text, re.S)
    sections[path] = m.group(0) if m else ""

for path in ("bull", "spiral_raster"):
    shifts = dict(
        (int(k), float(v))
        for k, v in re.findall(r"\[self-check\] iter (\d+): shift max=([0-9.eE+-]+)", sections[path])
    )
    root = pathlib.Path(f"outputs/accuracy_{path}_tol1e4_h30")

    def truth(suffix):
        f = root / f"compare_{tag}{suffix}" / "comparison_summary.json"
        return json.loads(f.read_text())["max_rel_l2"] if f.exists() else None

    print(f"\n{path} (Iterate | true max rel-L2 | ||d_k||):")
    t0 = truth("")
    print(f"  u^(0)  {t0:.4e}      ---" if t0 is not None else "  u^(0)  MISSING")
    est = 0.0
    for k in range(1, iters + 1):
        tk = truth(f"_iter{k}")
        dk = shifts.get(k)
        est += dk or 0.0
        tk_s = f"{tk:.4e}" if tk is not None else "MISSING"
        dk_s = f"{dk:.4e}" if dk is not None else "MISSING"
        print(f"  u^({k})  {tk_s}  {dk_s}")
    if t0 is not None and est > 0:
        print(f"  cumulative estimate sum(d_k) = {est:.4e}  vs true u^(0) error {t0:.4e}"
              f"  (ratio {est / t0:.4f})")
PY
