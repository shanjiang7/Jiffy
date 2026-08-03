#!/bin/bash
# Re-verify the self-convergence ladder (horizon-only) with the NEW multi-rank
# correction logic (fused bridge tracer 34cbf67 + per-edge horizon bc66e55).
#
# Both paths (bull + hybrid spiral-raster), 32 ranks / 31 cuts, tol1e4,
# 10 refinement iterations, correction weight 0.25 (shipping default).
# Verification criterion: cumulative estimate of u_0 vs true rel-L2 vs the
# serial reference -- ratio should stay ~1.000 (pre-fix: 1.000 on both paths,
# jobs 847803/847804). The ladder itself still uses the per-COMPONENT horizon
# map while production uses per-EDGE horizons; a ratio drifting from 1.0
# would mean the ladder needs rebasing onto the per-edge map.
#
# Fresh out-dirs (par32_ladder_fix / compare_par32_ladder_fix*) so the
# pre-fix baselines are preserved.
#
# Usage:  sbatch dev/run_selfcheck31_fix_sbatch.sh

#SBATCH -J selfchk31_fix
#SBATCH -N 8
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH -t 1:30:00
#SBATCH -o logs/selfchk31_fix_%j.out
#SBATCH -e logs/selfchk31_fix_%j.err
#SBATCH -p gh
#SBATCH -A ASC21034

set -uo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scratch/10226/shawnraul/work/Jiffy}
cd "${PROJECT_DIR}"
mkdir -p logs outputs

source "${PROJECT_DIR}/env_vista.sh"

SIM_CONFIG=configs/examples/sim_calibration.ini
SNAP_EVERY=25
ITERS=10
WEIGHT=0.25

run_ladder () {
  local PATHNAME=$1
  local ROOT=outputs/accuracy_${PATHNAME}_tol1e4_h30
  local CFG
  if [ "${PATHNAME}" = "bull" ]; then
    CFG=configs/accuracy/bull_tol1e4.ini
  else
    CFG=configs/accuracy/hybrid_spiral_raster_tol1e4.ini
  fi
  if [ ! -d "${ROOT}/serial/snapshots_ser" ]; then
    echo "ERROR: serial reference ${ROOT}/serial missing" >&2
    exit 1
  fi
  if [ -f "${ROOT}/compare_par32_ladder_fix_iter${ITERS}/comparison_summary.json" ]; then
    echo " [$(date)] ${PATHNAME}: ladder already done, skipping"
    return
  fi

  echo "====== [$(date)] ${PATHNAME}/tol1e4: 32 ranks, horizon-only ladder x${ITERS}, weight ${WEIGHT} (post-fix) ======"
  srun -N 8 -n 32 --ntasks-per-node=4 python src/hermes/scripts/segment_correction/main.py \
    --config "${SIM_CONFIG}" --path-config "${CFG}" \
    --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
    --planner-mode exact_dp --correction-weight "${WEIGHT}" --no-export-dag \
    --self-check --self-check-mode horizon --self-check-horizon-step 1 \
    --self-check-iters "${ITERS}" --self-check-save-iters \
    --out-dir "${ROOT}/par32_ladder_fix"

  echo "------ [$(date)] ${PATHNAME}: truth comparisons (production + each iterate) ------"
  for K in "" $(seq -f "_iter%g" 1 ${ITERS}); do
    srun -N 1 -n 1 --ntasks-per-node=1 python src/hermes/scripts/segment_correction/compare_runs.py \
      --par-snap-dir "${ROOT}/par32_ladder_fix/snapshots_par${K}" \
      --ser-snap-dir "${ROOT}/serial/snapshots_ser" \
      --out-dir "${ROOT}/compare_par32_ladder_fix${K}" \
      --source-on-only \
      --config "${SIM_CONFIG}" --path-config "${CFG}" --dt-us 10 | tail -3
    if [ -f "${ROOT}/compare_par32_ladder_fix${K}/comparison_summary.json" ]; then
      rm -rf "${ROOT}/par32_ladder_fix/snapshots_par${K}" \
             "${ROOT}/par32_ladder_fix/snapshots_par${K}_meta"
    fi
  done
}

run_ladder bull
run_ladder hybrid

echo ""
echo "[$(date)] ===== estimate vs true rel-L2 (31 cuts, tol1e4, horizon-only, post-fix) ====="
L=logs/selfchk31_fix_${SLURM_JOB_ID}.out
python3 - "$L" "${ITERS}" <<'PYEOF'
import json, re, sys

log = open(sys.argv[1]).read()
iters = int(sys.argv[2])
OLD = {"bull": 1.0767e-04, "hybrid": 3.9433e-04}  # pre-fix headline estimates

# Split the log into per-path sections so shift/cum regexes don't mix paths.
sections = {}
for path in ("bull", "hybrid"):
    m = re.search(rf"{path}/tol1e4: 32 ranks.*?(?======= \[|\Z)", log, re.S)
    sections[path] = m.group(0) if m else ""

for path in ("bull", "hybrid"):
    sec = sections[path]
    shifts = re.findall(r"\[self-check\] iter (\d+): shift max=([0-9.e+-]+)", sec)
    cums = re.findall(r"cumulative-estimate-of-u0 max=([0-9.e+-]+)", sec)

    def truth(suffix):
        try:
            d = json.load(open(f"outputs/accuracy_{path}_tol1e4_h30/compare_par32_ladder_fix{suffix}/comparison_summary.json"))
            return d["max_rel_l2"]
        except Exception:
            return None

    def fmt(x):
        return f"{x:.4e}" if isinstance(x, float) else "n/a"

    t0 = truth("")
    print(f"\n  --- {path} ---")
    print(f"  {'iterate':<10} {'true max rel-L2':>16} {'shift d_k':>12} {'cum est of u_0':>15}")
    print(f"  {'u_0 (prod)':<10} {fmt(t0):>16} {'-':>12} {'-':>15}")
    for k in range(1, iters + 1):
        tk = truth(f"_iter{k}")
        sh = shifts[k-1][1] if k-1 < len(shifts) else "-"
        cm = cums[k-1] if k-1 < len(cums) else "-"
        print(f"  {'u_' + str(k):<10} {fmt(tk):>16} {sh:>12} {cm:>15}")
    est = float(cums[-1]) if cums else None
    if est is not None and t0 is not None:
        print(f"  RESULT {path}: estimated {est:.4e}  vs  true {t0:.4e}"
              f"  (ratio {est/t0:.3f}; pre-fix est {OLD[path]:.4e} at w0.75)")
PYEOF
