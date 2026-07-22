#!/bin/bash
# Regenerate the paper's self-convergence table (tab:self_convergence,
# self_convergence_section.tex) with the NEW multi-rank correction logic:
# Bull path, tol1e4 config, 32 ranks / 31 cuts, SIX iterations, FOUR
# additional supersegments of correction per iteration.
#
# Two legs:
#   w075 : --correction-weight 0.75 -> same partition as the published table
#          (u0 = 1.0767e-4); direct old-vs-new comparison of every row
#   w025 : shipping default -> camera-ready candidate table if the paper
#          switches to the measured weight
#
# Usage:  sbatch dev/run_selfcheck31_step4_sbatch.sh

#SBATCH -J selfchk31_s4
#SBATCH -N 8
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH -t 1:30:00
#SBATCH -o logs/selfchk31_s4_%j.out
#SBATCH -e logs/selfchk31_s4_%j.err
#SBATCH -p gh
#SBATCH -A ASC21034

set -uo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scratch/10226/shawnraul/work/Jiffy}
cd "${PROJECT_DIR}"
mkdir -p logs outputs

source "${PROJECT_DIR}/env_vista.sh"

SIM_CONFIG=configs/examples/sim_calibration.ini
CFG=configs/accuracy/bull_tol1e4.ini
ROOT=outputs/accuracy_bull_tol1e4_h30
SNAP_EVERY=25
ITERS=6
STEP=4

if [ ! -d "${ROOT}/serial/snapshots_ser" ]; then
  echo "ERROR: serial reference ${ROOT}/serial missing" >&2
  exit 1
fi

run_ladder () {
  local WTAG=$1 WEIGHT=$2
  local OUT=${ROOT}/par32_ladder4_${WTAG}
  if [ -f "${ROOT}/compare_par32_ladder4_${WTAG}_iter${ITERS}/comparison_summary.json" ]; then
    echo " [$(date)] bull/step4/${WTAG}: already done, skipping"
    return
  fi

  echo "====== [$(date)] bull/tol1e4: 32 ranks, ladder x${ITERS}, step ${STEP} SS, weight ${WEIGHT} ======"
  srun -N 8 -n 32 --ntasks-per-node=4 python src/hermes/scripts/segment_correction/main.py \
    --config "${SIM_CONFIG}" --path-config "${CFG}" \
    --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
    --planner-mode exact_dp --correction-weight "${WEIGHT}" --no-export-dag \
    --self-check --self-check-mode horizon --self-check-horizon-step "${STEP}" \
    --self-check-iters "${ITERS}" --self-check-save-iters \
    --out-dir "${OUT}"

  echo "------ [$(date)] bull/step4/${WTAG}: truth comparisons ------"
  for K in "" $(seq -f "_iter%g" 1 ${ITERS}); do
    srun -N 1 -n 1 --ntasks-per-node=1 python src/hermes/scripts/segment_correction/compare_runs.py \
      --par-snap-dir "${OUT}/snapshots_par${K}" \
      --ser-snap-dir "${ROOT}/serial/snapshots_ser" \
      --out-dir "${ROOT}/compare_par32_ladder4_${WTAG}${K}" \
      --source-on-only \
      --config "${SIM_CONFIG}" --path-config "${CFG}" --dt-us 10 | tail -3
    if [ -f "${ROOT}/compare_par32_ladder4_${WTAG}${K}/comparison_summary.json" ]; then
      rm -rf "${OUT}/snapshots_par${K}" "${OUT}/snapshots_par${K}_meta"
    fi
  done
}

run_ladder w075 0.75
run_ladder w025 0.25

echo ""
echo "[$(date)] ===== tab:self_convergence regeneration (bull, tol1e4, 31 cuts, step ${STEP}) ====="
L=logs/selfchk31_s4_${SLURM_JOB_ID}.out
python3 - "$L" "${ITERS}" <<'PYEOF'
import json, re, sys

log = open(sys.argv[1]).read()
iters = int(sys.argv[2])

# Published table rows (pre-fix, w0.75) for side-by-side comparison.
PUB_TRUE = [1.0767e-4, 3.6498e-5, 1.6113e-5, 9.9789e-6, 3.3293e-6, 8.9734e-7, 1.8131e-7]
PUB_DK   = [None, 1.0767e-4, 3.6497e-5, 1.6113e-5, 9.9778e-6, 3.3284e-6, 8.9678e-7]

sections = re.split(r"====== \[", log)
for wtag, weight in (("w075", "0.75"), ("w025", "0.25")):
    sec = next((s for s in sections if f"weight {weight}" in s.split("\n", 1)[0]), "")
    shifts = re.findall(r"\[self-check\] iter (\d+): shift max=([0-9.e+-]+)", sec)
    cums = re.findall(r"cumulative-estimate-of-u0 max=([0-9.e+-]+)", sec)

    def truth(suffix):
        try:
            d = json.load(open(f"outputs/accuracy_bull_tol1e4_h30/compare_par32_ladder4_{wtag}{suffix}/comparison_summary.json"))
            return d["max_rel_l2"]
        except Exception:
            return None

    def fmt(x):
        return f"{x:.4e}" if isinstance(x, float) else ("-" if x is None else str(x))

    print(f"\n  --- weight {weight} ({'published-partition' if wtag=='w075' else 'shipping default'}) ---")
    hdr_pub = "  published(true/d_k)" if wtag == "w075" else ""
    print(f"  {'iterate':<8} {'true max rel-L2':>16} {'d_k':>12} {'cum est':>12}{hdr_pub}")
    for k in range(0, iters + 1):
        tk = truth("" if k == 0 else f"_iter{k}")
        sh = shifts[k-1][1] if 1 <= k <= len(shifts) else "-"
        cm = cums[k-1] if 1 <= k <= len(cums) else "-"
        pub = ""
        if wtag == "w075" and k < len(PUB_TRUE):
            pub = f"   {fmt(PUB_TRUE[k])} / {fmt(PUB_DK[k])}"
        print(f"  u_{k:<6} {fmt(tk):>16} {sh:>12} {cm:>12}{pub}")
    t0 = truth("")
    est = float(cums[-1]) if cums else None
    if est is not None and t0 is not None:
        print(f"  RESULT {wtag}: estimated {est:.4e} vs true {t0:.4e} (ratio {est/t0:.3f})")
PYEOF
