#!/bin/bash
# Validate the fused bridge tracer (one source-off run per component instead of
# one per successor).
#
# The fused run traces the same trajectory and samples it at the same steps, so
# the corrections it delivers must be IDENTICAL, not merely within tolerance.
# This job checks that, and measures what the removed redundancy was costing.
#
#   (a) correctness: Bull 8-rank accuracy at both tolerance targets, compared
#       against the cached serial references. Must reproduce exactly:
#         tol1e4  max = 2.3421788795e-06   mean = 3.3703840575e-09
#         tol1e7  max = 2.2811036227e-08   mean = 7.4877123666e-11
#       --correction-weight 0.75 is PINNED here on purpose: those baselines were
#       produced before the default changed to the measured 0.25 (commit
#       c8e4edc), and the weight moves the cut placement. Pinning it keeps this
#       job a clean test of the tracer change alone. Re-validating accuracy at
#       0.25 is a separate task.
#   (b) timing: Bull 8 ranks, h = 18 um, level_K = 0.01 K, timing-only, weight
#       0.25 -- the configuration of job 853164 (168.1 s; T(1) = 841.5 s from
#       job 853194 => 5.00x). Projected ~134 s => ~6.3x if the redundant
#       re-tracing was the binding cost.
#
# Usage:  sbatch dev/run_fused_bridge_validation_sbatch.sh

#SBATCH -J fused_val
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -o logs/fused_val_%j.out
#SBATCH -e logs/fused_val_%j.err
#SBATCH -p gh
#SBATCH -A ASC21034

set -uo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
while [ ! -f "${PROJECT_DIR}/env_vista.sh" ] && [ "${PROJECT_DIR}" != "/" ]; do
  PROJECT_DIR=$(dirname "${PROJECT_DIR}")
done
if [ ! -f "${PROJECT_DIR}/env_vista.sh" ]; then
  echo "ERROR: could not locate the Jiffy repo root (env_vista.sh)" >&2
  exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p logs outputs

source "${PROJECT_DIR}/env_vista.sh"

SIM_CONFIG=configs/examples/sim_calibration.ini    # h = 30 um accuracy grid
SNAP_EVERY=25

echo "###################################################################"
echo " Fused bridge tracer validation"
echo "###################################################################"

# ---- (a) correctness: identical corrections => identical rel-L2 -------------
for TOL in tol1e4 tol1e7; do
  ROOT=outputs/accuracy_bull_${TOL}_h30
  CFG=configs/accuracy/bull_${TOL}.ini
  if [ ! -d "${ROOT}/serial/snapshots_ser" ]; then
    echo "ERROR: missing serial reference ${ROOT}/serial/snapshots_ser" >&2
    exit 1
  fi

  echo ""
  echo "###### [$(date)] (a) bull/${TOL}: parallel 8 ranks (fused tracer) ######"
  srun -N 8 -n 8 --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config "${SIM_CONFIG}" --path-config "${CFG}" \
      --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
      --planner-mode exact_dp --correction-weight 0.75 --no-export-dag \
      --out-dir "${ROOT}/par8_fused"

  echo "------ [$(date)] bull/${TOL}: rel-L2 comparison (source-on only) ------"
  srun -N 1 -n 1 --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/compare_runs.py \
      --par-snap-dir "${ROOT}/par8_fused/snapshots_par" \
      --ser-snap-dir "${ROOT}/serial/snapshots_ser" \
      --out-dir "${ROOT}/compare_fused" \
      --source-on-only \
      --config "${SIM_CONFIG}" --path-config "${CFG}" --dt-us 10 | tail -6
done

# ---- (b) timing: the configuration of job 853164 ---------------------------
echo ""
echo "###### [$(date)] (b) Bull 8 ranks, h = 18 um, timing-only ######"
srun -N 8 -n 8 --ntasks-per-node=1 \
  python src/hermes/scripts/segment_correction/main.py \
    --config configs/examples/sim_ex1.ini \
    --path-config configs/examples/fast_heat.ini \
    --dt-us 10 --planner-mode exact_dp --correction-weight 0.25 \
    --timing-only --no-export-dag \
    --out-dir outputs/fused_val/bull_8r

echo ""
echo "===== [$(date)] fused bridge tracer validation ====="
python3 - <<'PY'
import json, pathlib

BASE = {
    "tol1e4": (2.3421788795439975e-06, 3.370384057515057e-09),
    "tol1e7": (2.2811036227e-08, 7.4877123666e-11),
}
print("(a) correctness -- corrections must be IDENTICAL, not merely in tolerance")
ok = True
for tol, (bmax, bmean) in BASE.items():
    f = pathlib.Path(f"outputs/accuracy_bull_{tol}_h30/compare_fused/comparison_summary.json")
    if not f.exists():
        print(f"  bull/{tol}: MISSING {f}")
        ok = False
        continue
    d = json.loads(f.read_text())
    dmax, dmean = float(d["max_rel_l2"]), float(d["mean_rel_l2"])
    rel = abs(dmax - bmax) / bmax if bmax else 0.0
    verdict = "IDENTICAL" if rel < 1e-9 else ("CLOSE" if rel < 1e-3 else "CHANGED")
    if verdict != "IDENTICAL":
        ok = False
    print(f"  bull/{tol}: max={dmax:.10e} (was {bmax:.10e})  "
          f"mean={dmean:.10e} (was {bmean:.10e})  rel_delta={rel:.2e}  {verdict}")
print("  =>", "PASS" if ok else "INVESTIGATE")

print()
print("(b) timing -- Bull 8 ranks, h = 18 um, level_K = 0.01 K")
f = pathlib.Path("outputs/fused_val/bull_8r/timing_summary.json")
if f.exists():
    d = json.loads(f.read_text())
    t8 = float(d["parallel_total_seconds"])
    t1 = 841.5   # job 853194; 1 rank has no cuts, so this change cannot move it
    print(f"  T(8) = {t8:.1f} s   speedup {t1/t8:.2f}x   (was 168.1 s, 5.00x)")
    print(f"  change: {100.0*(t8-168.1)/168.1:+.1f}% wall time")
    for key in ("per_rank_seconds", "rank_total_seconds", "rank_seconds"):
        pr = d.get(key)
        if pr:
            vals = [float(v) for v in (pr.values() if isinstance(pr, dict) else pr)]
            print(f"  rank spread {min(vals):.0f}-{max(vals):.0f} s   "
                  f"max/mean {max(vals)/(sum(vals)/len(vals)):.3f}   (was 133-168 s)")
            break
else:
    print(f"  MISSING {f}")
PY
