#!/bin/bash
# Validate the per-edge correction horizon (commit bc66e55): each source traces
# only as deep as its own retained influence reaches into a destination, instead
# of the destination's full per-component depth.
#
# Unlike the fused-tracer change, this ALTERS numerics: it drops below-epsilon
# thermal tails the solver was previously superposing. So the acceptance test is
# NOT "identical" -- it is:
#   (a) observed rel-L2 rises from the old baseline but STAYS UNDER the tolerance
#       target (1e-4 / 1e-7). The truncation is keyed to the retained DAG, so
#       nothing above epsilon is dropped; error should land between the old
#       value and the target.
#         tol1e4  old max 2.3422e-06   target 1e-4
#         tol1e7  old max 2.2811e-08   target 1e-7
#       weight 0.75 is PINNED so the partition matches the baseline and the ONLY
#       difference is the horizon (the fix does not touch partitioning).
#   (b) load balance: Bull 8 ranks, h=18um, level_K=0.01K, weight 0.25,
#       timing-only. Fused-only was 145.3 s, max/mean 1.022 (job 853319).
#       Runtime cost now equals the model, so expect ~128-132 s and skew ~1.00.
#
# Usage:  sbatch dev/run_peredge_validation_sbatch.sh

#SBATCH -J peredge_val
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -o logs/peredge_val_%j.out
#SBATCH -e logs/peredge_val_%j.err
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

SIM_CONFIG=configs/examples/sim_calibration.ini   # h = 30 um accuracy grid
SNAP_EVERY=25

echo "###################################################################"
echo " Per-edge correction horizon validation"
echo "###################################################################"

# ---- (a) accuracy: expected to rise but stay under target ------------------
for TOL in tol1e4 tol1e7; do
  ROOT=outputs/accuracy_bull_${TOL}_h30
  CFG=configs/accuracy/bull_${TOL}.ini
  if [ ! -d "${ROOT}/serial/snapshots_ser" ]; then
    echo "ERROR: missing serial reference ${ROOT}/serial/snapshots_ser" >&2
    exit 1
  fi
  echo ""
  echo "###### [$(date)] (a) bull/${TOL}: 8 ranks, per-edge horizon ######"
  srun -N 8 -n 8 --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config "${SIM_CONFIG}" --path-config "${CFG}" \
      --dt-us 10 --snap-every-steps "${SNAP_EVERY}" \
      --planner-mode exact_dp --correction-weight 0.75 --no-export-dag \
      --out-dir "${ROOT}/par8_peredge"

  srun -N 1 -n 1 --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/compare_runs.py \
      --par-snap-dir "${ROOT}/par8_peredge/snapshots_par" \
      --ser-snap-dir "${ROOT}/serial/snapshots_ser" \
      --out-dir "${ROOT}/compare_peredge" \
      --source-on-only \
      --config "${SIM_CONFIG}" --path-config "${CFG}" --dt-us 10 | tail -4
done

# ---- (b) balance: Bull eps=0.1, weight 0.25 --------------------------------
echo ""
echo "###### [$(date)] (b) Bull 8 ranks, h=18um, level_K=0.01K, timing-only ######"
srun -N 8 -n 8 --ntasks-per-node=1 \
  python src/hermes/scripts/segment_correction/main.py \
    --config configs/examples/sim_ex1.ini \
    --path-config configs/examples/fast_heat.ini \
    --dt-us 10 --planner-mode exact_dp --correction-weight 0.25 \
    --timing-only --no-export-dag \
    --out-dir outputs/peredge_val/bull_8r

echo ""
echo "===== [$(date)] per-edge horizon validation ====="
python3 - <<'PY'
import json, pathlib

TARGET = {"tol1e4": 1e-4, "tol1e7": 1e-7}
OLD    = {"tol1e4": 2.3421788795e-06, "tol1e7": 2.2811036227e-08}
print("(a) accuracy -- must RISE from old baseline but stay UNDER target")
ok = True
for tol in ("tol1e4", "tol1e7"):
    f = pathlib.Path(f"outputs/accuracy_bull_{tol}_h30/compare_peredge/comparison_summary.json")
    if not f.exists():
        print(f"  bull/{tol}: MISSING {f}"); ok = False; continue
    d = json.loads(f.read_text())
    dmax = float(d["max_rel_l2"])
    under = dmax < TARGET[tol]
    ok = ok and under
    print(f"  bull/{tol}: max={dmax:.4e}  (old {OLD[tol]:.4e}, target {TARGET[tol]:.0e})  "
          f"factor x{dmax/OLD[tol]:.1f}  {'UNDER target' if under else 'OVER TARGET!'}")
print("  =>", "PASS (within spec)" if ok else "FAIL (exceeds tolerance)")

print()
print("(b) balance -- Bull 8 ranks, level_K=0.01K")
f = pathlib.Path("outputs/peredge_val/bull_8r/timing_summary.json")
if f.exists():
    d = json.loads(f.read_text())
    t8 = float(d["parallel_total_seconds"])
    print(f"  T(8) = {t8:.1f} s   speedup {841.5/t8:.2f}x   "
          f"(fused-only 145.3 s / 5.79x; pre-fusion 168.9 s / 4.98x)")
    br = d.get("rank_timing_breakdown", {})
    if br:
        vals = [float(v.get("rank_total_seconds", 0)) for v in br.values()]
        succ = {int(k): [int(x) for x in v] for k, v in d.get("component_successors", {}).items()}
        p = json.loads(pathlib.Path("outputs/peredge_val/bull_8r/planning_summary.json").read_text())
        assign = {int(k): sorted(int(x) for x in v) for k, v in p["rank_assignments"].items()}
        print(f"  {'rank':>4} {'total':>8} {'base':>8} {'tracer':>8} {'#succ':>6}")
        for r in sorted(br, key=lambda x: int(x)):
            b = br[r]; rr = int(r)
            ns = sum(len(succ.get(c, [])) for c in assign.get(rr, []))
            print(f"  {rr:>4} {float(b.get('rank_total_seconds',0)):>8.1f} "
                  f"{float(b.get('base_solve_seconds',0)):>8.1f} "
                  f"{float(b.get('tracer_solve_seconds',0)):>8.1f} {ns:>6}")
        mean = sum(vals) / len(vals)
        print(f"  rank spread {min(vals):.0f}-{max(vals):.0f} s   "
              f"max/mean {max(vals)/mean:.3f}   (fused-only was 128-145 s, 1.022)")
        print(f"  predicted skew (planner): {float(p.get('predicted_skew',0)):.3f}")
else:
    print(f"  MISSING {f}")
PY
