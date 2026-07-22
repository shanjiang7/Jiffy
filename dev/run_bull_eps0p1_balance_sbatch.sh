#!/bin/bash
# Bull, 8 ranks, correction weight 0.25, level_K = 0.1 K -- re-run WITH the
# fused bridge tracer (commit 34cbf67) to see whether the load is now balanced.
#
# Same configuration as job 853194 (which ran BEFORE the fusion):
#   T(8) = 168.9 s, 4.98x, rank spread 129-169 s -- imbalanced, because
#   multi-successor ranks re-traced a shared bridge prefix per successor.
# Fusion removes that duplication (Bull h=18um went 168.1 -> 143.5 s, job
# 853294). This job reports the per-rank breakdown so the balance is visible.

#SBATCH -J bull_bal
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:40:00
#SBATCH -o logs/bull_bal_%j.out
#SBATCH -e logs/bull_bal_%j.err
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

OUT=outputs/bull_eps0p1_balance/par8
echo "###### [$(date)] Bull 8 ranks, weight 0.25, level_K = 0.1 K (fused tracer) ######"
srun -N 8 -n 8 --ntasks-per-node=1 \
  python src/hermes/scripts/segment_correction/main.py \
    --config configs/examples/sim_ex1.ini \
    --path-config configs/dev/bull_eps0p1.ini \
    --dt-us 10 --planner-mode exact_dp --correction-weight 0.25 \
    --timing-only --no-export-dag \
    --out-dir "${OUT}"

echo ""
echo "===== [$(date)] per-rank balance ====="
python3 - <<'PY'
import json, pathlib
f = pathlib.Path("outputs/bull_eps0p1_balance/par8/timing_summary.json")
if not f.exists():
    print("MISSING", f); raise SystemExit(1)
d = json.loads(f.read_text())
t8 = float(d["parallel_total_seconds"])
br = d["rank_timing_breakdown"]
succ = {int(k): [int(x) for x in v] for k, v in d.get("component_successors", {}).items()}
p = json.loads(pathlib.Path("outputs/bull_eps0p1_balance/par8/planning_summary.json").read_text())
assign = {int(k): sorted(int(x) for x in v) for k, v in p["rank_assignments"].items()}

print(f"T(8) = {t8:.1f} s   speedup {841.5/t8:.2f}x   (job 853194 pre-fusion: 168.9 s, 4.98x)")
print()
print(f"{'rank':>4} {'total':>8} {'base':>8} {'tracer':>8} {'recv':>8} {'#succ':>6}")
totals = []
for r in sorted(br, key=lambda x: int(x)):
    b = br[r]; rr = int(r)
    ns = sum(len(succ.get(c, [])) for c in assign.get(rr, []))
    tot = float(b.get("rank_total_seconds", 0))
    totals.append(tot)
    print(f"{rr:>4} {tot:>8.1f} {float(b.get('base_solve_seconds',0)):>8.1f} "
          f"{float(b.get('tracer_solve_seconds',0)):>8.1f} "
          f"{float(b.get('recv_wait_seconds',0)):>8.1f} {ns:>6}")
mean = sum(totals) / len(totals)
print()
print(f"rank spread {min(totals):.0f}-{max(totals):.0f} s   "
      f"max/mean {max(totals)/mean:.3f}   (pre-fusion was 129-169 s)")
print(f"predicted skew (planner): {float(p.get('predicted_skew', 0)):.3f}")
PY
