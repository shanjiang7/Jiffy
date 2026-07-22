#!/bin/bash
# Bull, 8 ranks, correction weight 0.25, level_K = 5 K (loosest threshold) --
# strong-scaling speedup WITH fusion + per-edge horizon (commits 34cbf67,
# bc66e55).
#
# T(1) is independent of level_K: a single rank has no cut boundaries, hence no
# corrections. Prior probes confirmed T(1) = 841.5 s at both 0.1 K and 0.01 K
# (jobs 853194, 853319), so only the 8-rank run is needed here.
#
# Context (8-rank Bull, h=18um, weight 0.25, post-fix):
#   level_K = 0.01 K : 132.1 s, 6.37x   (job 853360)
# Looser epsilon => shallower cuts => fewer corrections, so expect >= 6.37x,
# limited ultimately by base-work balance (ideal T(8) ~ 841.5/8 = 105 s).

#SBATCH -J bull_eps5
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:30:00
#SBATCH -o logs/bull_eps5_%j.out
#SBATCH -e logs/bull_eps5_%j.err
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

OUT=outputs/bull_eps5_speedup/par8
echo "###### [$(date)] Bull 8 ranks, weight 0.25, level_K = 5 K ######"
srun -N 8 -n 8 --ntasks-per-node=1 \
  python src/hermes/scripts/segment_correction/main.py \
    --config configs/examples/sim_ex1.ini \
    --path-config configs/dev/bull_eps5.ini \
    --dt-us 10 --planner-mode exact_dp --correction-weight 0.25 \
    --timing-only --no-export-dag \
    --out-dir "${OUT}"

echo ""
echo "===== [$(date)] Bull level_K = 5 K speedup ====="
python3 - <<'PY'
import json, pathlib
f = pathlib.Path("outputs/bull_eps5_speedup/par8/timing_summary.json")
if not f.exists():
    print("MISSING", f); raise SystemExit(1)
d = json.loads(f.read_text())
t1 = 841.5
t8 = float(d["parallel_total_seconds"])
lk = float(d.get("dependency_level_K", 0))
succ = {int(k): [int(x) for x in v] for k, v in d.get("component_successors", {}).items()}
br = d["rank_timing_breakdown"]
p = json.loads(pathlib.Path("outputs/bull_eps5_speedup/par8/planning_summary.json").read_text())
assign = {int(k): sorted(int(x) for x in v) for k, v in p["rank_assignments"].items()}

print(f"level_K = {lk} K   global_max_cut_depth = {p.get('global_max_cut_depth','?')}")
print(f"T(1) = {t1:.1f} s   T(8) = {t8:.1f} s   speedup {t1/t8:.2f}x   "
      f"eff {100*t1/t8/8:.1f}%   (ideal T(8) = {t1/8:.1f} s)")
print(f"  (level_K=0.01: 132.1 s, 6.37x  |  0.1: 145.3->132 post-fix)")
print()
print(f"{'rank':>4} {'base':>8} {'correction':>11} {'total':>8} {'#succ':>6}")
vals = []
for r in sorted(br, key=lambda x: int(x)):
    b = br[r]; rr = int(r)
    tot = float(b.get("rank_total_seconds", 0)); vals.append(tot)
    ns = sum(len(succ.get(c, [])) for c in assign.get(rr, []))
    print(f"{rr:>4} {float(b.get('base_solve_seconds',0)):>8.1f} "
          f"{float(b.get('tracer_solve_seconds',0)):>11.1f} {tot:>8.1f} {ns:>6}")
mean = sum(vals) / len(vals)
print(f"\nrank spread {min(vals):.0f}-{max(vals):.0f} s   max/mean {max(vals)/mean:.3f}   "
      f"predicted skew {float(p.get('predicted_skew',0)):.3f}")
PY
