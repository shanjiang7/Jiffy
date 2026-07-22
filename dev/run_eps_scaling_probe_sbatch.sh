#!/bin/bash
# Strong-scaling probe: how much does the retention threshold cost?
#   (a) Bull, 8 ranks, level_K = 0.1 K  (paper-era epsilon, current DAG code)
#   (b) Bull, 1 rank                    (baseline; at 1 rank there are no cuts,
#                                        hence no corrections, so T(1) is
#                                        epsilon-independent)
# Compare against the measured level_K = 0.01 K result (job 853164: 168.1 s).
#
# Usage: sbatch dev/run_eps_scaling_probe_sbatch.sh

#SBATCH -J eps_probe
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:30:00
#SBATCH -o logs/eps_probe_%j.out
#SBATCH -e logs/eps_probe_%j.err
#SBATCH -p gh
#SBATCH -A ASC21034

set -uo pipefail
PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
while [ ! -f "${PROJECT_DIR}/env_vista.sh" ] && [ "${PROJECT_DIR}" != "/" ]; do
  PROJECT_DIR=$(dirname "${PROJECT_DIR}")
done
cd "${PROJECT_DIR}"; mkdir -p logs outputs
source "${PROJECT_DIR}/env_vista.sh"

SIM=configs/examples/sim_ex1.ini
COMMON="--dt-us 10 --planner-mode exact_dp --correction-weight 0.25 --timing-only --no-export-dag"

echo "###### [$(date)] (a) Bull, 8 ranks, level_K = 0.1 K ######"
srun -N 8 -n 8 --ntasks-per-node=1 python src/hermes/scripts/segment_correction/main.py \
  --config "${SIM}" --path-config configs/dev/bull_eps0p1.ini ${COMMON} \
  --out-dir outputs/eps_probe/bull_eps0p1_8r

echo "###### [$(date)] (b) Bull, 1 rank (epsilon-independent baseline) ######"
srun -N 1 -n 1 --ntasks-per-node=1 python src/hermes/scripts/segment_correction/main.py \
  --config "${SIM}" --path-config configs/examples/fast_heat.ini ${COMMON} \
  --out-dir outputs/eps_probe/bull_1r

echo ""
echo "[$(date)] ===== strong-scaling probe: effect of the retention threshold ====="
python3 - <<'PYEOF'
import json, os
def get(p):
    try: return json.load(open(p))
    except Exception: return None
one  = get('outputs/eps_probe/bull_1r/timing_summary.json')
e01  = get('outputs/eps_probe/bull_eps0p1_8r/timing_summary.json')
e001 = get('outputs/strong_scaling_h18/bull/exact_dp/parallel_8r/timing_summary.json')
t1 = float(one['parallel_total_seconds']) if one else None
print(f"  T(1)                        = {t1:.1f} s" if t1 else "  T(1) missing")
for tag, d in (("level_K = 0.1  (8 ranks)", e01), ("level_K = 0.01 (8 ranks)", e001)):
    if not d: continue
    t8 = float(d['parallel_total_seconds']); rb = d['rank_timing_breakdown']
    base = sum(float(v['base_solve_seconds']) for v in rb.values())
    trac = sum(float(v['tracer_solve_seconds']) for v in rb.values())
    idle = sum(float(v['recv_wait_seconds'])+float(v['send_wait_seconds']) for v in rb.values())
    tots = [float(v['rank_total_seconds']) for v in rb.values()]
    sp = (t1/t8) if t1 else float('nan')
    print(f"  {tag}: T(8) = {t8:6.1f} s   speedup {sp:4.2f}x   eff {100*sp/8:4.1f}%")
    print(f"      corrections {100*trac/(base+trac):4.1f}% of compute   idle {idle:5.1f} s   rank spread {min(tots):.0f}-{max(tots):.0f} s")
PYEOF
