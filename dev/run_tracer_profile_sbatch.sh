#!/bin/bash
# Profile the fixed per-tracer overhead: Bull 8-rank runs at eps 0.2 K
# (spans ~90 SS) and eps 5 K (spans ~20 SS, fixed-cost dominated) with
# HERMES_TRACER_PROFILE=1. The per-rank tracer cost splits into
# setup / movement-cache builds / snapshot D2H; the remainder is the march.
# Timing runs proper must NOT use this flag (it adds sync points).

#SBATCH -J tracer_prof
#SBATCH -N 8
#SBATCH -n 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:45:00
#SBATCH -o logs/tracer_prof_%j.out
#SBATCH -e logs/tracer_prof_%j.err
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
export HERMES_TRACER_PROFILE=1

for SPEC in "eps0p2:configs/dev/bull_eps0p2.ini" "eps5:configs/dev/bull_eps5.ini"; do
  TAG=${SPEC%%:*}; CFG=${SPEC#*:}
  OUT=outputs/tracer_profile/${TAG}
  if [ -f "${OUT}/timing_summary.json" ]; then
    echo " [$(date)] ${TAG}: already done, skipping"
    continue
  fi
  echo "###### [$(date)] tracer profile: Bull 8 ranks, ${TAG} ######"
  srun -N 8 -n 8 --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config configs/examples/sim_ex1.ini \
      --path-config "${CFG}" \
      --dt-us 10 --planner-mode exact_dp --correction-weight 0.25 \
      --timing-only --no-export-dag \
      --out-dir "${OUT}"
done

echo ""
echo "===== [$(date)] tracer overhead breakdown ====="
python3 - <<'PY'
import json, pathlib
for tag in ("eps5", "eps0p2"):
    f = pathlib.Path(f"outputs/tracer_profile/{tag}/timing_summary.json")
    if not f.exists():
        print(f"{tag}: MISSING"); continue
    rb = json.loads(f.read_text())["rank_timing_breakdown"]
    tot = setup = cache = snap = runs = snaps = builds = 0.0
    for v in rb.values():
        if v.get("tracer_profile_runs", 0) == 0:
            continue
        tot += v["tracer_solve_seconds"]; setup += v["tracer_profile_setup_seconds"]
        cache += v["tracer_profile_cache_seconds"]; snap += v["tracer_profile_snap_d2h_seconds"]
        runs += v["tracer_profile_runs"]; snaps += v["tracer_profile_snapshots"]
        builds += v["tracer_profile_cache_builds"]
    march = tot - setup - cache - snap
    print(f"\n{tag}: {runs:.0f} tracers, {snaps:.0f} snapshots, {builds:.0f} cache builds")
    print(f"  per tracer: total={tot/runs:.3f}s  setup={setup/runs:.3f}s  "
          f"cache={cache/runs:.3f}s  snapD2H={snap/runs:.3f}s  march={march/runs:.3f}s")
    print(f"  shares: setup {setup/tot:.1%}  cache {cache/tot:.1%}  "
          f"snapD2H {snap/tot:.1%}  march {march/tot:.1%}")
PY
