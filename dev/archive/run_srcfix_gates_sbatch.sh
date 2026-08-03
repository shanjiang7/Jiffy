#!/bin/bash
# Re-gate the source-charging fix (range pays its own outgoing march):
# DP runs only (uniform partitions unchanged by cost model), compared against
# the stored dp_new/uniform results. Bull eps0.2+0.01 @8r, hybrid @8r,
# then Bull and hybrid @32/64 — all in one 64-node allocation.

#SBATCH -J srcfix
#SBATCH -N 64
#SBATCH -n 64
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:30:00
#SBATCH -o logs/srcfix_%j.out
#SBATCH -e logs/srcfix_%j.err
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

run_case () {
  local R=$1 CFG=$2 OUT=$3
  if [ -f "${OUT}/timing_summary.json" ]; then
    echo " [$(date)] ${OUT}: done, skipping"; return
  fi
  echo "###### [$(date)] srcfix R=${R} ${CFG} -> ${OUT} ######"
  srun -N "${R}" -n "${R}" --ntasks-per-node=1 \
    python src/hermes/scripts/segment_correction/main.py \
      --config configs/examples/sim_ex1.ini \
      --path-config "${CFG}" \
      --dt-us 10 --planner-mode exact_dp --correction-weight 0.21 \
      --timing-only --no-export-dag \
      --out-dir "${OUT}"
}

BULL=configs/examples/fast_heat.ini
HYB=configs/examples/hybrid_spiral_raster.ini

run_case 64 "${BULL}" outputs/gate3264/r64_dp_srcfix
run_case 64 "${HYB}"  outputs/gate_hybrid/r64_dp_srcfix
run_case 32 "${BULL}" outputs/gate3264/r32_dp_srcfix
run_case 32 "${HYB}"  outputs/gate_hybrid/r32_dp_srcfix
run_case 8  "${BULL}" outputs/gate8/eps0p01_dp_srcfix
run_case 8  configs/dev/bull_eps0p2.ini outputs/gate8/eps0p2_dp_srcfix
run_case 8  "${HYB}"  outputs/gate_hybrid/r8_dp_srcfix

echo ""
echo "===== [$(date)] source-charging fix vs previous ====="
python3 - <<'PY'
import json, pathlib
def stats(p):
    f = pathlib.Path(f"outputs/{p}/timing_summary.json")
    if not f.exists(): return None
    d = json.loads(f.read_text())
    rb = d["rank_timing_breakdown"]
    busy = [v["base_solve_seconds"]+v["tracer_solve_seconds"]+v["local_superpose_seconds"] for v in rb.values()]
    return d["parallel_total_seconds"], max(busy)/(sum(busy)/len(busy)), max(busy)/min(busy)
CASES = [
    ("Bull 8r e0.2",  "gate8/eps0p2_uniform",  "gate8/eps0p2_dp_new",  "gate8/eps0p2_dp_srcfix"),
    ("Bull 8r e0.01", "gate8/eps0p01_uniform", "gate8/eps0p01_dp_new", "gate8/eps0p01_dp_srcfix"),
    ("Hyb  8r",       "gate_hybrid/r8_uniform","gate_hybrid/r8_dp_new","gate_hybrid/r8_dp_srcfix"),
    ("Bull 32r",      "gate3264/r32_uniform",  "gate3264/r32_dp_new",  "gate3264/r32_dp_srcfix"),
    ("Hyb  32r",      "gate_hybrid/r32_uniform","gate_hybrid/r32_dp_new","gate_hybrid/r32_dp_srcfix"),
    ("Bull 64r",      "gate3264/r64_uniform",  "gate3264/r64_dp_new",  "gate3264/r64_dp_srcfix"),
    ("Hyb  64r",      "gate_hybrid/r64_uniform","gate_hybrid/r64_dp_new","gate_hybrid/r64_dp_srcfix"),
]
print(f"{'case':14s} {'uniform':>9} {'dp_new':>16} {'dp_srcfix':>16}")
for name, u, n, s in CASES:
    su, sn, ss = stats(u), stats(n), stats(s)
    fu = f"{su[0]:7.1f}s" if su else "   --  "
    fn = f"{sn[0]:7.1f}s k={sn[1]:.3f}" if sn else "      --       "
    fs = f"{ss[0]:7.1f}s k={ss[1]:.3f}" if ss else "      --       "
    print(f"{name:14s} {fu:>9} {fn:>16} {fs:>16}")
PY
