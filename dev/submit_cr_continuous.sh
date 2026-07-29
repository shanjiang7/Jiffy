#!/bin/bash
# Submit the continuous-hybrid validation chain (run from a LOGIN node):
#   1. stride-10 serial refs (bull/hybrid skip; continuous_hybrid x 2 tols run)
#   2. accuracy vs cuts: 8/16/32 ranks x 2 tolerances   [afterok refs]
#   3. tol1e4 refinement ladder, 10 iterations          [afterok refs]
set -euo pipefail
cd "$(dirname "$0")/.."

R=$(sbatch --parsable dev/run_cr_serial_refs_s10_sbatch.sh | tail -1)
echo "refs job:      ${R}"
A=$(sbatch --parsable -J cr_acc_ch --dependency=afterok:${R} \
      dev/run_cr_accuracy_cuts_s10_sbatch.sh continuous_hybrid | tail -1)
echo "accuracy job:  ${A} (afterok:${R})"
L=$(sbatch --parsable --dependency=afterok:${R} \
      dev/run_cr_ladder_continuous_sbatch.sh | tail -1)
echo "ladder job:    ${L} (afterok:${R})"
