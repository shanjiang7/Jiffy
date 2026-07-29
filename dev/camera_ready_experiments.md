# Camera-ready experiment campaign

All runs use the FINAL default logic: source-charged span (layer-clamped),
w = 0.21, a0 = 7.9 SS (`--correction-weight 0.21`, code defaults). Protocols
learned from the model campaign apply everywhere:
- uniform + DP pairs run back-to-back in the SAME allocation (kills node-draw
  offsets);
- imbalance metric = busy-time max/mean (skew); waits shown as their own
  segment in breakdown figures;
- runs are resumable (skip on existing timing_summary.json / comparison);
- every campaign job script sanitizes the model env before running:
  `unset HERMES_CORRECTION_FIXED_COST_SS HERMES_TRACER_PROFILE`
  (sbatch inherits the submit shell's env; stale overrides would silently
  change the model). planning_summary.json now records
  correction_fixed_cost_ss for per-run provenance.
Step 0 (commit the model + instrumentation + scripts tree) DONE 2026-07-29:
commits 54d4fd7 / 6d62288 / this one.

## 1. Strong scaling, 1-8 ranks, four paths            [paper Fig 9 analog]
- Paths: bull, texas, spiral-raster, hilbert (h = 18 um, sim_ex1).
- eps = 0.01 K (1e-7 config; fixes the submitted version's eps=0.1K+AABB
  provenance — camera-ready numbers must be artifact-reproducible).
- Ranks 1..8 x {uniform, exact_dp}. Speedup vs the 1-rank baseline
  (eps-independent). Reuse existing baselines where configs match
  (bull: outputs/eps_probe/bull_1r, 841.5 s); rerun missing ones
  (texas/hybrid/hilbert) once each.
- Figure: per path, two lines (blue = DP, orange = uniform) + ideal.

## 2. Strong scaling, 8-64 ranks, Bull                  [new in camera-ready]
Two problem sizes, ranks 8,16,24,...,64 x {uniform, exact_dp}:
- 2a. Single-layer Bull (1,678 SS): shows the starvation regime honestly
  (corrections dominate; speedup saturates ~16-17x).
- 2b. Multi-layer Bull (16 layers, ~26.8k SS; DECIDED 2026-07-29): strong
  scaling headroom well past 64 ranks (per-rank work at 64 ranks = 2x the
  single-layer 8-rank granularity). Needs one 1-rank baseline on the
  multi-layer problem (~3.7 h, single run). Sweep cost note: the 8-rank
  16-layer point runs ~35 min; full 8..64 sweep both modes ~3-3.5 h.
- Layer-clamped span logic already in the planner (validated in unit test).
- Figure: speedup curves to 64; the single-vs-multi-layer contrast IS the
  message (what large problems buy at scale).

## 3. Runtime breakdowns (figC style)                   [no new jobs]
- 8 + 64 ranks, single-layer Bull and multi-layer Bull, uniform vs DP.
- Rendered from the timing summaries of #2's same-allocation pairs via
  dev/plot_meeting_figs.py fig_c_scaled (already path/dataset-generic).

## 4. Accuracy vs cut count                             [Tables IV/V extension]
- Paths: bull, spiral-raster; tolerances: 1e-4, 1e-7 (h = 30 um accuracy
  configs); cuts: 7/15/31 (ranks 8/16/32).
- --snap-every-steps 10 (finer error sampling than the previous 25).
  PREREQUISITE: regenerate the four serial references at stride 10
  (bull/hybrid x tol1e4/tol1e7, ~1-1.5 h each, single GPU, once).
- 2 paths x 2 tols x 3 cut counts = 12 parallel runs + compares; delete
  parallel snapshots after each comparison (disk).
- Table: max/mean rel-L2 vs target per (path, tol, cuts) — shows how the
  observed error approaches the target as cuts increase; pairs with the
  self-convergence appendix for the above-target discussion.

## 5. Weak scaling to 64 ranks, two tolerances          [new axis: eps]
- Paths: bull (extent x sqrt(P)), spiral-raster (motif x P), configs in
  configs/weak_scaling/ (extend to P = 64 if only smaller exist).
- P in {1, 2, 4, 8, 16, 32, 64}; tolerances 1e-4 and 1e-7 per path.
- Partitioner: exact_dp only (weak-scaling figure has 4 curves already:
  2 paths x 2 tols; uniform would double runs for a secondary point).
- Metric: efficiency T(1)/T(P); the eps-dependence of the efficiency floor
  is the new result (loose target ~flat, tight target decays with the
  correction share).

## Execution order and jobs
0. Commit pending tree; optional: hybrid 32-rank accuracy re-check under the
   new model (same pattern as job 873745).
1. Serial prerequisites (queue-light, start first):
   - 4 accuracy serial refs at stride 10 (one 4-GPU job, ~1.5 h).
   - Multi-layer Bull (16 layers) 1-rank baseline (~3.7 h).
   - Missing 1-8 baselines: texas/hybrid/hilbert 1-rank (~0.5-1 h each).
2. #1 as four jobs (one per path, 8 nodes; pattern:
   scripts/scaling/run_strong_scaling_sbatch.sh updated to eps 0.01 + w 0.21).
3. #2 as two 64-node jobs (single-layer, multi-layer), both modes interleaved.
4. #4 as one 8-node job (32 ranks max via 4 ranks/GPU; pattern: job 873745).
5. #5 as two jobs (one per path; largest weak problem sets node count).
6. Render all figures; write results tables.

## Decisions (settled 2026-07-29)
- Multi-layer count: 16 layers.
- Accuracy snapshots: stride 10 (regenerate 4 serial refs first).
- Weak scaling: exact_dp only.
- Repetitions: single runs (eps 0.01 differences >> noise); median-of-3
  only if any headline pair lands within ~2%.
