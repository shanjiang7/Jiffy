# SC26 AD/AE timeline and Chameleon plan (internal notes)

Working notes for the artifact-evaluation process. Not part of the artifact
surface; remove before the repository is frozen for the DOI if desired.

## Timeline (SC26 Reproducibility Initiative)

| Date        | Milestone                                          |
|-------------|----------------------------------------------------|
| 24 Jul 2026 | Applications for badges (**mandatory**)            |
| 24 Jul 2026 | Submission of AE appendix (optional but submitted) |
| 31 Jul 2026 | Committee feedback on completeness of AD/AE        |
| 7 Aug 2026  | Submission of revised AD/AE appendix               |
| 25 Aug 2026 | Artifact frozen with permanent DOI (Zenodo)        |
| 25 Sep 2026 | Final badge decisions                              |

## How the evaluation works

- Two phases: (1) completeness review of the AD/AE appendices (the
  Jul 31 → Aug 7 loop); (2) hands-on evaluation — reviewers actually
  install and run the artifact, with an **8-hour review budget** per
  artifact.
- ACM badges (incremental):
  1. **Artifacts Available** — code archived with a permanent DOI
     (freeze by Aug 25; Zenodo snapshot of this repo).
  2. **Artifacts Evaluated–Functional** — documented, complete, runs.
  3. **Results Reproduced** — key results recreated; explicitly not
     bitwise; downscaled experiments are acceptable within the 8-hour
     budget.

## Chameleon Cloud (reviewer platform)

- Reviewers evaluate on **Chameleon Cloud** (NSF bare-metal testbed,
  UChicago + TACC). Authors were invited to the shared project
  **CHI-261657 "SC'26 Authors Artifact Evaluations"** (invitation from
  no-reply@chameleoncloud.org — legitimate; accept it) so authors and
  reviewers use identical infrastructure.
- Relevant GPU hardware: 4× A100 nodes (PCIe + one SXM/NVLink) at
  CHI@UC; composable up-to-8× A100 single node at CHI@UC; two
  bare-metal 4× H100 nodes at CHI@TACC; older V100 / RTX 6000 / MI100.
- Differences from TACC Vista to plan for:
  - **No Slurm / idev / modules** — bare-metal Ubuntu with root; launch
    with `mpirun -np N` directly (the example README already notes the
    srun → mpirun substitution).
  - **Multiple GPUs per node** instead of one per node —
    `bind_local_gpu()` maps node-local ranks to distinct GPUs, so 4–8
    ranks on one fat node should work; untested off Vista as of
    2026-07-23.
  - GPU memory 40–80 GB (A100/H100) vs 96 GB GH200 — accuracy (h = 30 µm)
    configs are the safe ones; the h = 18 µm scaling grid may not fit.

## Action plan

1. Accept the CHI-261657 invitation.
2. Submit badge applications + AE appendix by Jul 24.
3. Next week: lease a 4× A100 or H100 node on Chameleon, run
   `INSTALL.md` + `examples/straight_line` + a 4-rank accuracy case on
   reviewer hardware; fold any friction into the Aug 7 revision.
4. Zenodo DOI + repository freeze before Aug 25 (also replaces the
   "DOI: to be created after acceptance" placeholder in the AD).
