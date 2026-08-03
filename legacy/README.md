# Legacy code

`multi_level_solver.py` is the original standalone multi-level HERMES
solver script that predates JIFFY's fused single-grid solver. Nothing in
the live package imports it; it is kept for provenance. It is the only
consumer of the `[grid.level1]`/`[grid.level2]` refinement parameters and
`cg_tol_level1`/`cg_tol_level2` beyond the CFL-derived-dt fallback (see
`src/hermes/runtime/setup.py`).
