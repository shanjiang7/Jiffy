#!/usr/bin/env python3
"""
Detailed DP (exact_dp) vs uniform comparison from a pair of timing-only runs.

Reads <root>/<tag>_exact_dp and <root>/<tag>_uniform (timing_summary.json +
planning_summary.json) and reports:
  - makespan and speedup ratio
  - per-rank side-by-side: base / correction / recv-wait / total
  - where each planner put the partition boundaries, with the model's charged
    correction span at each boundary (the quantity DP minimizes)
  - predicted vs measured skew for both planners

Usage:
    python dev/compare_dp_uniform.py --root outputs/dp_vs_uniform --tag eps0p01
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(root: Path, tag: str, mode: str):
    d = root / f"{tag}_{mode}"
    t = json.loads((d / "timing_summary.json").read_text())
    p = json.loads((d / "planning_summary.json").read_text())
    return t, p


def rank_rows(t: dict, p: dict) -> dict[int, dict]:
    br = t["rank_timing_breakdown"]
    succ = {int(k): [int(x) for x in v] for k, v in t.get("component_successors", {}).items()}
    assign = {int(k): sorted(int(x) for x in v) for k, v in p["rank_assignments"].items()}
    rows = {}
    for r in sorted(br, key=lambda x: int(x)):
        b, rr = br[r], int(r)
        rows[rr] = {
            "base": float(b.get("base_solve_seconds", 0)),
            "corr": float(b.get("tracer_solve_seconds", 0)),
            "recv": float(b.get("recv_wait_seconds", 0)),
            "total": float(b.get("rank_total_seconds", 0)),
            "n_succ": sum(len(succ.get(c, [])) for c in assign.get(rr, [])),
        }
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--root", default="outputs/dp_vs_uniform")
    ap.add_argument("--tag", default="eps0p01")
    args = ap.parse_args()
    root = Path(args.root)

    tu, pu = load(root, args.tag, "uniform")
    td, pd_ = load(root, args.tag, "exact_dp")

    Tu = float(tu["parallel_total_seconds"])
    Td = float(td["parallel_total_seconds"])
    print(f"===== {args.tag}: uniform vs exact_dp (Bull, 8 ranks) =====")
    print(f"  makespan: uniform {Tu:.1f} s   exact_dp {Td:.1f} s   "
          f"ratio {Tu/Td:.3f}x  (DP {100*(Tu/Td-1):+.1f}%)")
    print()

    ru, rd = rank_rows(tu, pu), rank_rows(td, pd_)
    print(f"  {'':>4} {'---------- uniform ----------':^37} {'---------- exact_dp ---------':^37}")
    print(f"  {'rank':>4} {'base':>8} {'corr':>7} {'recv':>7} {'total':>7} {'#s':>3}"
          f"   {'base':>8} {'corr':>7} {'recv':>7} {'total':>7} {'#s':>3}")
    for r in sorted(ru):
        u, d = ru[r], rd.get(r, {})
        print(f"  {r:>4} {u['base']:>8.1f} {u['corr']:>7.1f} {u['recv']:>7.1f} "
              f"{u['total']:>7.1f} {u['n_succ']:>3}"
              f"   {d.get('base',0):>8.1f} {d.get('corr',0):>7.1f} {d.get('recv',0):>7.1f} "
              f"{d.get('total',0):>7.1f} {d.get('n_succ',0):>3}")

    for label, rows in (("uniform", ru), ("exact_dp", rd)):
        vals = [v["total"] for v in rows.values()]
        mean = sum(vals) / len(vals)
        corr = sum(v["corr"] for v in rows.values())
        print(f"\n  {label}: rank spread {min(vals):.0f}-{max(vals):.0f} s   "
              f"max/mean {max(vals)/mean:.3f}   total correction {corr:.1f} s")

    print(f"\n  predicted skew: uniform {float(pu.get('predicted_skew', 0)):.3f}   "
          f"exact_dp {float(pd_.get('predicted_skew', 0)):.3f}")

    # Boundary placement: where each planner cut, and what the model charged.
    print(f"\n  boundary placement (rank -> [start_ss, end_ss], charged correction span):")
    for label, p in (("uniform", pu), ("exact_dp", pd_)):
        pt = p.get("partition", {})
        ranges = pt.get("rank_partition_ranges", {})
        spans = pt.get("rank_boundary_correction_span_ss", {})
        parts = []
        for r in sorted(ranges, key=int):
            g = ranges[r]
            parts.append(f"r{r}:[{g['start_ss']}-{g['end_ss']}]/{spans.get(r, 0):.0f}")
        print(f"    {label:>9}: " + "  ".join(parts))


if __name__ == "__main__":
    main()
