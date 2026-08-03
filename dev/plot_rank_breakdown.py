#!/usr/bin/env python3
"""
Per-rank runtime breakdown at 64 ranks for the 15-layer Bull build:
optimized (exact_dp) vs uniform partitioning, stacked source-on solve +
correction (tracer) time per rank. Communication waits are <1.5% of the
runtime and are omitted (stated in the caption).

Data: outputs/cr_strong_scaling_ml15/bull/<planner>/parallel_64r/
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "outputs/cr_strong_scaling_ml15/bull"
BASE_C = "#9dc3e6"
CORR_C = "#c1272d"


def load(planner: str):
    d = json.load(open(f"{ROOT}/{planner}/parallel_64r/timing_summary.json"))
    rows = d["rank_timing_breakdown"]
    rows = rows if isinstance(rows, list) else list(rows.values())
    base = [r["base_solve_seconds"] for r in rows]
    corr = [r["tracer_solve_seconds"] + r["local_superpose_seconds"]
            for r in rows]
    return base, corr


def main() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.6, 7.6), dpi=250, sharex=True,
                             sharey=True)
    for ax, (planner, title) in zip(
            axes, (("exact_dp", "Optimized"), ("uniform", "Uniform"))):
        base, corr = load(planner)
        ranks = range(len(base))
        tot = [b + c for b, c in zip(base, corr)]
        mx, mean = max(tot), sum(tot) / len(tot)
        ax.bar(ranks, base, width=0.82, color=BASE_C, label="Source-on solve")
        ax.bar(ranks, corr, width=0.82, bottom=base, color=CORR_C,
               label="Correction")
        ax.axhline(mx, ls="--", color="k", lw=1.4)
        ax.text(63.6, mx + 4, f"max {mx:.0f} s   (max/mean = {mx/mean:.2f})",
                ha="right", fontsize=15)
        ax.set_title(f"{title} partitioning", fontsize=19, loc="left", pad=8)
        ax.set_ylabel("Busy time [s]", fontsize=17)
        ax.tick_params(labelsize=14)
        ax.set_xlim(-1, 64)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=15, loc="lower right", framealpha=0.95)
    axes[1].set_xlabel("Rank", fontsize=17)
    fig.tight_layout()
    out = f"{ROOT}/../rank_breakdown_64r.png"
    fig.savefig(out)
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
