#!/usr/bin/env python3
"""
Collect strong-scaling timing runs into a speedup table (and optional plot).

Reads every `<root>/<planner>/parallel_<N>r/timing_summary.json` written by
the timing-only runs, computes speedup and parallel efficiency against the
1-rank baseline of the same planner mode, prints a table, and writes a CSV
next to the run root.

Usage:
    python scripts/scaling/collect_scaling.py --root outputs/strong_scaling_h18/bull
    python scripts/scaling/collect_scaling.py --root outputs/strong_scaling_h18 --all --plot
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_runs(root: Path) -> dict[str, dict[int, float]]:
    """{planner_mode: {n_ranks: seconds}} for one path's run root."""
    runs: dict[str, dict[int, float]] = {}
    for summary in sorted(root.glob("*/parallel_*r/timing_summary.json")):
        mode = summary.parent.parent.name
        try:
            data = json.loads(summary.read_text())
            n = int(data["num_ranks"])
            seconds = float(data["parallel_total_seconds"])
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            print(f"  [warn] skipping {summary}: {exc}")
            continue
        runs.setdefault(mode, {})[n] = seconds
    return runs


def rows_for(label: str, runs: dict[str, dict[int, float]]) -> list[dict]:
    rows = []
    for mode in sorted(runs):
        by_rank = runs[mode]
        baseline = by_rank.get(1)
        for n in sorted(by_rank):
            seconds = by_rank[n]
            speedup = (baseline / seconds) if baseline else float("nan")
            rows.append(
                {
                    "path": label,
                    "planner": mode,
                    "ranks": n,
                    "seconds": round(seconds, 3),
                    "speedup": round(speedup, 3),
                    "efficiency": round(speedup / n, 3),
                }
            )
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("  (no completed runs found)")
        return
    print(f"  {'path':<10} {'planner':<10} {'ranks':>5} {'seconds':>10} {'speedup':>8} {'eff':>7}")
    for r in rows:
        speedup = "n/a" if r["speedup"] != r["speedup"] else f"{r['speedup']:.2f}"
        eff = "n/a" if r["efficiency"] != r["efficiency"] else f"{r['efficiency']:.2f}"
        print(
            f"  {r['path']:<10} {r['planner']:<10} {r['ranks']:>5} "
            f"{r['seconds']:>10.1f} {speedup:>8} {eff:>7}"
        )


def maybe_plot(all_rows: list[dict], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib unavailable; skipping plot")
        return

    paths = sorted({r["path"] for r in all_rows})
    fig, axs = plt.subplots(1, len(paths), figsize=(4 * len(paths), 3.6), squeeze=False)
    for ax, path in zip(axs[0], paths):
        for mode in sorted({r["planner"] for r in all_rows if r["path"] == path}):
            pts = sorted(
                (r["ranks"], r["speedup"])
                for r in all_rows
                if r["path"] == path and r["planner"] == mode and r["speedup"] == r["speedup"]
            )
            if pts:
                ax.plot(*zip(*pts), marker="o", label=mode)
        max_rank = max((r["ranks"] for r in all_rows if r["path"] == path), default=8)
        ax.plot([1, max_rank], [1, max_rank], "k--", lw=0.8, label="ideal")
        ax.set_title(path)
        ax.set_xlabel("ranks")
        ax.set_ylabel("speedup")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"  [ok] plot: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--root", required=True, help="Run root (one path), or the parent with --all")
    p.add_argument("--label", default=None, help="Path label for single-root mode")
    p.add_argument("--all", action="store_true", help="Treat --root as the parent of per-path roots")
    p.add_argument("--plot", action="store_true", help="Also write a speedup plot")
    args = p.parse_args()

    root = Path(args.root)
    roots = sorted(d for d in root.iterdir() if d.is_dir()) if args.all else [root]

    all_rows: list[dict] = []
    for r in roots:
        label = args.label or r.name
        all_rows.extend(rows_for(label, load_runs(r)))

    print_table(all_rows)
    if all_rows:
        csv_path = root / "scaling_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  [ok] csv: {csv_path}")
        if args.plot:
            maybe_plot(all_rows, root / "strong_scaling.png")


if __name__ == "__main__":
    main()
