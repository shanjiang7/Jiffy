#!/usr/bin/env python3
"""Figures for the advisor meeting (2026-07-28).

  A  strong scaling, ranks 1-8, one curve per error target (eps ladder)
  B  correction-time fraction at 8 ranks vs error target
  C  per-rank time breakdown, uniform vs exact-DP, eps = 0.01 K

Reads existing timing_summary.json files only; figures whose data is
incomplete are skipped with a note. Run from the repo root:

    python dev/plot_meeting_figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- palette (validated 4-slot categorical, light surface) --------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

OUT = Path("outputs/meeting_figs")
OUT.mkdir(parents=True, exist_ok=True)

EPS = [
    ("eps5", "5", "1e-4", S1),
    ("eps0p2", "0.2", "1e-5", S2),
    ("eps0p03", "0.03", "1e-6", S3),
    ("eps0p01", "0.01", "1e-7", S4),
]
T1_PATH = Path("outputs/eps_probe/bull_1r/timing_summary.json")
SWEEP = Path("outputs/eps_strong_scaling")


def load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return None


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def tracer_fraction(d: dict) -> float:
    rb = d["rank_timing_breakdown"]
    tr = sum(v["tracer_solve_seconds"] for v in rb.values())
    tot = sum(v["rank_total_seconds"] for v in rb.values())
    return tr / tot


# -- Figure A: strong scaling curves -----------------------------------------
def fig_a() -> None:
    t1 = load(T1_PATH)
    if t1 is None:
        print("A: missing 1-rank baseline, skipped")
        return
    T1 = t1["parallel_total_seconds"]
    ranks = list(range(2, 9))
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    ax.plot([1, 8], [1, 8], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
            zorder=1)
    ax.annotate("ideal", (7.55, 7.78), color=MUTED, fontsize=9, ha="right")

    plotted = 0
    for tag, eps, target, color in EPS:
        if tag == "eps0p03":
            continue  # 1e-6 omitted: three thresholds keep the figure readable
        xs, ys = [1], [1.0]
        for r in ranks:
            d = load(SWEEP / f"{tag}_r{r}" / "timing_summary.json")
            if d is not None:
                xs.append(r)
                ys.append(T1 / d["parallel_total_seconds"])
        if len(xs) < 3:
            print(f"A: {tag} has {len(xs)-1} points, curve skipped")
            continue
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                markerfacecolor=color, markeredgecolor=SURFACE,
                markeredgewidth=1, zorder=3,
                label=f"target {target} (ε = {eps} K)")
        # per-point speedup labels (paper style); offsets keep the three
        # series' numbers apart where the curves converge at low ranks
        offset = {"eps5": (0, 8), "eps0p2": (10, -12), "eps0p01": (-12, -12)}[tag]
        for x, y in zip(xs[1:], ys[1:]):
            off = offset
            if tag == "eps0p2" and x == xs[-1]:
                off = (-6, 8)  # keep clear of the curve-name labels at right
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=off, ha="center", fontsize=7.5,
                        color=color, zorder=4)
        plotted += 1
    if plotted == 0:
        print("A: no sweep data yet, figure not written")
        plt.close(fig)
        return

    ax.set_xlim(0.8, 8.7)
    ax.set_ylim(0.8, 8.3)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("MPI ranks (one GPU each)", color=INK2, fontsize=10)
    ax.set_ylabel("speedup vs 1 rank", color=INK2, fontsize=10)
    ax.set_title("Strong scaling vs accuracy threshold — Bull path, h = 18 µm",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figA_strong_scaling_vs_eps.{ext}",
                    facecolor=SURFACE)
    plt.close(fig)
    print(f"A: written with {plotted} curves")


# -- Figure A16: strong scaling to 16 ranks (3 thresholds) -------------------
SWEEP16 = Path("outputs/eps_strong_scaling16")


def fig_a16() -> None:
    t1 = load(T1_PATH)
    if t1 is None:
        print("A16: missing 1-rank baseline, skipped")
        return
    T1 = t1["parallel_total_seconds"]
    ranks = list(range(2, 17, 2))
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    ax.plot([1, 16], [1, 16], color=MUTED, linewidth=1.2,
            linestyle=(0, (4, 3)), zorder=1)
    ax.annotate("ideal", (15.2, 15.6), color=MUTED, fontsize=9, ha="right")

    plotted = 0
    for tag, eps, target, color in EPS:
        if tag == "eps0p03":
            continue  # not sampled in the 16-rank sweep
        xs, ys = [1], [1.0]
        for r in ranks:
            d = load(SWEEP16 / f"{tag}_r{r}" / "timing_summary.json")
            if d is not None:
                xs.append(r)
                ys.append(T1 / d["parallel_total_seconds"])
        if len(xs) < 3:
            print(f"A16: {tag} has {len(xs)-1} points, curve skipped")
            continue
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                markerfacecolor=color, markeredgecolor=SURFACE,
                markeredgewidth=1, zorder=3,
                label=f"target {target} (ε = {eps} K)")
        ax.annotate(f"{ys[-1]:.2f}", (xs[-1] + 0.3, ys[-1]), color=color,
                    fontsize=9, va="center", fontweight="bold")
        plotted += 1
    if plotted == 0:
        print("A16: no sweep data yet, figure not written")
        plt.close(fig)
        return

    ax.set_xlim(0.5, 17.4)
    ax.set_ylim(0.5, 16.6)
    ax.set_xticks([1] + ranks)
    ax.set_xlabel("MPI ranks (one GPU each)", color=INK2, fontsize=10)
    ax.set_ylabel("speedup vs 1 rank", color=INK2, fontsize=10)
    ax.set_title("Strong scaling to 16 ranks vs accuracy threshold — Bull, h = 18 µm",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figA16_strong_scaling_vs_eps.{ext}",
                    facecolor=SURFACE)
    plt.close(fig)
    print(f"A16: written with {plotted} curves")


# -- Figure B: correction fraction vs threshold ------------------------------
def fig_b() -> None:
    labels, totals, corrs = [], [], []
    for tag, eps, target, _ in EPS:
        if tag == "eps0p03":
            continue  # match figure A's three thresholds
        d = load(SWEEP / f"{tag}_r8" / "timing_summary.json")
        if d is None:
            print(f"B: missing {tag} 8-rank point")
            continue
        T = d["parallel_total_seconds"]
        labels.append(f"{target}\nε = {eps} K")
        totals.append(T)
        corrs.append(tracer_fraction(d) * T)
    if len(totals) < 2:
        print("B: not enough data yet, figure not written")
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    xs = range(len(totals))
    rest = [t - c for t, c in zip(totals, corrs)]
    ax.bar(xs, rest, width=0.55, color=S1, edgecolor=SURFACE, linewidth=1.2,
           zorder=3, label="base solve + other")
    ax.bar(xs, corrs, bottom=rest, width=0.55, color=S2, edgecolor=SURFACE,
           linewidth=1.2, zorder=3, label="correction (source-off)")
    for i, (t, c, r) in enumerate(zip(totals, corrs, rest)):
        ax.annotate(f"{100 * c / t:.1f}%", (i + 0.33, r + c / 2),
                    ha="left", va="center", color=S2, fontsize=9,
                    fontweight="bold")
        ax.annotate(f"{t:.0f} s", (i, t), textcoords="offset points",
                    xytext=(0, 4), ha="center", color=INK2, fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, color=INK2, fontsize=9)
    ax.set_xlim(-0.6, len(totals) - 0.2)
    ax.set_ylabel("total runtime (s), 8 ranks", color=INK2, fontsize=10)
    ax.set_title("Correction share of total runtime — 8 ranks",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figB_correction_fraction.{ext}", facecolor=SURFACE)
    plt.close(fig)
    print(f"B: written with {len(totals)} bars")


# -- Figure C: per-rank breakdown, uniform vs DP at eps = 0.01 ---------------
def fig_c_scaled(ranks: int, root: str = "outputs/gate3264",
                 pathname: str = "Bull", prefix: str = "figC",
                 dp_dir: str | None = None, uniform_dir: str | None = None,
                 outdir: Path | None = None) -> None:
    """Two-panel per-rank breakdown, uniform vs DP, any path/dataset."""
    cases = [
        ("uniform", (uniform_dir or f"{root}/r{ranks}_uniform") + "/timing_summary.json"),
        ("exact-DP", (dp_dir or f"{root}/r{ranks}_dp_srcfix") + "/timing_summary.json"),
    ]
    outdir = outdir or OUT
    outdir.mkdir(parents=True, exist_ok=True)
    data = [(name, load(Path(p))) for name, p in cases]
    if any(d is None for _, d in data):
        print(f"{prefix}{ranks}: missing data under {root}, figure not written")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    comp = [("base solve", "base_solve_seconds", S1),
            ("correction (source-off)", "tracer_solve_seconds", S2),
            ("wait + other", None, S3)]
    for ax, (name, d) in zip(axes, data):
        style_axes(ax)
        rb = d["rank_timing_breakdown"]
        rlist = sorted(rb, key=int)
        bottoms = [0.0] * len(rlist)
        for label, key, color in comp:
            if key is not None:
                vals = [rb[r][key] for r in rlist]
            else:
                vals = [rb[r]["rank_total_seconds"]
                        - rb[r]["base_solve_seconds"]
                        - rb[r]["tracer_solve_seconds"] for r in rlist]
            ax.bar(range(len(rlist)), vals, bottom=bottoms, width=0.8,
                   color=color, edgecolor=SURFACE, linewidth=0.4, zorder=3,
                   label=label)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        wall = d["parallel_total_seconds"]
        ax.axhline(wall, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)),
                   zorder=4)
        ax.annotate(f"wall clock {wall:.1f} s", (0.02, wall),
                    xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(0, 4), color=INK2,
                    fontsize=9)
        step = max(1, ranks // 8)
        ax.set_xticks(range(0, len(rlist), step))
        ax.set_xticklabels(rlist[::step], color=INK2, fontsize=8)
        ax.set_xlabel("rank", color=INK2, fontsize=10)
        ax.set_title(name, color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("seconds", color=INK2, fontsize=10)
    ratio = data[0][1]["parallel_total_seconds"] / data[1][1]["parallel_total_seconds"]
    fig.suptitle(
        f"Partitioning at ε = 0.01 K, {pathname}, {ranks} ranks — "
        f"DP is {100*(ratio-1):.1f}% faster",
        color=INK, fontsize=11, x=0.02, ha="left")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK2,
               loc="upper right", bbox_to_anchor=(0.99, 0.97), ncol=3,
               columnspacing=1.2, handlelength=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{prefix}{ranks}_dp_vs_uniform_eps0p01.{ext}",
                    facecolor=SURFACE)
    plt.close(fig)
    print(f"C{ranks}: written")


def fig_c() -> None:
    cases = [
        ("uniform", "outputs/gate8/eps0p01_uniform/timing_summary.json"),
        ("exact-DP", "outputs/gate8/eps0p01_dp_srcfix/timing_summary.json"),
    ]
    data = [(name, load(Path(p))) for name, p in cases]
    if any(d is None for _, d in data):
        print("C: missing dp_vs_uniform data, figure not written")
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    comp = [("base solve", "base_solve_seconds", S1),
            ("correction (source-off)", "tracer_solve_seconds", S2),
            ("wait + other", None, S3)]
    for ax, (name, d) in zip(axes, data):
        style_axes(ax)
        rb = d["rank_timing_breakdown"]
        ranks = sorted(rb, key=int)
        bottoms = [0.0] * len(ranks)
        for label, key, color in comp:
            if key is not None:
                vals = [rb[r][key] for r in ranks]
            else:
                vals = [rb[r]["rank_total_seconds"]
                        - rb[r]["base_solve_seconds"]
                        - rb[r]["tracer_solve_seconds"] for r in ranks]
            ax.bar(range(len(ranks)), vals, bottom=bottoms, width=0.6,
                   color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                   label=label)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        wall = d["parallel_total_seconds"]
        ax.axhline(wall, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)),
                   zorder=4)
        ax.annotate(f"wall clock {wall:.1f} s", (0.02, wall), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(0, 4), color=INK2,
                    fontsize=9)
        ax.set_xticks(range(len(ranks)))
        ax.set_xticklabels(ranks, color=INK2, fontsize=9)
        ax.set_xlabel("rank", color=INK2, fontsize=10)
        ax.set_title(name, color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("seconds", color=INK2, fontsize=10)
    ratio = data[0][1]["parallel_total_seconds"] / data[1][1]["parallel_total_seconds"]
    fig.suptitle(
        f"Partitioning at ε = 0.01 K (1e-7 target), Bull, 8 ranks — "
        f"DP is {100*(ratio-1):.1f}% faster",
        color=INK, fontsize=11, x=0.02, ha="left")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK2,
               loc="upper right", bbox_to_anchor=(0.99, 0.97), ncol=3,
               columnspacing=1.2, handlelength=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figC_dp_vs_uniform_eps0p01.{ext}",
                    facecolor=SURFACE)
    plt.close(fig)
    print("C: written")


# -- Figure D: self-convergence ladder, estimate vs truth --------------------
def fig_d() -> None:
    root = Path("outputs/accuracy_bull_tol1e4_h30")
    # True max rel-L2 vs the serial reference after k repair rungs
    # (default weight w = 0.25; job 855007).
    true_err = []
    base = load(root / "compare_par32_ladder4_w025/comparison_summary.json")
    if base is None:
        print("D: missing ladder comparison data, figure not written")
        return
    true_err.append(base["max_rel_l2"])
    for k in range(1, 7):
        d = load(root / f"compare_par32_ladder4_w025_iter{k}/comparison_summary.json")
        if d is None:
            break
        true_err.append(d["max_rel_l2"])
    # Self-check ladder shifts d_k (max), measured WITHOUT any reference
    # (logs/selfchk31_s4_855007.out, w025 block). d_{k+1} estimates the
    # remaining error after rung k; the cumulative estimate of the
    # production error is 1.4062e-4 = truth (ratio 1.000).
    d_shift = [1.4062e-04, 5.6412e-05, 1.6113e-05, 9.9778e-06,
               3.9384e-06, 6.3892e-07]

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.set_yscale("log")

    ax.axhline(1e-4, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
               zorder=1)
    ax.annotate("1e-4 target", (2.6, 1e-4), textcoords="offset points",
                xytext=(0, 4), color=MUTED, fontsize=9, ha="center")

    ks = list(range(len(true_err)))
    ax.plot(ks, true_err, color=S1, linewidth=2, marker="o", markersize=6,
            markerfacecolor=S1, markeredgecolor=SURFACE, markeredgewidth=1,
            zorder=3, label="true error (vs serial reference)")
    ax.plot(range(len(d_shift)), d_shift, color=S2, linewidth=0,
            marker="o", markersize=9, markerfacecolor="none",
            markeredgecolor=S2, markeredgewidth=2, zorder=4,
            label="self-check estimate (no reference)")

    ax.annotate(f"{true_err[0]:.2e}", (0, true_err[0]),
                textcoords="offset points", xytext=(12, 4), color=INK2,
                fontsize=9)
    ax.annotate(f"{true_err[-1]:.2e}", (ks[-1], true_err[-1]),
                textcoords="offset points", xytext=(0, -14), ha="center",
                color=INK2, fontsize=9)

    ax.annotate("estimate = truth at every rung\n(cumulative ratio 1.000)",
                (0.70, 0.80), xycoords="axes fraction", color=INK,
                fontsize=10, fontweight="bold", ha="center")

    ax.set_xticks(ks)
    ax.set_xlabel("repair iteration (+4 supersegments of horizon each)",
                  color=INK2, fontsize=10)
    ax.set_ylabel("max rel-L2 error", color=INK2, fontsize=10)
    ax.set_title("Self-convergence ladder — Bull 1e-4 config, 32 ranks",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figD_self_convergence.{ext}", facecolor=SURFACE)
    plt.close(fig)
    print(f"D: written ({len(true_err)} true points, {len(d_shift)} estimates)")


if __name__ == "__main__":
    fig_a()
    fig_a16()
    fig_b()
    fig_c()
    fig_c_scaled(32)
    fig_c_scaled(64)
    for r in (8, 32, 64):
        fig_c_scaled(r, root="outputs/gate_hybrid",
                     pathname="Spiral-Raster", prefix="figC_hyb")
    # linear cost model variant -> outputs/linear_model/
    _lin_out = Path("outputs/linear_model")
    for r in (8, 32, 64):
        fig_c_scaled(
            r, pathname="Bull (linear cost model)", prefix="figL_bull",
            dp_dir=f"outputs/linear_model/bull/r{r}_dp_linear",
            uniform_dir=("outputs/gate8/eps0p01_uniform" if r == 8
                         else f"outputs/gate3264/r{r}_uniform"),
            outdir=_lin_out)
        fig_c_scaled(
            r, pathname="Spiral-Raster (linear cost model)", prefix="figL_hyb",
            dp_dir=f"outputs/linear_model/hyb/r{r}_dp_linear",
            uniform_dir=f"outputs/gate_hybrid/r{r}_uniform",
            outdir=_lin_out)
    fig_d()
