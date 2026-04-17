"""
Plotting utilities for the supersegment DAG pipeline.

Functions
---------
plot_path_with_supersegments  – laser path with SS boundary markers
plot_dag_spatial              – DAG edges overlaid on laser path
plot_components               – laser path coloured by DAG component
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from hermes.utils.dag_io import Edge


# ── helpers ───────────────────────────────────────────────────────────────────

def _traj_mm(supersegments, len_scale: float) -> np.ndarray:
    pts = []
    for ss in supersegments:
        for seg in ss.segments:
            for st in seg.steps:
                pts.append((float(st.x_nd), float(st.y_nd)))
    return np.asarray(pts, dtype=float) * float(len_scale) * 1000.0


def _ss_anchors_mm(supersegments, len_scale: float) -> dict[int, Tuple[float, float]]:
    s = float(len_scale) * 1000.0
    pos = {}
    for ss in supersegments:
        st = ss.segments[0].steps[0]
        pos[int(ss.id)] = (float(st.x_nd) * s, float(st.y_nd) * s)
    return pos


# ── public API ────────────────────────────────────────────────────────────────

def plot_path_with_supersegments(
    supersegments,
    *,
    len_scale: float,
    out_path: Path,
) -> None:
    """Plot laser path with supersegment start points marked."""
    XY = _traj_mm(supersegments, len_scale)
    if XY.size == 0:
        print("[skip] no points to plot")
        return

    s = float(len_scale) * 1000.0
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    ax.plot(XY[:, 0], XY[:, 1], "-", color="skyblue", alpha=0.6, lw=0.8, zorder=1)

    for ss in supersegments:
        st = ss.segments[0].steps[0]
        x, y = float(st.x_nd) * s, float(st.y_nd) * s
        ax.plot(x, y, "o", ms=7, color="steelblue", markeredgecolor="black",
                markeredgewidth=0.8, zorder=5)
        ax.text(x, y, f" SS{ss.id}", fontsize=6, color="black",
                ha="left", va="bottom", zorder=6)

    if supersegments:
        st0 = supersegments[0].segments[0].steps[0]
        ax.plot(float(st0.x_nd) * s, float(st0.y_nd) * s,
                "s", ms=10, color="lime", markeredgecolor="black",
                markeredgewidth=1.0, zorder=7, label="start")
        stl = supersegments[-1].segments[-1].steps[-1]
        ax.plot(float(stl.x_nd) * s, float(stl.y_nd) * s,
                "D", ms=10, color="red", markeredgecolor="black",
                markeredgewidth=1.0, zorder=7, label="end")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    n_steps = sum(len(seg.steps) for ss in supersegments for seg in ss.segments)
    ax.set_title(f"Laser path – {n_steps} steps, {len(supersegments)} supersegments")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  [ok] wrote: {out_path}")


def plot_dag_spatial(
    ss_nodes,
    edges: List["Edge"],
    *,
    len_scale: float,
    out_path: Path,
) -> None:
    """Plot supersegment DAG edges (chain=grey, cross=red) over laser path."""
    XY = _traj_mm(ss_nodes, len_scale)
    if XY.size == 0:
        return
    ss_pos = _ss_anchors_mm(ss_nodes, len_scale)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    ax.plot(XY[:, 0], XY[:, 1], "-", color="skyblue", alpha=0.4, lw=0.5, zorder=1)

    chain_edges, cross_edges = [], []
    for e in edges:
        src, dst = int(e.src), int(e.dst)
        if src in ss_pos and dst in ss_pos:
            (chain_edges if dst == src + 1 else cross_edges).append(e)

    for e in chain_edges:
        ax.annotate("", xy=ss_pos[int(e.dst)], xytext=ss_pos[int(e.src)],
                    arrowprops=dict(arrowstyle="->", color="grey", alpha=0.3, lw=0.5),
                    zorder=2)
    for e in cross_edges:
        ax.annotate("", xy=ss_pos[int(e.dst)], xytext=ss_pos[int(e.src)],
                    arrowprops=dict(arrowstyle="->", color="red", alpha=0.6, lw=1.2),
                    zorder=4)

    all_ids = sorted(ss_pos.keys())
    ax.scatter([ss_pos[i][0] for i in all_ids],
               [ss_pos[i][1] for i in all_ids],
               c="steelblue", s=30, edgecolors="black", linewidths=0.4, zorder=5)

    stride = max(1, len(all_ids) // 40)
    for k in range(0, len(all_ids), stride):
        sid = all_ids[k]
        ax.text(ss_pos[sid][0], ss_pos[sid][1], f" {sid}",
                fontsize=6, color="black", ha="left", va="bottom", zorder=6)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"Supersegment DAG – {len(ss_nodes)} SS, "
                 f"{len(chain_edges)} chain + {len(cross_edges)} cross-edges (red)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  [ok] wrote: {out_path}")


def plot_components(
    components,
    edges: List[Tuple[int, int]],
    n_ss: int,
    *,
    out_path: Path,
) -> None:
    """
    Node-link graph of the supersegment DAG, coloured and grouped by component.

    Layout: nodes placed at x = ss_id along a horizontal line.
    Chain edges arc below; cross-edges arc above, labelled with gap Δ.
    Trivial nodes: grey.  Coupled nodes: tab10 colours per component.
    """
    coupled_comps = [c for c in components if c.kind == "coupled"]
    cmap = plt.get_cmap("tab10", max(len(coupled_comps), 1))

    ss_to_comp: dict[int, object] = {}
    comp_color: dict[int, object] = {}
    for k, c in enumerate(coupled_comps):
        comp_color[int(c.id)] = cmap(k % 10)
        for ss_id in range(int(c.start_ss), int(c.end_ss) + 1):
            ss_to_comp[ss_id] = c
    for c in components:
        if c.kind == "trivial":
            comp_color[int(c.id)] = "silver"
            for ss_id in range(int(c.start_ss), int(c.end_ss) + 1):
                ss_to_comp[ss_id] = c

    # Node x positions (evenly spaced)
    xs = {i: float(i) for i in range(n_ss)}
    y0 = 0.0

    fig, ax = plt.subplots(figsize=(max(12, n_ss * 0.25), 5), dpi=150)

    # Draw nodes
    for ss_id in range(n_ss):
        c = ss_to_comp.get(ss_id)
        color = comp_color[int(c.id)] if c else "silver"
        is_coupled = c is not None and c.kind == "coupled"
        ax.plot(xs[ss_id], y0, "s", ms=8 if is_coupled else 5,
                color=color, markeredgecolor="black", markeredgewidth=0.5, zorder=4)

    # Label coupled component spans with a bracket
    for c in coupled_comps:
        x_lo = xs[int(c.start_ss)]
        x_hi = xs[min(int(c.end_ss), n_ss - 1)]
        color = comp_color[int(c.id)]
        ax.annotate("", xy=(x_hi, y0 - 0.3), xytext=(x_lo, y0 - 0.3),
                    arrowprops=dict(arrowstyle="|-|", color=color, lw=1.5))
        ax.text((x_lo + x_hi) / 2, y0 - 0.45,
                f"C{c.id}  sz={c.size}  d={c.depth}",
                ha="center", va="top", fontsize=7, color=color)

    # Draw edges
    for src, dst in edges:
        if src >= n_ss or dst >= n_ss:
            continue
        c = ss_to_comp.get(src)
        is_cross = dst - src > 1
        if is_cross:
            color = comp_color[int(c.id)] if c and c.kind == "coupled" else "red"
            gap = dst - src
            rad = 0.3 + 0.05 * gap          # arc upward
            ax.annotate("", xy=(xs[dst], y0), xytext=(xs[src], y0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.2,
                                        connectionstyle=f"arc3,rad=-{rad:.2f}"),
                        zorder=5)
            ax.text((xs[src] + xs[dst]) / 2, y0 + rad * 0.8,
                    f"Δ{gap}", ha="center", va="bottom", fontsize=6, color=color)
        else:
            ax.annotate("", xy=(xs[dst], y0), xytext=(xs[src], y0),
                        arrowprops=dict(arrowstyle="->", color="grey", lw=0.5, alpha=0.4,
                                        connectionstyle="arc3,rad=0.15"),
                        zorder=2)

    # SS id labels (every N nodes to avoid clutter)
    stride = max(1, n_ss // 60)
    for ss_id in range(0, n_ss, stride):
        ax.text(xs[ss_id], y0 + 0.12, str(ss_id),
                ha="center", va="bottom", fontsize=6, color="black")

    n_trivial = sum(1 for c in components if c.kind == "trivial")
    ax.set_title(f"DAG Components – {len(components)} total "
                 f"({n_trivial} trivial, {len(coupled_comps)} coupled)")
    ax.set_xlim(-1, n_ss)
    ax.set_ylim(-1.0, 1.5)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  [ok] wrote: {out_path}")


def plot_dag_networkx(
    edges: List["Edge"],
    n_nodes: int,
    *,
    supersegments=None,
    components=None,
    len_scale: float = 1.0,
    out_path: Path,
) -> None:
    """
    Spatial DAG graph placed at real (x, y) laser positions.

    - Circle nodes for every SS, coloured by component kind.
    - Black lines for chain edges (SS_i → SS_{i+1}).
    - For each coupled component: one red line from start_ss → end_ss
      (individual cross-edges are suppressed; the red span conveys the coupling).
    """
    # ── spatial positions ────────────────────────────────────────────
    if supersegments is not None:
        ss_pos = _ss_anchors_mm(supersegments, len_scale)
        XY = _traj_mm(supersegments, len_scale)
    else:
        ss_pos = {i: (float(i), 0.0) for i in range(n_nodes)}
        XY = None

    # ── component membership ─────────────────────────────────────────
    ss_kind: dict[int, str] = {}
    comp_of_ss: dict[int, object] = {}
    comp_color: dict[int, object] = {}
    coupled_comps = []

    COUPLED_COLOR = "tomato"

    if components is not None:
        coupled_comps = [c for c in components if c.kind == "coupled"]
        for c in coupled_comps:
            comp_color[c.id] = COUPLED_COLOR
            for ss_id in range(int(c.start_ss), int(c.end_ss) + 1):
                ss_kind[ss_id] = "coupled"
                comp_of_ss[ss_id] = c
        for c in components:
            if c.kind == "trivial":
                for ss_id in range(int(c.start_ss), int(c.end_ss) + 1):
                    ss_kind.setdefault(ss_id, "trivial")
                    comp_of_ss.setdefault(ss_id, c)

    # ── figure ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)

    if XY is not None and XY.size > 0:
        ax.plot(XY[:, 0], XY[:, 1], "-", color="skyblue", alpha=0.25, lw=0.5, zorder=1)

    # Chain edges — black lines connecting consecutive SS nodes
    chain_edges = [
        (int(e.src), int(e.dst)) for e in edges if int(e.dst) == int(e.src) + 1
    ]
    for src, dst in chain_edges:
        if src in ss_pos and dst in ss_pos:
            x0, y0 = ss_pos[src]
            x1, y1 = ss_pos[dst]
            ax.plot([x0, x1], [y0, y1], "-", color="black", lw=0.9, alpha=0.55, zorder=2)

    # Coupled components — one red span line per component (start_ss → end_ss)
    for c in coupled_comps:
        s_id, e_id = int(c.start_ss), int(c.end_ss)
        if s_id in ss_pos and e_id in ss_pos:
            x0, y0 = ss_pos[s_id]
            x1, y1 = ss_pos[e_id]
            ax.plot([x0, x1], [y0, y1], "-", color="red", lw=2.5, alpha=0.85, zorder=3,
                    solid_capstyle="round")
            # Label at midpoint
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my, f"C{c.id}(sz={c.size})", fontsize=6, color="darkred",
                    ha="center", va="bottom", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # Nodes — circles coloured by kind
    for ss_id in sorted(ss_pos.keys()):
        kind = ss_kind.get(ss_id, "trivial")
        if kind == "coupled":
            c = comp_of_ss.get(ss_id)
            color = comp_color.get(c.id, "tomato") if c else "tomato"
            ms = 7
        else:
            color = "lightgrey"
            ms = 5
        x, y = ss_pos[ss_id]
        ax.plot(x, y, "o", ms=ms, color=color,
                markeredgecolor="black", markeredgewidth=0.5, zorder=5)

    # SS id labels (thinned to avoid clutter)
    stride = max(1, n_nodes // 40)
    for k, ss_id in enumerate(sorted(ss_pos.keys())):
        if k % stride == 0:
            x, y = ss_pos[ss_id]
            ax.text(x, y, f" {ss_id}", fontsize=6, color="black",
                    ha="left", va="bottom", zorder=7)

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="black", lw=1.5, label="chain edge (SS → SS+1)"),
        Line2D([0], [0], color="red",   lw=2.5, label="coupled component span (start → end)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgrey",
               markeredgecolor="black", ms=7, label="trivial SS"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COUPLED_COLOR,
               markeredgecolor="black", ms=7, label="coupled SS"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(
        f"Spatial DAG — {n_nodes} SS | {len(chain_edges)} chain edges (black) | "
        f"{len(coupled_comps)} coupled components (red span)"
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  [ok] wrote: {out_path}")


def save_plots(
    out_dir: Path,
    error_matrix: np.ndarray,
    seg_errors: np.ndarray,
    num_segments: int,
    error_matrix_roi: np.ndarray | None = None,
) -> None:
    loops = np.arange(1, error_matrix.shape[0] + 1)
    segs = np.arange(1, num_segments + 1)

    plt.figure(figsize=(8, 4))
    log_err = np.log10(np.maximum(error_matrix, 1e-16))
    im = plt.imshow(log_err, origin="lower", aspect="auto", cmap="viridis")
    plt.xticks(np.arange(num_segments), [str(s) for s in segs])
    plt.yticks(np.arange(len(loops)), [str(k) for k in loops])
    plt.xlabel("Segment Index")
    plt.ylabel("Loop Index")
    plt.title("log10(Relative L2 Error) vs Sequential")
    cbar = plt.colorbar(im)
    cbar.set_label("log10(rel L2)")
    plt.tight_layout()
    plt.savefig(out_dir / "error_matrix_loops_x_segments.png", dpi=200)
    plt.close()

    if error_matrix_roi is not None:
        plt.figure(figsize=(8, 4))
        log_err_roi = np.log10(np.maximum(error_matrix_roi, 1e-16))
        im = plt.imshow(log_err_roi, origin="lower", aspect="auto", cmap="viridis")
        plt.xticks(np.arange(num_segments), [str(s) for s in segs])
        plt.yticks(np.arange(len(loops)), [str(k) for k in loops])
        plt.xlabel("Segment Index")
        plt.ylabel("Loop Index")
        plt.title("log10(Relative L2 Error) ROI Domain vs Sequential")
        cbar = plt.colorbar(im)
        cbar.set_label("log10(rel L2)")
        plt.tight_layout()
        plt.savefig(out_dir / "error_matrix_roi_loops_x_segments.png", dpi=200)
        plt.close()

    plt.figure(figsize=(6, 4))
    plt.semilogy(loops, seg_errors, marker="o")
    plt.xlabel("Loop Index")
    plt.ylabel("Relative L2 Error")
    plt.title(f"Segment {num_segments} Convergence")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / f"segment_{num_segments}_l2_convergence.png", dpi=200)
    plt.close()


__all__ = [
    "plot_path_with_supersegments",
    "plot_dag_spatial",
    "plot_components",
    "plot_group_ranges",
    "plot_dag_networkx",
    "save_plots",
]



def plot_group_ranges(
    groups,
    edges: List[Tuple[int, int]],
    n_ss: int,
    *,
    out_path: Path,
    title: str,
    group_metrics: dict[int, dict] | None = None,
    max_nodes: int | None = None,
) -> None:
    """
    Plot a horizontal DAG view coloured by runtime groups.

    The grouped view emphasizes cut locations and whether an edge crosses a cut.
    Edges internal to a super segment are shown in dark grey, while any edge that
    crosses a cut is shown in red.
    """
    if n_ss <= 0:
        return

    n_plot = int(n_ss)
    if max_nodes is not None and int(max_nodes) > 0:
        n_plot = min(int(n_ss), int(max_nodes))
    if int(n_plot) <= 0:
        return

    ordered_groups = sorted(groups, key=lambda g: (int(g.start_ss), int(g.id)))
    ss_to_group: dict[int, object] = {}
    node_color = 'tab:blue'
    for group in ordered_groups:
        for ss_id in range(int(group.start_ss), min(int(group.end_ss), int(n_plot) - 1) + 1):
            ss_to_group[ss_id] = group

    xs = {i: float(i) for i in range(n_plot)}
    y0 = 0.0
    fig, ax = plt.subplots(figsize=(max(12, n_plot * 0.25), 5), dpi=150)

    for ss_id in range(n_plot):
        ax.plot(
            xs[ss_id],
            y0,
            'o',
            ms=6.5,
            color=node_color,
            markeredgecolor='black',
            markeredgewidth=0.5,
            zorder=4,
        )

    cut_boundaries = [int(group.end_ss) for group in ordered_groups[:-1] if int(group.end_ss) < int(n_plot) - 1]
    for idx, boundary in enumerate(cut_boundaries):
        x_cut = xs[int(boundary)] + 0.5
        ax.plot(
            [x_cut, x_cut],
            [y0 - 0.225, y0 + 0.225],
            color='black',
            lw=1.0,
            alpha=0.85,
            zorder=3,
        )
        right_group = ordered_groups[int(idx) + 1]
        right_metric = group_metrics.get(int(right_group.id), {}) if group_metrics is not None else {}
        cut_cost = float(
            right_metric.get(
                'correction_span_segments',
                right_metric.get('correction_span_ss', 0.0),
            )
        )
        ax.text(
            x_cut,
            y0 - 0.3,
            f'C={cut_cost:.0f}',
            ha='center',
            va='top',
            fontsize=5.5,
            color='black',
            zorder=6,
        )

    for src, dst in edges:
        if src >= n_plot or dst >= n_plot:
            continue
        src_group = ss_to_group.get(src)
        dst_group = ss_to_group.get(dst)
        same_group = (
            src_group is not None
            and dst_group is not None
            and int(src_group.id) == int(dst_group.id)
        )
        crosses_cut = not same_group
        is_cross = dst - src > 1
        if is_cross:
            color = 'red' if crosses_cut else 'dimgray'
            gap = dst - src
            rad = 0.3 + 0.05 * gap
            ax.annotate(
                '',
                xy=(xs[dst], y0),
                xytext=(xs[src], y0),
                arrowprops=dict(
                    arrowstyle='->',
                    color=color,
                    lw=1.0 if same_group else 1.2,
                    alpha=0.8 if same_group else 1.0,
                    connectionstyle=f'arc3,rad=-{rad:.2f}',
                ),
                zorder=5,
            )
        else:
            ax.annotate(
                '',
                xy=(xs[dst], y0),
                xytext=(xs[src], y0),
                arrowprops=dict(
                    arrowstyle='->',
                    color='red' if crosses_cut else 'dimgray',
                    lw=0.8 if crosses_cut else 0.6,
                    alpha=0.95 if crosses_cut else 0.6,
                    connectionstyle='arc3,rad=0.15',
                ),
                zorder=2,
            )

    for ss_id in range(n_plot):
        ax.text(
            xs[ss_id],
            y0 - 0.08,
            str(ss_id),
            ha='center',
            va='top',
            fontsize=5.5,
            color='black',
            zorder=6,
        )

    ax.set_xlim(-1, n_plot)
    ax.set_ylim(-0.8, 1.3)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f'  [ok] wrote: {out_path}')
