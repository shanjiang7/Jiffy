"""
DAG utilities: Edge type, Component type, DAG formatting, and I/O.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Edge:
    """Directed edge src → dst."""
    src: int
    dst: int


@dataclass
class Component:
    id: int
    start_ss: int   # inclusive
    end_ss: int     # inclusive
    depth: int      # max(dst - src) for cross-edges inside; 1 if trivial
    kind: str       # 'trivial' or 'coupled'

    @property
    def size(self) -> int:
        return self.end_ss - self.start_ss + 1


# ── Algorithms ────────────────────────────────────────────────────────────────

def find_components(n_ss: int, edges: list[tuple[int, int]]) -> list[Component]:
    """Identify trivial sequences vs coupled blocks in the supersegment DAG."""
    if n_ss == 0:
        return []

    cross = sorted([(src, dst) for src, dst in edges if dst - src > 1], key=lambda x: x[0])

    merged = []
    for src, dst in cross:
        if merged and src <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], dst)
        else:
            merged.append([src, dst])

    components, comp_id, cursor = [], 0, 0

    for start, end in merged:
        if cursor < start:
            components.append(Component(comp_id, cursor, start - 1, 1, "trivial"))
            comp_id += 1
        
        depth = max((d - s for s, d in cross if s >= start and d <= end), default=0)
        components.append(Component(comp_id, start, end, depth, "coupled"))
        comp_id += 1
        cursor = end + 1

    if cursor < n_ss:
        components.append(Component(comp_id, cursor, n_ss - 1, 1, "trivial"))

    return components


def split_components_at_layer_boundaries(
    components: list[Component],
    ss_per_layer: int,
) -> list[Component]:
    """Split any component that crosses a layer boundary into per-layer pieces."""
    if ss_per_layer <= 0:
        raise ValueError("ss_per_layer must be >= 1")

    out: list[Component] = []
    comp_id = 0
    layer_span = int(ss_per_layer)
    for comp in components:
        start = int(comp.start_ss)
        end = int(comp.end_ss)
        while start <= end:
            layer_end = ((start // layer_span) + 1) * layer_span - 1
            piece_end = min(end, layer_end)
            out.append(
                Component(
                    id=comp_id,
                    start_ss=start,
                    end_ss=piece_end,
                    depth=int(comp.depth),
                    kind=str(comp.kind),
                )
            )
            comp_id += 1
            start = piece_end + 1
    return out


# ── I/O ───────────────────────────────────────────────────────────────────────

def write_edges_csv(edges: Iterable[Edge], out_csv: str | Path, *, header: str) -> Path:
    out_path = Path(out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray([[int(e.src), int(e.dst)] for e in edges], dtype=int)
    if rows.size == 0:
        rows = np.zeros((0, 2), dtype=int)
    np.savetxt(out_path, rows, delimiter=",", header=header, comments="", fmt="%d")
    return out_path


def load_edges_csv(csv_path: Path) -> tuple[list[tuple[int, int]], int]:
    """Read dag_edges.csv → (edge list, max node id seen)."""
    edges = []
    max_node = -1
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src, dst = int(row["src_ss_id"]), int(row["dst_ss_id"])
            edges.append((src, dst))
            max_node = max(max_node, src, dst)
    return edges, max_node


def write_components_csv(components: list[Component], out_path: Path) -> Path:
    """Write component table to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["comp_id", "start_ss", "end_ss", "size", "depth", "kind"])
        for c in components:
            writer.writerow([c.id, c.start_ss, c.end_ss, c.size, c.depth, c.kind])
    return out_path


def write_dependency_lists(edges: Iterable[Edge], n_nodes: int, out_prefix: Path) -> tuple[Path, Path]:
    n = int(n_nodes)
    in_deps: list[list[int]] = [[] for _ in range(n)]
    out_deps: list[list[int]] = [[] for _ in range(n)]
    for e in edges:
        u, v = int(e.src), int(e.dst)
        if 0 <= u < n and 0 <= v < n:
            out_deps[u].append(v)
            in_deps[v].append(u)

    def _fmt(xs: list[int]) -> str:
        return ";".join(str(x) for x in sorted(set(xs))) if xs else ""

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    in_path = out_prefix.parent / f"{out_prefix.name}_in.csv"
    out_path_csv = out_prefix.parent / f"{out_prefix.name}_out.csv"

    with in_path.open("w") as f:
        f.write("dst_ss_id,in_deps\n")
        for v in range(n):
            f.write(f"{v},{_fmt(in_deps[v])}\n")
    with out_path_csv.open("w") as f:
        f.write("src_ss_id,out_deps\n")
        for u in range(n):
            f.write(f"{u},{_fmt(out_deps[u])}\n")

    return in_path, out_path_csv


def write_intervals_csv(
    intervals: list[tuple[int, int]],
    out_path: Path,
    *,
    header: tuple[str, str] = ("start_node", "end_node"),
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([header[0], header[1]])
        for start, end in intervals:
            writer.writerow([int(start), int(end)])
    return out_path


__all__ = [
    "Edge",
    "Component",
    "find_components",
    "split_components_at_layer_boundaries",
    "write_edges_csv",
    "load_edges_csv",
    "write_components_csv",
    "write_dependency_lists",
    "write_intervals_csv",
]
