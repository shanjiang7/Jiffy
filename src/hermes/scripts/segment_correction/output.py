from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hermes.utils.snapshot_utils import crop_snapshot


def comp_start_step(path_defs, steps_per_ss: int) -> dict:
    mapping = {}
    acc = 0
    for pd in path_defs:
        mapping[pd.component_id] = acc
        acc += int(getattr(pd, "total_steps", int(pd.weight) * int(steps_per_ss)))
    return mapping


def _component_layer_offsets(path_defs, *, ss_per_layer: int, steps_per_ss: int) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    if int(ss_per_layer) < 1:
        raise ValueError("ss_per_layer must be >= 1")

    layer_idx_by_comp: dict[int, int] = {}
    layer_start_step_by_comp: dict[int, int] = {}
    layer_total_steps: dict[int, int] = {}

    acc_ss = 0
    acc_layer_steps = 0
    for pd in path_defs:
        if (int(acc_ss) % int(ss_per_layer)) == 0:
            acc_layer_steps = 0
        comp_id = int(pd.component_id)
        layer_idx = int(acc_ss) // int(ss_per_layer)
        layer_idx_by_comp[comp_id] = layer_idx
        layer_start_step_by_comp[comp_id] = int(acc_layer_steps)
        acc_layer_steps += int(getattr(pd, "total_steps", int(pd.weight) * int(steps_per_ss)))
        layer_total_steps[layer_idx] = int(acc_layer_steps)
        acc_ss += int(pd.weight)

    return layer_idx_by_comp, layer_start_step_by_comp, layer_total_steps


def build_component_start_snapshot_steps(
    path_defs,
    *,
    interval_steps: int,
    max_snapshots_per_component: int,
) -> dict[int, list[int]]:
    if interval_steps <= 0:
        raise ValueError("interval_steps must be >= 1")
    if max_snapshots_per_component <= 0:
        raise ValueError("max_snapshots_per_component must be >= 1")

    snapshot_steps_by_component: dict[int, list[int]] = {}
    for pd in path_defs:
        total_steps = int(pd.total_steps)
        rel_steps: list[int] = []
        for idx in range(int(max_snapshots_per_component)):
            rel_step = idx * int(interval_steps)
            if rel_step >= total_steps:
                break
            rel_steps.append(rel_step)
        if not rel_steps:
            rel_steps = [0]
        snapshot_steps_by_component[int(pd.component_id)] = rel_steps
    return snapshot_steps_by_component


def build_global_stride_snapshot_steps(
    path_defs,
    *,
    ss_per_layer: int,
    steps_per_ss: int,
    snap_every_steps: int,
) -> dict[int, list[int]]:
    if int(snap_every_steps) <= 0:
        raise ValueError("snap_every_steps must be >= 1")

    _, layer_start_step_by_comp, _ = _component_layer_offsets(
        path_defs,
        ss_per_layer=int(ss_per_layer),
        steps_per_ss=int(steps_per_ss),
    )

    stride = int(snap_every_steps)
    snapshot_steps_by_component: dict[int, list[int]] = {}
    for pd in path_defs:
        comp_id = int(pd.component_id)
        comp_start = int(layer_start_step_by_comp[comp_id])
        comp_total = int(getattr(pd, "total_steps", int(pd.weight) * int(steps_per_ss)))
        comp_end = comp_start + comp_total
        first_global = ((comp_start + stride - 1) // stride) * stride
        rel_steps = [int(g - comp_start) for g in range(first_global, comp_end, stride)]
        snapshot_steps_by_component[comp_id] = rel_steps
    return snapshot_steps_by_component


def position_after_steps(path_def, n_steps: int) -> tuple[float, float]:
    steps_left = max(0, min(int(n_steps), int(path_def.total_steps)))
    x = float(path_def.x_start)
    y = float(path_def.y_start)
    for leg in path_def.legs:
        if steps_left <= 0:
            break
        take = min(steps_left, int(leg.steps))
        x += float(leg.dx_step) * take
        y += float(leg.dy_step) * take
        steps_left -= take
    return x, y


def save_parallel_snapshots(
    *,
    rank: int,
    snaps_dir: Path,
    meta_dir: Path,
    final_states_host,
    path_defs,
    path_def_by_id,
    start_step_map,
    ss_per_layer: int,
    steps_per_ss: int,
    ctx,
    h_m: float,
    snapshot_steps_by_component: dict[int, list[int]] | None = None,
    snap_every_steps: int | None = None,
) -> None:
    layer_idx_by_comp, layer_start_step_by_comp, layer_total_steps = _component_layer_offsets(
        path_defs,
        ss_per_layer=int(ss_per_layer),
        steps_per_ss=int(steps_per_ss),
    )
    meta_records: list[dict[str, object]] = []
    rank_exec_index = 0

    for comp_id, snaps in final_states_host.items():
        pd = path_def_by_id[int(comp_id)]
        start_step = start_step_map[comp_id]
        layer_idx = int(layer_idx_by_comp[int(comp_id)])
        layer_start_step = int(layer_start_step_by_comp[int(comp_id)])
        steps_in_layer = int(layer_total_steps[int(layer_idx)])
        if snapshot_steps_by_component is not None:
            rel_snapshot_steps = snapshot_steps_by_component.get(int(comp_id), [])
        else:
            if snap_every_steps is None:
                raise ValueError("snap_every_steps is required when snapshot_steps_by_component is not provided")
            rel_snapshot_steps = [k * int(snap_every_steps) for k in range(len(snaps))]
        for k, arr in enumerate(snaps):
            if k >= len(rel_snapshot_steps):
                break
            rel_step = int(rel_snapshot_steps[k])
            global_step = start_step + rel_step
            within_layer_step = layer_start_step + rel_step
            if within_layer_step >= steps_in_layer:
                break
            if arr.ndim == 1 and arr.size == int(ctx.nx) * int(ctx.ny) * int(ctx.nz):
                cropped = crop_snapshot(arr, ctx.nx, ctx.ny, ctx.nz, h_m)
            else:
                cropped = np.array(arr, copy=True)
            fname = f"layer_{layer_idx:02d}_step_{within_layer_step:09d}.npy"
            np.save(snaps_dir / fname, cropped)
            cx_nd, cy_nd = position_after_steps(pd, rel_step)
            meta_records.append({
                "file": fname,
                "rank": int(rank),
                "component_id": int(comp_id),
                "exec_index": int(rank_exec_index),
                "layer": int(layer_idx),
                "kind": "step",
                "index": int(within_layer_step),
                "component_step": int(rel_step),
                "center_x_nd": float(cx_nd),
                "center_y_nd": float(cy_nd),
            })
            rank_exec_index += 1

    meta_path = meta_dir / f"rank_{rank:02d}.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for rec in meta_records:
            f.write(json.dumps(rec) + "\n")
