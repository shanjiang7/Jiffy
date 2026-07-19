"""
Independent serial reference run.

True sequential ground-truth solve (no runtime component plan involved).
It executes the original laser path sequentially, grouped only by layer, so the
saved serial snapshots are independent of DAG/component partitioning choices
such as dependency level_K, planner mode, or grouping parameters.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict

import cupy as cp
import numpy as np

from hermes.pipelines.config import PipelineConfig
from hermes.pipelines.ss_builder import build_ss_from_cfg
from hermes.runtime.setup import load_sim_setup, select_float_type
from hermes.scripts.outer_solver import build_outer_context, run_ss_outer
from hermes.utils.dag_utils import Component
from hermes.utils.mpi_utils import bind_local_gpu
from hermes.utils.path_utils import build_path_defs_from_components, resolve_path
from hermes.utils.snapshot_utils import crop_snapshot


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Independent serial reference run")
    p.add_argument("--config", default="configs/examples/sim_ex1.ini", help="Base simulation config")
    p.add_argument("--path-config", required=True, help="Laser path config")
    p.add_argument("--out-dir", default="outputs/serial_reference", help="Output directory")
    p.add_argument("--dt-us", type=float, help="Override dt in microseconds")
    p.add_argument(
        "--snap-every-steps",
        type=int,
        default=100,
        help="Save a snapshot every N timesteps across each layer (default: 100).",
    )
    p.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Override num_layers from path config INI (default: use INI value)",
    )
    p.add_argument(
        "--solver-mode",
        choices=("fused", "legacy"),
        default="fused",
        help="Level-3 outer solver mode (default: fused).",
    )
    p.add_argument(
        "--boundary-visualization-json",
        help=(
            "Metadata JSON produced by dev/serial_emulated_run.py "
            "--boundary-correction-snapshot-mode. When provided, only the matching "
            "boundary window is saved."
        ),
    )
    p.add_argument(
        "--save-temperature-K",
        action="store_true",
        help="Save absolute temperature in Kelvin instead of nondimensional solver state.",
    )
    return p.parse_args(argv)


def _build_layer_components(*, num_layers: int, ss_per_layer: int) -> list[Component]:
    if int(num_layers) < 1:
        raise ValueError("num_layers must be >= 1")
    if int(ss_per_layer) < 1:
        raise ValueError("ss_per_layer must be >= 1")
    comps: list[Component] = []
    for layer_idx in range(int(num_layers)):
        start_ss = int(layer_idx) * int(ss_per_layer)
        end_ss = start_ss + int(ss_per_layer) - 1
        comps.append(
            Component(
                id=int(layer_idx),
                start_ss=int(start_ss),
                end_ss=int(end_ss),
                depth=1,
                kind="trivial",
            )
        )
    return comps


def _build_supersegment_components(*, num_layers: int, ss_per_layer: int) -> list[Component]:
    if int(num_layers) < 1:
        raise ValueError("num_layers must be >= 1")
    if int(ss_per_layer) < 1:
        raise ValueError("ss_per_layer must be >= 1")
    comps: list[Component] = []
    comp_id = 0
    for layer_idx in range(int(num_layers)):
        layer_start = int(layer_idx) * int(ss_per_layer)
        for ss_offset in range(int(ss_per_layer)):
            ss_idx = layer_start + int(ss_offset)
            comps.append(
                Component(
                    id=int(comp_id),
                    start_ss=int(ss_idx),
                    end_ss=int(ss_idx),
                    depth=1,
                    kind="trivial",
                )
            )
            comp_id += 1
    return comps


def _build_layer_snapshot_steps_by_supersegment(
    layer_defs,
    *,
    snap_every_steps: int,
) -> list[list[int]]:
    stride = int(snap_every_steps)
    if stride <= 0:
        raise ValueError("snap_every_steps must be >= 1")

    snapshot_steps_by_ss: list[list[int]] = []
    layer_step_offset = 0
    for path_def in layer_defs:
        total_steps = int(path_def.total_steps)
        first_global = ((layer_step_offset + stride - 1) // stride) * stride
        rel_steps = [
            int(global_step - layer_step_offset)
            for global_step in range(first_global, layer_step_offset + total_steps, stride)
        ]
        snapshot_steps_by_ss.append(rel_steps)
        layer_step_offset += total_steps
    return snapshot_steps_by_ss


def _build_layer_snapshot_steps_from_within_layer_steps(
    layer_defs,
    *,
    within_layer_steps: list[int],
) -> list[list[int]]:
    requested = sorted({int(step) for step in within_layer_steps if int(step) >= 0})
    snapshot_steps_by_ss: list[list[int]] = []
    layer_step_offset = 0
    for path_def in layer_defs:
        total_steps = int(path_def.total_steps)
        comp_start = int(layer_step_offset)
        comp_end = int(layer_step_offset) + int(total_steps)
        rel_steps = [
            int(step) - int(comp_start)
            for step in requested
            if int(comp_start) <= int(step) < int(comp_end)
        ]
        snapshot_steps_by_ss.append(rel_steps)
        layer_step_offset += total_steps
    return snapshot_steps_by_ss


def _release_cupy_temporaries() -> None:
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def main(argv=None):
    bind_local_gpu()

    args = parse_args(argv)
    setup = load_sim_setup(args.config, dt_us=args.dt_us, allow_cfl=False)
    project_root, config_path = setup.project_root, setup.config_path
    rc, phys, dt_s = setup.rc, setup.phys, setup.dt_s
    float_type = select_float_type(rc)

    path_config_path = resolve_path(project_root, args.path_config, "")
    out_dir = (project_root / args.out_dir).resolve()
    boundary_viz = None
    if args.boundary_visualization_json:
        boundary_viz_path = resolve_path(project_root, args.boundary_visualization_json, "")
        with open(boundary_viz_path, "r", encoding="utf-8") as f:
            boundary_viz = json.load(f)

    dt_nd = setup.dt_nd
    ctx = build_outer_context(rc, phys, float_type, dt_nd, solver_mode=args.solver_mode)
    n_all = ctx.nx * ctx.ny * ctx.nz

    pipeline_cfg = PipelineConfig.from_ini(path_config_path, num_layers=args.num_layers)
    pipeline_cfg = pipeline_cfg.with_solver_motion(dt_s=dt_s, solver_velocity_mps=rc.laser.v)
    ss_result = build_ss_from_cfg(pipeline_cfg)
    if not ss_result.supersegments:
        print("[done] no supersegments to process.")
        sys.exit(0)

    num_layers = int(ss_result.num_layers)
    ss_per_layer = int(ss_result.ss_per_layer)
    num_supersegments = len(ss_result.supersegments)
    layer_components = _build_layer_components(num_layers=num_layers, ss_per_layer=ss_per_layer)
    ss_components = _build_supersegment_components(num_layers=num_layers, ss_per_layer=ss_per_layer)
    ss_path_defs = build_path_defs_from_components(
        supersegments=ss_result.supersegments,
        components=ss_components,
        dt_s=dt_s,
        len_scale=phys.len_scale,
        coord_scale=ss_result.len_scale / phys.len_scale,
    )
    if len(ss_path_defs) != len(ss_components):
        raise RuntimeError(
            f"Expected {len(ss_components)} supersegment path defs, got {len(ss_path_defs)}."
        )

    layer_path_defs: dict[int, list] = defaultdict(list)
    for ss_idx, pd in enumerate(ss_path_defs):
        layer_idx = int(ss_idx) // int(ss_per_layer)
        layer_path_defs[layer_idx].append(pd)

    if len(layer_path_defs) != len(layer_components):
        raise RuntimeError(
            f"Expected {len(layer_components)} layers of path defs, got {len(layer_path_defs)}."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== Independent Serial Reference Run ===")
    print(f"config:       {config_path}")
    print(f"path-config:  {path_config_path}")
    print(f"dt:           {dt_s:.6e} s  ({dt_s * 1e6:.6f} us)")
    print(f"solver mode:  {args.solver_mode}")
    print(f"reference:    original path order (independent of DAG/component partitioning)")
    print(
        f"phys.len_scale: {phys.len_scale:.4e} m   "
        f"dx_step = {rc.laser.v * dt_s / phys.len_scale:.4e} "
        f"(HERMES ND = {rc.laser.v * dt_s * 1e6:.2f} um/step)"
    )
    print(f"Layers:       {num_layers}")
    print(f"SS/layer:     {ss_per_layer}")
    print(f"Total SS:     {num_supersegments}")
    if boundary_viz is None:
        print(f"snapshot mode: global every {int(args.snap_every_steps)} steps")
    else:
        print(
            "snapshot mode: boundary window from "
            f"{args.boundary_visualization_json}"
        )

    ambient_gpu = cp.full((n_all,), ctx.u0, dtype=float_type)
    h_m = float(rc.level3.h_tuple[0])
    snaps_dir = out_dir / ("snapshots_full_T_K" if bool(args.save_temperature_K) else "snapshots_ser")
    snaps_dir.mkdir(parents=True, exist_ok=True)
    snapshot_stride_steps = int(args.snap_every_steps)
    total_elapsed = 0.0
    saved_snapshot_count = 0
    if boundary_viz is None:
        layer_iterable = range(num_layers)
        target_last_within_layer_step = None
    else:
        target_layer = int(boundary_viz["target_context"]["layer"])
        layer_iterable = [target_layer]
        requested_steps = [
            int(step)
            for step in boundary_viz["snapshot_within_layer_steps"]
            if int(step) >= 0
        ]
        target_last_within_layer_step = max(requested_steps) if requested_steps else None

    for layer_idx in layer_iterable:
        layer_defs = layer_path_defs[int(layer_idx)]
        steps_per_layer = sum(int(pd.total_steps) for pd in layer_defs)
        if boundary_viz is None or int(boundary_viz["target_context"]["layer"]) != int(layer_idx):
            snapshot_steps_by_ss = (
                _build_layer_snapshot_steps_by_supersegment(
                    layer_defs,
                    snap_every_steps=snapshot_stride_steps,
                )
                if boundary_viz is None
                else [[] for _ in layer_defs]
            )
        else:
            snapshot_steps_by_ss = _build_layer_snapshot_steps_from_within_layer_steps(
                layer_defs,
                within_layer_steps=[
                    int(step) for step in boundary_viz["snapshot_within_layer_steps"]
                ],
            )
        print(f"\n[layer {layer_idx + 1}/{num_layers}] starting serial reference...", flush=True)
        t0 = time.perf_counter()
        u_state = ambient_gpu
        layer_step_offset = 0
        for ss_idx_in_layer, path_def in enumerate(layer_defs):
            if (
                boundary_viz is not None
                and target_last_within_layer_step is not None
                and int(layer_step_offset) > int(target_last_within_layer_step)
            ):
                print(
                    f"[layer {layer_idx + 1}/{num_layers}] early exit after requested snapshots",
                    flush=True,
                )
                break
            ss_snapshot_steps = snapshot_steps_by_ss[int(ss_idx_in_layer)]
            max_steps = None
            if (
                boundary_viz is not None
                and ss_snapshot_steps
                and target_last_within_layer_step is not None
                and int(layer_step_offset)
                <= int(target_last_within_layer_step)
                < int(layer_step_offset) + int(path_def.total_steps)
            ):
                max_steps = int(target_last_within_layer_step) - int(layer_step_offset) + 1
            comp_snaps, final_u = run_ss_outer(
                ctx=ctx,
                u_init=u_state,
                x_start=float(path_def.x_start),
                y_start=float(path_def.y_start),
                legs=path_def.legs,
                steps_per_ss=int(path_def.total_steps),
                max_steps=max_steps,
                snapshot_stride_steps=None,
                snapshot_steps=ss_snapshot_steps,
                source_on=True,
            )
            for snap_idx, arr in enumerate(comp_snaps):
                if snap_idx >= len(ss_snapshot_steps):
                    break
                within_layer_step = layer_step_offset + int(ss_snapshot_steps[snap_idx])
                if within_layer_step >= steps_per_layer:
                    break
                fname = f"layer_{layer_idx:02d}_step_{within_layer_step:09d}.npy"
                cropped = crop_snapshot(cp.asnumpy(arr), ctx.nx, ctx.ny, ctx.nz, h_m)
                if bool(args.save_temperature_K):
                    cropped = cropped.astype(np.float64, copy=False) * float(phys.deltaT) + float(phys.Ts)
                np.save(snaps_dir / fname, cropped)
                saved_snapshot_count += 1
            if comp_snaps:
                arr = None
                cropped = None
            layer_step_offset += int(path_def.total_steps)
            u_state = final_u
            del final_u
            del comp_snaps
            _release_cupy_temporaries()
            if (
                boundary_viz is not None
                and target_last_within_layer_step is not None
                and ss_snapshot_steps
                and int(layer_step_offset) > int(target_last_within_layer_step)
            ):
                print(
                    f"[layer {layer_idx + 1}/{num_layers}] early exit after requested snapshots",
                    flush=True,
                )
                break
            if ((ss_idx_in_layer + 1) % 25) == 0 or (ss_idx_in_layer + 1) == len(layer_defs):
                elapsed_now = time.perf_counter() - t0
                print(
                    f"[layer {layer_idx + 1}/{num_layers}] "
                    f"SS {ss_idx_in_layer + 1}/{len(layer_defs)} done  +{elapsed_now:.1f}s",
                    flush=True,
                )
        elapsed = time.perf_counter() - t0
        total_elapsed += elapsed
        print(f"[layer {layer_idx + 1}/{num_layers}] done in {elapsed:.1f} s", flush=True)

    print(f"\n[timing] serial total: {total_elapsed:.3f} s  ({num_layers} layer(s))")
    print(f"Saved snapshots to {snaps_dir}")

    with open(out_dir / "serial_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "serial_total_seconds": total_elapsed,
                "reference_mode": "independent_path_order",
                "num_layers": num_layers,
                "ss_per_layer": ss_per_layer,
                "num_supersegments": num_supersegments,
                "nominal_steps_per_ss": int(ss_path_defs[0].total_steps),
                "snap_every_steps": int(snapshot_stride_steps),
                "boundary_visualization_json": args.boundary_visualization_json,
                "save_temperature_K": bool(args.save_temperature_K),
                "snapshot_dir": str(snaps_dir),
                "saved_snapshot_count": int(saved_snapshot_count),
                "early_exit_after_boundary_snapshots": boundary_viz is not None,
            },
            f,
            indent=2,
        )
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
