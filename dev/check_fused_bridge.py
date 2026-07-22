#!/usr/bin/env python3
"""
Verify that the fused bridge tracer is equivalent to the per-successor tracers.

A component with several successors used to run one source-off tracer per
successor, each starting from the same end state and following the same legs.
`_build_fused_tracer_run` collapses those into a single run. This check asserts
the invariants that equivalence rests on, so the runtime change cannot silently
alter the corrections it delivers:

  1. each successor's snapshot steps are byte-identical to the per-successor
     `_build_edge_window` result;
  2. every shorter bridge's legs are a prefix of the fused leg list, and all
     bridges share a start point (so the fused run traces the same trajectory);
  3. the fused run is long enough for every successor's window;
  4. every per-successor step resolves to a captured snapshot.

Pure planning arithmetic -- no GPU, no config. Run from the repo root:

    python dev/check_fused_bridge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hermes.motion.types import PathDef, PathLeg  # noqa: E402
from hermes.scripts.segment_correction.core import (  # noqa: E402
    _build_edge_window,
    _build_fused_tracer_run,
)

STEPS_PER_SS = 10


def make_components(n: int, steps_each: int) -> dict[int, PathDef]:
    """A chain of components, each a single leg of `steps_each` steps."""
    return {
        cid: PathDef(
            component_id=cid,
            x_start=float(cid),
            y_start=float(-cid),
            legs=[PathLeg(dx_step=1.0, dy_step=0.0, steps=steps_each)],
            total_steps=steps_each,
            weight=steps_each // STEPS_PER_SS,
        )
        for cid in range(n)
    }


def check(src: int, successors: list[int], n_components: int = 8, steps_each: int = 40) -> None:
    path_def_by_id = make_components(n_components, steps_each)
    ordered = list(range(n_components))
    index = {cid: i for i, cid in enumerate(ordered)}
    horizon_by_dst = {cid: 2 for cid in ordered}
    common = dict(
        ordered_component_ids=ordered,
        component_index=index,
        path_def_by_id=path_def_by_id,
        steps_per_ss=STEPS_PER_SS,
        snapshot_stride_steps=5,
        snapshot_steps_by_component=None,
    )

    per_successor = {
        dst: _build_edge_window(
            src_component=src, dst_component=dst, horizon_ss=horizon_by_dst[dst], **common
        )
        for dst in successors
    }
    (fx, fy, fused_legs, fused_max_steps, union_steps, per_dst_steps) = _build_fused_tracer_run(
        src_component=src, dst_components=successors,
        horizon_ss_by_dst=horizon_by_dst, **common
    )

    for dst, (bx, by, legs, max_steps, snapshot_steps, _gap) in per_successor.items():
        assert per_dst_steps[dst] == snapshot_steps, (
            f"src={src} dst={dst}: snapshot steps differ\n"
            f"  fused: {per_dst_steps[dst]}\n  single: {snapshot_steps}"
        )
        assert fused_legs[: len(legs)] == legs, f"src={src} dst={dst}: legs are not a prefix"
        assert (bx, by) == (fx, fy), f"src={src} dst={dst}: start point differs"
        assert fused_max_steps >= max_steps, f"src={src} dst={dst}: fused run too short"

    expected_max = max(v[3] for v in per_successor.values())
    assert fused_max_steps == expected_max, "fused run length is not the max of the successors'"

    index_by_step = {s: i for i, s in enumerate(union_steps)}
    for dst, steps in per_dst_steps.items():
        for step in steps:
            assert step in index_by_step, f"src={src} dst={dst}: step {step} missing from union"

    single_total = sum(v[3] for v in per_successor.values())
    saved = single_total - fused_max_steps
    print(
        f"  src={src} successors={successors}: "
        f"{len(successors)} tracer(s), {single_total} steps -> 1 tracer, "
        f"{fused_max_steps} steps  (saves {saved} steps, "
        f"{100.0 * saved / single_total:.0f}%)"
    )


def check_per_edge(src: int, successors: list[int], n_components: int = 8,
                   steps_each: int = 40) -> None:
    """Per-edge horizons truncate each successor to its own reach while keeping
    the fused-run invariants: run length covers every window, longest legs are a
    prefix superset, every per-successor step is captured."""
    path_def_by_id = make_components(n_components, steps_each)
    ordered = list(range(n_components))
    index = {cid: i for i, cid in enumerate(ordered)}
    common = dict(
        ordered_component_ids=ordered,
        component_index=index,
        path_def_by_id=path_def_by_id,
        steps_per_ss=STEPS_PER_SS,
        snapshot_stride_steps=5,
        snapshot_steps_by_component=None,
    )
    # A distant source reaches only 1 SS into each destination, where the
    # destination's own correction window (its nearest predecessor's reach)
    # would be 3. The shallow per-edge depth must shrink the trace.
    shallow = {dst: 1 for dst in successors}
    deep = {dst: 3 for dst in successors}

    (fx, fy, fused_legs, fused_max_steps, union_steps, per_dst_steps) = _build_fused_tracer_run(
        src_component=src, dst_components=successors,
        horizon_ss_by_dst=shallow, **common
    )
    # each successor gets its own single-successor edge-horizon build
    for dst in successors:
        bx, by, legs, max_steps, snap_steps, _gap = _build_edge_window(
            src_component=src, dst_component=dst, horizon_ss=1, **common
        )
        assert per_dst_steps[dst] == snap_steps, f"per-edge src={src} dst={dst}: steps differ"
        assert fused_legs[: len(legs)] == legs, f"per-edge src={src} dst={dst}: legs not a prefix"
        assert (bx, by) == (fx, fy), f"per-edge src={src} dst={dst}: start differs"
        assert fused_max_steps >= max_steps, f"per-edge src={src} dst={dst}: run too short"
    index_by_step = {s: i for i, s in enumerate(union_steps)}
    for dst, steps in per_dst_steps.items():
        for step in steps:
            assert step in index_by_step, f"per-edge src={src} dst={dst}: step {step} missing"

    # And truncation actually happened: compare against the full-window build.
    (_, _, _, full_max, _, _) = _build_fused_tracer_run(
        src_component=src, dst_components=successors,
        horizon_ss_by_dst=deep, **common
    )
    print(f"  src={src} successors={successors}: per-edge {fused_max_steps} steps "
          f"vs full-window {full_max} steps  (truncated {full_max - fused_max_steps})")
    assert fused_max_steps <= full_max, "per-edge horizon should never trace deeper"


def main() -> None:
    print("fused bridge tracer equivalence:")
    check(src=0, successors=[1])            # single successor: must be unchanged
    check(src=0, successors=[1, 2])
    check(src=2, successors=[3, 4])         # the rank-2 case from job 853164
    check(src=1, successors=[2, 3, 4])
    check(src=0, successors=[3])            # non-adjacent single successor
    check(src=0, successors=[2, 1])         # successors given out of path order
    print("per-edge horizon truncation:")
    check_per_edge(src=1, successors=[2, 3])
    check_per_edge(src=0, successors=[1, 2, 3])
    check_per_edge(src=2, successors=[3, 4])
    print("all checks passed")


if __name__ == "__main__":
    main()
