from __future__ import annotations

import math

from dataclasses import dataclass, replace
from pathlib import Path

from hermes.config.units import parse_length_expr
from hermes.utils.laser_io import load_cfg, load_segment_config, SegmentConfig
from hermes.DAG.dependency import DependencyModel


@dataclass(frozen=True)
class DependencyConfig:
    model: DependencyModel
    mock_numerical_source_steps: int
    # Number of source-on deposit steps for the numerical r_eps lookup.
    # 1 reproduces the published single-pulse mock source; a value equal to
    # steps_per_segment ("segment" in the INI) deposits a full segment so
    # r_eps(tau) measures whole-segment influence — the quantity the Table I
    # calibration defines epsilon against.
    lookup_source_on_steps: int = 1


@dataclass(frozen=True)
class DagConfig:
    back_window: int


@dataclass(frozen=True)
class LayerConfig:
    num_layers: int


@dataclass(frozen=True)
class PipelineConfig:
    cfg_path: Path
    effective_motion_step_nd: float | None
    len_scale: float
    
    segment: SegmentConfig
    steps_per_segment: int
    width_roi_m: float
    first_n_points: int
    segments_per_supersegment: int
    
    dependency: DependencyConfig
    dag: DagConfig
    layers: LayerConfig

    def compute_effective_motion_step_nd(self, *, dt_s: float, solver_velocity_mps: float) -> float:
        return float(solver_velocity_mps) * float(dt_s) / float(self.len_scale)

    def require_effective_motion_step_nd(self) -> float:
        if self.effective_motion_step_nd is None:
            raise ValueError(
                "PipelineConfig.effective_motion_step_nd is unset. "
                "Call with_solver_motion(dt_s=..., solver_velocity_mps=...) before building the DAG."
            )
        return float(self.effective_motion_step_nd)

    def with_solver_motion(
        self,
        *,
        dt_s: float,
        solver_velocity_mps: float,
    ) -> "PipelineConfig":
        return self.with_motion_params(
            effective_motion_step_nd=self.compute_effective_motion_step_nd(
                dt_s=dt_s,
                solver_velocity_mps=solver_velocity_mps,
            ),
            segment_velocity_mps=solver_velocity_mps,
        )

    def with_motion_params(
        self,
        *,
        effective_motion_step_nd: float | None = None,
        segment_velocity_mps: float | None = None,
    ) -> "PipelineConfig":
        step_nd = self.effective_motion_step_nd if effective_motion_step_nd is None else float(effective_motion_step_nd)
        segment = self.segment
        if segment_velocity_mps is not None:
            segment = SegmentConfig(
                P_W=segment.P_W,
                V_mps=float(segment_velocity_mps),
                t0_s=segment.t0_s,
                width_m=segment.width_m,
            )
        return replace(self, effective_motion_step_nd=step_nd, segment=segment)

    @classmethod
    def from_ini(cls, path: Path | str, *, num_layers: int | None = None) -> PipelineConfig:
        cfg_path = Path(path).expanduser().resolve()
        cfg = load_cfg(cfg_path)
        
        len_scale = float(cfg.get("trajectory", "len_scale", fallback="1.0"))
        seg_cfg = load_segment_config(cfg)
        
        # We need to know the total number of points so that we can fallback if first_n_points is not set.
        # But wait, first_n_points requires knowing the size of the array. It's safe to default to a huge number,
        # or just 1000000000 since array slicing handles it.
        first_n_points = int(cfg.get("run", "first_n_points", fallback="2000000000"))
        
        model = DependencyModel(
            len_scale=len_scale,
            level_K=float(cfg.get("dependency", "level_K", fallback=str(DependencyModel.level_K))),
            resolution_m=float(cfg.get("dependency", "resolution_m", fallback=str(DependencyModel.resolution_m))),
            bc=str(cfg.get("dependency", "bc", fallback=str(DependencyModel.bc))),
            spacing_m=float(cfg.get("dependency", "spacing_m", fallback=str(DependencyModel.spacing_m))),
            window_x_um=int(cfg.get("dependency", "window_x_um", fallback=str(DependencyModel.window_x_um))),
            window_y_um=int(cfg.get("dependency", "window_y_um", fallback=str(DependencyModel.window_y_um))),
            window_z_um=int(cfg.get("dependency", "window_z_um", fallback=str(DependencyModel.window_z_um))),
            target_patch_step_stride=int(
                cfg.get(
                    "dependency",
                    "target_patch_step_stride",
                    fallback=str(DependencyModel.target_patch_step_stride),
                )
            ),
            pair_test=str(
                cfg.get("dependency", "pair_test", fallback=str(DependencyModel.pair_test))
            ).strip().lower(),
            segment_samples=int(
                cfg.get(
                    "dependency",
                    "segment_samples",
                    fallback=str(DependencyModel.segment_samples),
                )
            ),
        )
        if int(model.target_patch_step_stride) < 1:
            raise ValueError("[dependency].target_patch_step_stride must be >= 1.")
        if model.pair_test not in {"aabb", "chords"}:
            raise ValueError("[dependency].pair_test must be 'aabb' or 'chords'.")
        if int(model.segment_samples) < 2:
            raise ValueError("[dependency].segment_samples must be >= 2.")
        mock_numerical_source_steps = int(
            cfg.get("dependency", "mock_numerical_source_steps", fallback="1")
        )
        if mock_numerical_source_steps < 1:
            raise ValueError("[dependency].mock_numerical_source_steps must be >= 1.")
        

        steps_per_segment = int(cfg.get("run", "steps_per_segment", fallback="200"))
        raw_lookup_src = str(
            cfg.get("dependency", "lookup_source_on_steps", fallback="1")
        ).strip().lower()
        if raw_lookup_src == "segment":
            lookup_source_on_steps = steps_per_segment
        elif raw_lookup_src == "chord":
            # One chord's worth of deposit — matches the geometric unit the
            # chords pair test retains edges for (each chord carries its own
            # deposit time).
            n_chords = max(1, int(model.segment_samples) - 1)
            lookup_source_on_steps = max(1, math.ceil(steps_per_segment / n_chords))
        else:
            lookup_source_on_steps = int(raw_lookup_src)
        if lookup_source_on_steps < 1:
            raise ValueError(
                "[dependency].lookup_source_on_steps must be >= 1, 'segment', or 'chord'."
            )

        return cls(
            cfg_path=cfg_path,
            effective_motion_step_nd=None,
            len_scale=len_scale,
            segment=seg_cfg,
            steps_per_segment=steps_per_segment,
            width_roi_m=float(parse_length_expr(cfg.get("run", "width_roi_m", fallback="0.0001mm"))),
            first_n_points=first_n_points,
            segments_per_supersegment=int(cfg.get("run", "segments_per_supersegment", fallback="8")),
            dependency=DependencyConfig(
                model=model,
                mock_numerical_source_steps=mock_numerical_source_steps,
                lookup_source_on_steps=lookup_source_on_steps,
            ),
            dag=DagConfig(
                back_window=int(cfg.get("dag", "back_window", fallback="100")),
            ),
            layers=LayerConfig(
                num_layers=(num_layers if num_layers is not None
                            else int(cfg.get("layers", "num_layers", fallback="1"))),
            ),
        )
