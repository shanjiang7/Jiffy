"""
Shared simulation setup for entry-point scripts.

Every runner (serial reference, parallel run, planning preview, calibration,
comparison) previously repeated the same preamble: resolve the sim config,
load it, build the physics parameters, pick the float type, derive dt. This
module is the single source of truth for that sequence, so the serial
reference and the parallel run can never disagree on physics setup.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hermes.physics.material import phys_parameter
from hermes.runtime.config import load_config
from hermes.utils.path_utils import resolve_path


def repo_root() -> Path:
    """Repository root, derived from the hermes package location (<root>/src/hermes)."""
    import hermes

    return Path(next(iter(hermes.__path__))).resolve().parents[1]


def compute_dt_s(rc, phys=None, *, dt_us: float | None = None, allow_cfl: bool = True) -> float:
    """
    Solver timestep in seconds. Priority: explicit --dt-us override, then the
    CFL condition (if allowed), then [time].dt from the sim config.

    The independent serial reference passes allow_cfl=False: it must not
    silently derive a different dt than the parallel run it is compared to.
    """
    if dt_us is not None:
        return float(dt_us) * 1e-6
    if allow_cfl and rc.time.CFL is not None:
        if phys is None:
            raise ValueError("CFL-derived dt requires phys.")
        return (float(rc.time.CFL) * (float(rc.level1.h_tuple[0]) ** 2)) / float(phys.kappa)
    if rc.time.dt is not None:
        return float(rc.time.dt)
    if allow_cfl:
        raise ValueError("Need either [time].CFL or [time].dt in sim config.")
    raise ValueError(
        "Independent serial reference does not fall back to [time].CFL. "
        "Pass --dt-us explicitly or set [time].dt in the sim config."
    )


@dataclass(frozen=True)
class SimSetup:
    project_root: Path
    config_path: Path
    rc: object
    phys: object
    dt_s: float

    @property
    def dt_nd(self) -> float:
        return float(self.dt_s) / float(self.phys.time_scale)


def load_sim_setup(
    config: str | Path,
    *,
    dt_us: float | None = None,
    allow_cfl: bool = True,
    default_config: str = "configs/examples/sim_ex1.ini",
) -> SimSetup:
    root = repo_root()
    config_path = resolve_path(root, str(config), default_config)
    rc = load_config(config_path)
    t_spot_on = 2.0 * rc.laser.x_span_m / rc.laser.v
    phys = phys_parameter(
        rc.laser.Q, rc.laser.x_span_m, t_spot_on, mat_ch=rc.material.to_override_dict()
    )
    dt_s = compute_dt_s(rc, phys, dt_us=dt_us, allow_cfl=allow_cfl)
    return SimSetup(project_root=root, config_path=config_path, rc=rc, phys=phys, dt_s=dt_s)


def select_float_type(rc):
    """cupy dtype from [run].float_type (imported lazily: GPU-free callers stay GPU-free)."""
    import cupy as cp

    return cp.float64 if rc.float_type_str.lower() == "float64" else cp.float32
