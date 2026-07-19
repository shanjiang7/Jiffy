from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathComplexityOptions:
    target_rel_l2: float
    iterate: bool = False
    max_iterations: int = 1


@dataclass(frozen=True)
class DiagnosticCheckOptions:
    gamma: float = 2.0
    tol: float = 1e-4
    snap_every_steps: int = 100
    compare: str = "rel_l2"


def _read_ini(path: Path, *, required_section: str) -> configparser.ConfigParser:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    parser = configparser.ConfigParser()
    parser.read(cfg_path)
    if not parser.has_section(required_section):
        raise ValueError(f"Missing [{required_section}] section in {cfg_path}")
    return parser


def load_path_complexity_options(path: Path) -> PathComplexityOptions:
    parser = _read_ini(path, required_section="path_complexity")
    section = "path_complexity"
    target_rel_l2 = float(parser.get(section, "target_rel_l2", fallback="1e-4"))
    if target_rel_l2 <= 0.0:
        raise ValueError("[path_complexity].target_rel_l2 must be > 0.")
    max_iterations = int(parser.get(section, "max_iterations", fallback="1"))
    if max_iterations < 1:
        raise ValueError("[path_complexity].max_iterations must be >= 1.")
    return PathComplexityOptions(
        target_rel_l2=float(target_rel_l2),
        iterate=parser.getboolean(section, "iterate", fallback=False),
        max_iterations=int(max_iterations),
    )


def load_diagnostic_check_options(path: Path) -> DiagnosticCheckOptions:
    parser = _read_ini(path, required_section="diagnostic_check")
    section = "diagnostic_check"
    gamma = float(parser.get(section, "gamma", fallback="2.0"))
    tol = float(parser.get(section, "tol", fallback="1e-4"))
    snap_every_steps = int(parser.get(section, "snap_every_steps", fallback="100"))
    compare = str(parser.get(section, "compare", fallback="rel_l2")).strip().lower()
    if gamma <= 1.0:
        raise ValueError("[diagnostic_check].gamma must be > 1.")
    if tol <= 0.0:
        raise ValueError("[diagnostic_check].tol must be > 0.")
    if snap_every_steps < 1:
        raise ValueError("[diagnostic_check].snap_every_steps must be >= 1.")
    if compare != "rel_l2":
        raise ValueError("[diagnostic_check].compare currently supports only 'rel_l2'.")
    return DiagnosticCheckOptions(
        gamma=float(gamma),
        tol=float(tol),
        snap_every_steps=int(snap_every_steps),
        compare=str(compare),
    )


__all__ = [
    "DiagnosticCheckOptions",
    "PathComplexityOptions",
    "load_diagnostic_check_options",
    "load_path_complexity_options",
]
