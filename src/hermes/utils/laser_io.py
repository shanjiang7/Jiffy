"""
I/O utilities for segments and supersegments.

Functions
---------
load_cfg              – load a ConfigParser from an INI file
load_segment_config   – read [run] section into SegmentConfig
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hermes.utils.segment_types import Segment, SuperSegment


@dataclass
class SegmentConfig:
    """Configuration parameters for building Segment objects from config file."""
    P_W: float
    V_mps: float
    t0_s: float
    width_m: float


def load_cfg(path: Path | None = None) -> configparser.ConfigParser:
    """Load a ConfigParser from the given INI path.

    Supports lightweight inheritance via:

    [include]
    base = other.ini

    Included files are resolved relative to the current INI and loaded first;
    the current file then overrides any inherited values.
    """
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config" / "laser_path.ini"
    else:
        path = Path(path).expanduser().resolve()
    return _load_cfg_recursive(path, visited=set())


def _new_cfg() -> configparser.ConfigParser:
    return configparser.ConfigParser(inline_comment_prefixes=("#", ";"))


def _merge_cfg(dst: configparser.ConfigParser, src: configparser.ConfigParser) -> None:
    for section in src.sections():
        if section == "include":
            continue
        if not dst.has_section(section):
            dst.add_section(section)
        for key, value in src.items(section):
            dst.set(section, key, value)


def _resolve_include_paths(raw_cfg: configparser.ConfigParser, cfg_path: Path) -> list[Path]:
    if not raw_cfg.has_section("include"):
        return []
    include_val = raw_cfg.get("include", "base", fallback="").strip()
    if not include_val:
        return []
    paths: list[Path] = []
    for token in include_val.split(","):
        rel = token.strip()
        if not rel:
            continue
        inc = Path(rel).expanduser()
        if not inc.is_absolute():
            inc = (cfg_path.parent / inc).resolve()
        paths.append(inc)
    return paths


def _load_cfg_recursive(path: Path, visited: set[Path]) -> configparser.ConfigParser:
    cfg = _new_cfg()
    if path in visited:
        raise ValueError(f"Cyclic config include detected at {path}")
    visited = set(visited)
    visited.add(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path} (referenced directly or via an "
            "[include] chain). A missing include would otherwise silently "
            "fall back to built-in defaults that differ from path_base.ini."
        )

    raw_cfg = _new_cfg()
    # Some INI comments contain non-ASCII symbols (e.g., Delta).
    # Always read with UTF-8 to avoid locale-dependent decode failures.
    with open(path, encoding="utf-8") as f:
        raw_cfg.read_file(f)

    merged = _new_cfg()
    for inc_path in _resolve_include_paths(raw_cfg, path):
        inc_cfg = _load_cfg_recursive(inc_path, visited=visited)
        _merge_cfg(merged, inc_cfg)
    _merge_cfg(merged, raw_cfg)
    return merged


def load_segment_config(cfg) -> SegmentConfig:
    """Load segment configuration from a ConfigParser ([run] section)."""
    from hermes.config.units import parse_length_expr
    return SegmentConfig(
        P_W=float(cfg.get("run", "segment_P_W", fallback="200.0")),
        V_mps=float(cfg.get("run", "segment_V_mps", fallback="1.0")),
        t0_s=float(cfg.get("run", "segment_t0_s", fallback="0.0")),
        width_m=float(parse_length_expr(cfg.get("run", "width_m", fallback="0.001"))),
    )
