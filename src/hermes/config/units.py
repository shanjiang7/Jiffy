# Modified from config.py in hermes-gpu-heat repository.
from __future__ import annotations

import re

# -----------------------------
# Minimal length-unit parsing
# -----------------------------
_SI = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "\u00b5m": 1e-6, "nm": 1e-9}
_num_unit = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([a-zA-Z\u00b5]*)\s*$")


def parse_length_expr(s: str) -> float:
    """Supports '96 * 50um', '12*50 um', '0.001', '1e-4m' -> meters"""
    parts = [p.strip() for p in str(s).split("*")]
    val = 1.0
    for p in parts:
        if not p:
            continue
        m = _num_unit.match(p)
        if not m:
            raise ValueError(f"Cannot parse length token: {p!r}")
        num = float(m.group(1))
        unit = (m.group(2) or "m").strip()
        if unit not in _SI:
            raise ValueError(f"Unknown unit {unit!r} in {p!r}; allowed {list(_SI)}")
        val *= num * _SI[unit]
    return float(val)
