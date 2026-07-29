"""One-unit figure + geometry checks for the continuous-hybrid path (square)."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/scratch/10226/shawnraul/work/Jiffy/src")
from hermes.laser_path.path_builders import (  # noqa: E402
    build_continuous_hybrid_sections_nd,
    build_square_spiral_nd,
)

MM = 1e-3
LS = 1.0

sections = build_continuous_hybrid_sections_nd(
    origin_m=(0.0, 0.0),
    spiral_side_m=8 * MM,
    track_pitch_m=0.2 * MM,
    raster_width_m=4 * MM,
    raster_line_pitch_m=0.2 * MM,
    gap_m=0.2 * MM,
    tile_gap_m=0.2 * MM,
    repeats=2,
    len_scale=LS,
)

# ---- numeric checks -------------------------------------------------------
full = np.vstack([pts for pts, _ in sections])
legs = np.linalg.norm(np.diff(full, axis=0), axis=1)
print(f"sections: {len(sections)}  all source_on: {all(on for _, on in sections)}")
print(f"total path length: {legs.sum()/MM:.1f} mm")

gaps = [np.linalg.norm(sections[i + 1][0][0] - sections[i][0][-1]) for i in range(len(sections) - 1)]
print(f"max section junction gap: {max(gaps)/MM:.2e} mm (0 = continuous)")

A = build_square_spiral_nd(origin_m=(0, 0), side_m=8 * MM, line_pitch_m=0.4 * MM,
                           len_scale=LS, direction="inward")
B = build_square_spiral_nd(origin_m=(0.2 * MM, 0.2 * MM), side_m=7.6 * MM,
                           line_pitch_m=0.4 * MM, len_scale=LS, direction="outward")
print(f"A->B link length (center): {np.linalg.norm(B[0]-A[-1])/MM:.3f} mm")


def seg_intersect(p1, p2, p3, p4, eps=1e-12):
    """Proper crossing of open segments (shared endpoints don't count)."""
    d1, d2 = p2 - p1, p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < eps:
        return False
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / denom
    return eps < t < 1 - eps and eps < u < 1 - eps


crossings = sum(
    seg_intersect(A[i], A[i + 1], B[j], B[j + 1])
    for i in range(len(A) - 1) for j in range(len(B) - 1)
)
print(f"exact A x B track crossings: {crossings}")

# interleave spacing: sample points on B (away from center), distance to A
dense = []
for i in range(len(B) - 1):
    n = max(2, int(np.linalg.norm(B[i + 1] - B[i]) / (0.1 * MM)))
    dense.append(B[i] + (B[i + 1] - B[i]) * np.linspace(0, 1, n)[:, None])
Bd = np.vstack(dense)
r = np.linalg.norm(Bd - np.array([4 * MM, 4 * MM]), axis=1)
Bd = Bd[r > 1.0 * MM]  # exclude center-link neighborhood


def dist_to_polyline(p, poly):
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
    proj = a + t[:, None] * ab
    return np.min(np.linalg.norm(p - proj, axis=1))


d = np.array([dist_to_polyline(p, A) for p in Bd[::20]])
print(f"B->A interleave spacing (target 0.200 mm): "
      f"min {d.min()/MM:.3f}  median {np.median(d)/MM:.3f}  max {d.max()/MM:.3f} mm")

# bridge clearance: min distance from bridge polyline to both threads
bridge = sections[1][0]
bd = []
for i in range(len(bridge) - 1):
    n = max(2, int(np.linalg.norm(bridge[i + 1] - bridge[i]) / (0.05 * MM)))
    bd.append(bridge[i] + (bridge[i + 1] - bridge[i]) * np.linspace(0.02, 0.98, n)[:, None])
bd = np.vstack(bd)
db = min(dist_to_polyline(p, np.vstack([A, B])) for p in bd[::10])
print(f"bridge min clearance to spiral tracks: {db/MM:.3f} mm (target ~0.2)")

# ---- figure ---------------------------------------------------------------
BLUE, ORANGE, GREEN, GRAY = "#4269D0", "#EFB118", "#3CA951", "#9498A0"
fig, ax = plt.subplots(figsize=(11, 6.4))

spiral1, bridge1, raster1 = sections[0][0], sections[1][0], sections[2][0]
h1 = len(A)
ax.plot(spiral1[:h1, 0] / MM, spiral1[:h1, 1] / MM, color=BLUE, lw=1.1,
        label="spiral thread A (inward)")
ax.plot(spiral1[h1 - 1:, 0] / MM, spiral1[h1 - 1:, 1] / MM, color=ORANGE, lw=1.1,
        label="spiral thread B (outward)")
ax.plot(bridge1[:, 0] / MM, bridge1[:, 1] / MM, color=GRAY, lw=1.6, ls=(0, (4, 2)),
        label="bridge (laser ON)")
ax.plot(raster1[:, 0] / MM, raster1[:, 1] / MM, color=GREEN, lw=1.1,
        label="serpentine raster")
for pts, _ in sections[3:]:
    ax.plot(pts[:, 0] / MM, pts[:, 1] / MM, color=GRAY, lw=0.7, alpha=0.45)

ax.annotate("enter unit 1", xy=(0, 0), xytext=(-2.9, -1.3),
            arrowprops=dict(arrowstyle="->", color="0.25"), fontsize=9)
ax.annotate("U-turn at center", xy=(4, 4), xytext=(-3.6, 4.6),
            arrowprops=dict(arrowstyle="->", color="0.25"), fontsize=9)
ax.annotate("exit via mouth corridor", xy=(0.2, 6.5), xytext=(-4.1, 7.6),
            arrowprops=dict(arrowstyle="->", color="0.25"), fontsize=9)
ax.annotate("to unit 2 (no travel move)", xy=(12.55, 0.0), xytext=(9.2, -1.55),
            arrowprops=dict(arrowstyle="->", color="0.25"), fontsize=9)

ax.annotate("", xy=(0, -0.5), xytext=(8, -0.5), arrowprops=dict(arrowstyle="<->", color="0.4"))
ax.text(4, -0.95, "8 mm double square spiral", ha="center", fontsize=8, color="0.3")
ax.annotate("", xy=(8.2, -0.5), xytext=(12.2, -0.5), arrowprops=dict(arrowstyle="<->", color="0.4"))
ax.text(10.2, -0.95, "4 mm raster", ha="center", fontsize=8, color="0.3")

axins = ax.inset_axes([0.765, 0.56, 0.22, 0.40])
ax.plot([], [])
axins.plot(spiral1[:h1, 0] / MM, spiral1[:h1, 1] / MM, color=BLUE, lw=1.6)
axins.plot(spiral1[h1 - 1:, 0] / MM, spiral1[h1 - 1:, 1] / MM, color=ORANGE, lw=1.6)
axins.plot(bridge1[:, 0] / MM, bridge1[:, 1] / MM, color=GRAY, lw=1.6, ls=(0, (4, 2)))
axins.set_xlim(-0.15, 1.05)
axins.set_ylim(-0.15, 1.05)
axins.set_xticks([]); axins.set_yticks([])
axins.set_title("corner: 0.2 mm interleave + exit", fontsize=8)
ax.indicate_inset_zoom(axins, edgecolor="0.5")

ax.set_aspect("equal")
ax.set_xlim(-4.6, 26)
ax.set_ylim(-2.2, 10.6)
ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
ax.set_title("continuous-hybrid: one unit (unit 2 faded) — single uninterrupted laser-on stroke")
ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
ax.grid(alpha=0.15, lw=0.5)
fig.tight_layout()
out = "/scratch/10226/shawnraul/work/Jiffy/outputs/continuous_hybrid_unit.png"
fig.savefig(out, dpi=160)
print("figure:", out)
