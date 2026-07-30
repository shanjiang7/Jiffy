# Modified from path_builders.py in hermes-gpu-heat repository.
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

PathSectionND = tuple[np.ndarray, bool]


_DIR_TOKENS = {
    "+x": (1, 0), "x+": (1, 0), "posx": (1, 0),
    "-x": (-1, 0), "x-": (-1, 0), "negx": (-1, 0),
    "+y": (0, 1), "y+": (0, 1), "posy": (0, 1),
    "-y": (0, -1), "y-": (0, -1), "negy": (0, -1),
}


def _parse_dir_to_unit(dir_str: str) -> tuple[float, float]:
    """
    Parse 'dir' like '+y', '-x' (aliases 'x+', 'posx', ...) or a diagonal
    combination '+y,+x' into a unit vector (vx, vy).
    """
    sx = 0
    sy = 0
    for t in (t.strip().lower() for t in dir_str.split(",") if t.strip()):
        if t not in _DIR_TOKENS:
            raise ValueError(f"Unknown direction token in dir={dir_str!r}: {t!r}")
        dx, dy = _DIR_TOKENS[t]
        sx += dx
        sy += dy
    if sx == 0 and sy == 0:
        raise ValueError(f"dir={dir_str!r} results in zero direction.")
    n = math.hypot(sx, sy)
    return sx / n, sy / n


def _to_nd(x_m: float, len_scale: float) -> float:
    return float(x_m) / float(len_scale)


# ---------- 1) single-line scan ----------
def build_single_line_nd(
    *,
    start_xy_m: tuple[float, float],
    length_m: float,
    dir_str: str | None,
    len_scale: float,
    vx: float | None = None,
    vy: float | None = None,
) -> np.ndarray:
    """
    Returns 2-point polyline (start, end) in NON-DIMENSIONAL units.
    dir_str supports '+x', '-x', '+y', '-y', and diagonals like '+y,+x'.
    Alternatively, pass (vx, vy) as a unit vector (one of dir_str or vx/vy).
    """
    if dir_str is not None:
        ux, uy = _parse_dir_to_unit(dir_str)
    else:
        if vx is None or vy is None:
            raise ValueError("Provide either dir_str or (vx, vy).")
        n = math.hypot(vx, vy)
        if n == 0:
            raise ValueError("(vx, vy) must be non-zero.")
        ux, uy = vx / n, vy / n

    x0_m, y0_m = float(start_xy_m[0]), float(start_xy_m[1])
    x1_m = x0_m + float(length_m) * ux
    y1_m = y0_m + float(length_m) * uy

    waypoints_m = np.array([[x0_m, y0_m], [x1_m, y1_m]], dtype=float)
    waypoints_nd = waypoints_m / float(len_scale)
    return waypoints_nd


# ---------- 2) segments (piecewise) ----------
def build_segments_nd(
    *,
    start_xy_m: tuple[float, float],
    segments: list[dict],
    repeat: int = 1,
    len_scale: float,
) -> np.ndarray:
    x_m, y_m = start_xy_m
    path: list[tuple[float, float]] = [(x_m, y_m)]
    for _ in range(max(1, int(repeat))):
        for seg in segments:
            L = float(seg["length_m"])
            if "dir" in seg:
                ux, uy = _parse_dir_to_unit(seg["dir"])
            else:
                vx, vy = float(seg["vx"]), float(seg["vy"])
                nrm = (vx * vx + vy * vy) ** 0.5
                if nrm == 0.0:
                    raise ValueError("segment with zero direction")
                ux, uy = vx / nrm, vy / nrm
            x_m += ux * L
            y_m += uy * L
            path.append((x_m, y_m))
    return np.column_stack(
        [
            [_to_nd(px, len_scale) for px, _ in path],
            [_to_nd(py, len_scale) for _, py in path],
        ]
    )


# ---------- 3) raster / serpentine ----------
def build_raster_nd(
    *,
    origin_m: tuple[float, float],
    width_m: float,
    height_m: float,
    line_pitch_m: float | None = None,
    passes: int | None = None,
    x_dir_sign: int = +1,
    x_major: bool = True,
    len_scale: float,
) -> np.ndarray:
    x0_m, y0_m = origin_m

    if passes is None and line_pitch_m is None:
        raise ValueError("Provide either passes or line_pitch_m for raster.")
    if passes is not None and line_pitch_m is None:
        passes = int(passes)
        if passes < 1:
            raise ValueError("passes must be >= 1")
        line_pitch_m = height_m / max(passes - 1, 1) if passes > 1 else 0.0
    elif passes is None and line_pitch_m is not None:
        if height_m <= 0:
            passes = 1
            line_pitch_m = 0.0
        else:
            passes = int(round(height_m / line_pitch_m)) + 1
    else:
        passes = int(passes)

    pts: list[tuple[float, float]] = []
    dir_sign = int(np.sign(x_dir_sign)) or +1

    for p in range(passes):
        step_val = p * float(line_pitch_m)
        if x_major:
            y_m = y0_m + step_val
            if dir_sign > 0:
                x_start, x_end = x0_m, x0_m + width_m
            else:
                x_start, x_end = x0_m + width_m, x0_m
            pts.append((x_start, y_m))
            pts.append((x_end, y_m))
        else:
            x_m = x0_m + step_val
            if dir_sign > 0:
                y_start, y_end = y0_m, y0_m + width_m
            else:
                y_start, y_end = y0_m + width_m, y0_m
            pts.append((x_m, y_start))
            pts.append((x_m, y_end))

        dir_sign *= -1

    return np.column_stack(
        [
            [_to_nd(px, len_scale) for px, _ in pts],
            [_to_nd(py, len_scale) for _, py in pts],
        ]
    )


def build_island_raster_sections_nd(
    *,
    origin_m: tuple[float, float],
    island_width_m: float,
    island_height_m: float,
    line_pitch_m: float,
    rows: int,
    cols: int,
    island_gap_x_m: float = 0.0,
    island_gap_y_m: float = 0.0,
    x_major: bool = True,
    raster_dir_sign: int = +1,
    raster_dir_mode: str = "fixed",
    snake: bool = True,
    len_scale: float,
) -> list[PathSectionND]:
    if island_width_m <= 0.0 or island_height_m <= 0.0:
        raise ValueError("island_width_m and island_height_m must be > 0")
    if line_pitch_m <= 0.0:
        raise ValueError("line_pitch_m must be > 0")
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")
    raster_dir_mode = str(raster_dir_mode).strip().lower()
    if raster_dir_mode not in {"fixed", "checkerboard"}:
        raise ValueError("raster_dir_mode must be 'fixed' or 'checkerboard'")

    x0_m, y0_m = float(origin_m[0]), float(origin_m[1])
    sections: list[PathSectionND] = []

    for row_idx in range(int(rows)):
        if snake and (row_idx % 2 == 1):
            col_iter = range(int(cols) - 1, -1, -1)
        else:
            col_iter = range(int(cols))

        for col_idx in col_iter:
            island_origin_m = (
                x0_m + col_idx * (float(island_width_m) + float(island_gap_x_m)),
                y0_m + row_idx * (float(island_height_m) + float(island_gap_y_m)),
            )
            island_dir_sign = int(raster_dir_sign)
            if raster_dir_mode == "checkerboard" and ((row_idx + col_idx) % 2 == 1):
                island_dir_sign *= -1

            raster_nd = build_raster_nd(
                origin_m=island_origin_m,
                width_m=float(island_width_m) if x_major else float(island_height_m),
                height_m=float(island_height_m) if x_major else float(island_width_m),
                line_pitch_m=float(line_pitch_m),
                passes=None,
                x_dir_sign=island_dir_sign,
                x_major=bool(x_major),
                len_scale=len_scale,
            )

            if sections:
                prev_last = sections[-1][0][-1]
                next_first = raster_nd[0]
                sections.append((np.vstack([prev_last, next_first]), False))

            sections.append((raster_nd, True))

    return sections


# ---------- 4) explicit waypoints ----------
def build_waypoints_nd(*, waypoints_m: list[tuple[float, float]], close_loop: bool = False, len_scale: float) -> np.ndarray:
    pts = list(waypoints_m)
    if close_loop and len(pts) >= 2:
        pts.append(pts[0])
    return np.column_stack(
        [
            [_to_nd(px, len_scale) for px, _ in pts],
            [_to_nd(py, len_scale) for _, py in pts],
        ]
    )


def concatenate_sections_nd(sections: list[PathSectionND]) -> np.ndarray:
    pts: list[np.ndarray] = []
    for section_pts, _ in sections:
        arr = np.asarray(section_pts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"section points must have shape (N,2); got {arr.shape}")
        if arr.shape[0] == 0:
            continue
        if not pts:
            pts.append(arr.copy())
            continue
        last_pt = pts[-1][-1]
        if np.allclose(last_pt, arr[0]):
            pts.append(arr[1:].copy())
        else:
            pts.append(arr.copy())
    if not pts:
        return np.zeros((0, 2), dtype=float)
    return np.vstack(pts)


def _hilbert_rot(n: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def _hilbert_d2xy(order: int, d: int) -> tuple[int, int]:
    n = 1 << int(order)
    x = 0
    y = 0
    t = int(d)
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _hilbert_rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def build_hilbert_nd(
    *,
    origin_m: tuple[float, float],
    size_m: float,
    order: int,
    len_scale: float,
) -> np.ndarray:
    if size_m <= 0.0:
        raise ValueError("size_m must be > 0")
    order = int(order)
    if order < 1:
        raise ValueError("order must be >= 1")

    grid_n = 1 << order
    step_m = float(size_m) / float(grid_n - 1)
    x0_m, y0_m = float(origin_m[0]), float(origin_m[1])

    pts_m: list[tuple[float, float]] = []
    for d in range(grid_n * grid_n):
        ix, iy = _hilbert_d2xy(order, d)
        pts_m.append((x0_m + ix * step_m, y0_m + iy * step_m))

    return np.column_stack(
        [
            [_to_nd(px, len_scale) for px, _ in pts_m],
            [_to_nd(py, len_scale) for _, py in pts_m],
        ]
    )


def build_square_spiral_outward_nd(
    *,
    origin_m: tuple[float, float],
    side_m: float,
    line_pitch_m: float,
    len_scale: float,
) -> np.ndarray:
    if side_m <= 0.0:
        raise ValueError("side_m must be > 0")
    if line_pitch_m <= 0.0:
        raise ValueError("line_pitch_m must be > 0")
    if line_pitch_m > side_m:
        raise ValueError("line_pitch_m must be <= side_m")

    x0_m, y0_m = float(origin_m[0]), float(origin_m[1])
    pitch = float(line_pitch_m)
    tol = 1e-12 * max(1.0, abs(side_m))
    directions = (
        (1.0, 0.0),   # east
        (0.0, 1.0),   # north
        (-1.0, 0.0),  # west
        (0.0, -1.0),  # south
    )

    # Define the canonical inward spiral as:
    # start from the lower-left corner and rotate counterclockwise.
    x_cur = x0_m
    y_cur = y0_m
    inward_pts: list[tuple[float, float]] = [(x_cur, y_cur)]
    cur_len = float(side_m)
    dir_idx = 0
    while cur_len > tol:
        dx, dy = directions[dir_idx % 4]
        x_next = x_cur + dx * cur_len
        y_next = y_cur + dy * cur_len
        if abs(x_next - x_cur) > tol or abs(y_next - y_cur) > tol:
            inward_pts.append((x_next, y_next))
        x_cur, y_cur = x_next, y_next
        dir_idx += 1
        if dir_idx % 2 == 0:
            cur_len -= pitch

    outward_pts = list(reversed(inward_pts))
    return np.column_stack(
        [
            [_to_nd(px, len_scale) for px, _ in outward_pts],
            [_to_nd(py, len_scale) for _, py in outward_pts],
        ]
    )


def build_square_spiral_nd(
    *,
    origin_m: tuple[float, float],
    side_m: float,
    line_pitch_m: float,
    len_scale: float,
    direction: str = "outward",
) -> np.ndarray:
    spiral_outward_nd = build_square_spiral_outward_nd(
        origin_m=origin_m,
        side_m=side_m,
        line_pitch_m=line_pitch_m,
        len_scale=len_scale,
    )
    direction = direction.strip().lower()
    if direction == "outward":
        return spiral_outward_nd
    if direction == "inward":
        return spiral_outward_nd[::-1].copy()
    raise ValueError("spiral direction must be 'outward' or 'inward'")


def build_double_square_spiral_nd(
    *,
    origin_m: tuple[float, float],
    side_m: float,
    track_pitch_m: float,
    len_scale: float,
) -> np.ndarray:
    """Interleaved double square spiral: in on thread A, out on thread B.

    Thread A is the standard inward square spiral from the lower-left corner
    (rings at even multiples of the track pitch), built with twice the track
    pitch so its own rings are 2p apart. Thread B is the same spiral started
    one track pitch further in (rings at odd multiples), traversed outward.
    Both threads co-rotate counterclockwise, so ring k of one thread nests
    strictly between rings k and k+1 of the other and the tracks never
    cross; the composite fill has uniform track_pitch_m spacing. The two
    inner endpoints are adjacent near the tile center, so the A->B link is
    a short straight leg. The path enters at origin (the corner) and exits
    at origin + (p, p), whose only track-free escape is the spiral mouth
    corridor running up the left side at x = origin_x + p.
    """
    if side_m <= 0.0:
        raise ValueError("side_m must be > 0")
    p_m = float(track_pitch_m)
    if p_m <= 0.0 or 4.0 * p_m >= float(side_m):
        raise ValueError("track_pitch_m must be in (0, side_m / 4)")

    x0_m, y0_m = float(origin_m[0]), float(origin_m[1])
    thread_a = build_square_spiral_nd(
        origin_m=(x0_m, y0_m),
        side_m=float(side_m),
        line_pitch_m=2.0 * p_m,
        len_scale=len_scale,
        direction="inward",
    )
    thread_b = build_square_spiral_nd(
        origin_m=(x0_m + p_m, y0_m + p_m),
        side_m=float(side_m) - 2.0 * p_m,
        line_pitch_m=2.0 * p_m,
        len_scale=len_scale,
        direction="outward",
    )
    return np.vstack([thread_a, thread_b])


def build_continuous_hybrid_sections_nd(
    *,
    origin_m: tuple[float, float],
    spiral_side_m: float,
    track_pitch_m: float,
    raster_width_m: float,
    raster_line_pitch_m: float,
    gap_m: float = 0.0,
    tile_gap_m: float = 0.0,
    repeats: int = 1,
    len_scale: float,
) -> list[PathSectionND]:
    """Continuous spiral+raster path: one uninterrupted laser-on stroke.

    Each unit is a double square spiral (enter at the lower-left corner,
    exit one pitch inside it) bridged to a vertical serpentine raster. The
    exit bridge runs north through the spiral's track-free mouth corridor,
    then east one pitch above the tile's top track into the raster's first
    line — every bridge leg is parallel to (or a collinear extension of)
    the fill at >= one track pitch, so nothing is retraced and no tracks
    cross. The raster's exit and the next unit's corner entry both lie on
    the baseline, so consecutive units chain with a short baseline leg.
    All sections carry source_on=True: there is no travel move anywhere.
    """
    if spiral_side_m <= 0.0:
        raise ValueError("spiral_side_m must be > 0")
    if raster_width_m <= 0.0:
        raise ValueError("raster_width_m must be > 0")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    x0_m, y0_m = float(origin_m[0]), float(origin_m[1])
    s_m = float(spiral_side_m)
    p_m = float(track_pitch_m)
    unit_span_m = s_m + float(gap_m) + float(raster_width_m) + float(tile_gap_m)

    sections: list[PathSectionND] = []
    prev_end_nd: np.ndarray | None = None
    for unit_idx in range(int(repeats)):
        ux_m = x0_m + unit_idx * unit_span_m
        spiral_nd = build_double_square_spiral_nd(
            origin_m=(ux_m, y0_m),
            side_m=s_m,
            track_pitch_m=p_m,
            len_scale=len_scale,
        )

        raster_x0_m = ux_m + s_m + float(gap_m)
        raster_nd = build_raster_nd(
            origin_m=(raster_x0_m, y0_m),
            width_m=s_m,
            height_m=float(raster_width_m),
            line_pitch_m=float(raster_line_pitch_m),
            passes=None,
            x_dir_sign=-1,  # enter the first line from the top
            x_major=False,
            len_scale=len_scale,
        )

        if prev_end_nd is not None:
            sections.append((np.vstack([prev_end_nd, spiral_nd[0]]), True))
        sections.append((spiral_nd, True))

        # Exit bridge: north through the mouth corridor (x = ux + p, track-free
        # by construction), east at one pitch above the top track, then down
        # into the raster's first line as a collinear extension.
        bridge_m = np.array(
            [
                [ux_m + p_m, y0_m + p_m],
                [ux_m + p_m, y0_m + s_m + p_m],
                [raster_x0_m, y0_m + s_m + p_m],
                [raster_x0_m, y0_m + s_m],
            ]
        )
        bridge_nd = bridge_m / float(len_scale)
        if not np.allclose(bridge_nd[0], spiral_nd[-1]):
            raise AssertionError("double-spiral exit does not meet the bridge")
        sections.append((bridge_nd, True))
        sections.append((raster_nd, True))
        prev_end_nd = raster_nd[-1]

    return sections


# --------- 5) Build from picture  --------
def _require_picture_deps():
    try:
        from matplotlib.path import Path as MplPath  # noqa: F401
        from skimage import io, color, filters, measure  # noqa: F401
        from scipy.interpolate import splprep, splev  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Picture mode requires scikit-image, scipy, and matplotlib. "
            "Install them or avoid [path.picture]."
        ) from e


def extract_image_boundary(
    img_path: str | Path,
    len_scale: float,
    horizontal_length: float,
    vertical_length: float | None = None,
):
    _require_picture_deps()
    from matplotlib.path import Path as MplPath
    from skimage import io, color, filters, measure
    from scipy.interpolate import splprep, splev

    img = io.imread(str(img_path))

    if img.ndim == 3 and img.shape[2] == 4:
        img = color.rgba2rgb(img)

    if img.ndim == 3:
        gray_img = color.rgb2gray(img)
    else:
        gray_img = img.astype(np.float32)
        if gray_img.max() > 1.0:
            m = gray_img.max()
            gray_img = gray_img / (m if m != 0 else 1.0)

    thresh = filters.threshold_otsu(gray_img)
    binary_img = gray_img > thresh

    contours = measure.find_contours(binary_img, level=0.5)
    if not contours:
        raise ValueError(f"No contour found in image: {img_path}")
    contour = max(contours, key=len)

    height = binary_img.shape[0]
    boundary_coords = np.array(contour, dtype=float)
    boundary_coords[:, 0] = height - boundary_coords[:, 0]

    tck, _ = splprep([boundary_coords[:, 1], boundary_coords[:, 0]], s=0, per=True)
    u_new = np.linspace(0.0, 1.0, 200)
    x_new, y_new = splev(u_new, tck)

    x_new = np.asarray(x_new, dtype=float)
    y_new = np.asarray(y_new, dtype=float)

    horizontal_length_nd = float(horizontal_length) / float(len_scale)
    if vertical_length is None:
        rngx = (x_new.max() - x_new.min())
        rngy = (y_new.max() - y_new.min())
        aspect = (rngy / (rngx + 1e-15))
        vertical_length_nd = aspect * horizontal_length_nd
    else:
        vertical_length_nd = float(vertical_length) / float(len_scale)

    x_new_nd = (x_new - x_new.min()) / (x_new.max() - x_new.min() + 1e-15) * horizontal_length_nd
    y_new_nd = (y_new - y_new.min()) / (y_new.max() - y_new.min() + 1e-15) * vertical_length_nd

    boundary_path = MplPath(np.column_stack((x_new_nd, y_new_nd)))
    return x_new_nd, y_new_nd, boundary_path


def _vertical_line_polygon_intersections_y(x_poly: np.ndarray, y_poly: np.ndarray, x0: float) -> list[float]:
    """
    Compute y-coordinates where the vertical line x=x0 intersects the polygon boundary.
    """
    xs = np.asarray(x_poly, dtype=float)
    ys = np.asarray(y_poly, dtype=float)
    if xs.ndim != 1 or ys.ndim != 1 or xs.shape[0] != ys.shape[0]:
        raise ValueError("x_poly/y_poly must be 1D arrays of equal length")
    n = xs.shape[0]
    out: list[float] = []
    for i in range(n):
        x1, y1 = float(xs[i]), float(ys[i])
        x2, y2 = float(xs[(i + 1) % n]), float(ys[(i + 1) % n])
        if x2 == x1:
            continue
        xmin = x1 if x1 < x2 else x2
        xmax = x2 if x1 < x2 else x1
        if not (xmin <= x0 < xmax):
            continue
        t = (x0 - x1) / (x2 - x1)
        y = y1 + t * (y2 - y1)
        out.append(float(y))
    out.sort()
    return out


def create_path_columns(
    x_poly_nd: np.ndarray,
    y_poly_nd: np.ndarray,
    boundary_path,
    *,
    dx_nd: float,
    dy_nd: float,
) -> list[list[float]]:
    """
    Column (x) scan using polygon intersections to find in-shape y-interval endpoints.
    """
    dx_nd = float(dx_nd)
    dy_nd = float(dy_nd)
    if not (dx_nd > 0.0 and dy_nd > 0.0):
        raise ValueError("dx_nd and dy_nd must be > 0")

    x_min = float(np.min(x_poly_nd))
    x_max = float(np.max(x_poly_nd))
    y_min = float(np.min(y_poly_nd))
    y_max = float(np.max(y_poly_nd))

    n_cols = int(np.floor((x_max - x_min) / dx_nd)) + 1
    n_cols = max(1, n_cols)
    x_cols = x_min + np.arange(n_cols, dtype=float) * dx_nd
    if x_cols[-1] < x_max and (x_max - x_cols[-1]) > 1e-12 * max(1.0, abs(x_max)):
        x_cols = np.append(x_cols, x_max)

    pts: list[list[float]] = []
    for col_idx, x in enumerate(x_cols):
        ys = _vertical_line_polygon_intersections_y(x_poly_nd, y_poly_nd, float(x))
        if len(ys) < 2:
            continue
        intervals = []
        for j in range(0, len(ys) - 1, 2):
            y0, y1 = float(ys[j]), float(ys[j + 1])
            lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
            lo = max(lo, y_min)
            hi = min(hi, y_max)
            if hi > lo:
                intervals.append((lo, hi))
        if not intervals:
            continue

        forward = (col_idx % 2 == 0)
        if not forward:
            intervals = list(reversed(intervals))

        for (lo, hi) in intervals:
            y_start, y_end = (lo, hi) if forward else (hi, lo)
            step = dy_nd if y_end >= y_start else -dy_nd
            ys_line = [y_start]
            y_cur = y_start
            while True:
                y_next = y_cur + step
                if (step > 0 and y_next >= y_end) or (step < 0 and y_next <= y_end):
                    break
                ys_line.append(y_next)
                y_cur = y_next
            if ys_line[-1] != y_end:
                ys_line.append(y_end)
            for y in ys_line:
                if boundary_path.contains_point((float(x), float(y))):
                    pts.append([float(x), float(y)])

    return pts


def build_picture_nd(
    *,
    img_path: str | Path,
    horizontal_length_m: float,
    len_scale: float,
    vertical_length_m: float | None = None,
    column_res_x_m: float | None = None,
    column_res_y_m: float | None = None,
    column_res_m: float | None = None,
    anchor_to_first: bool = True,
    n: int | None = None,
    step_nd: float | None = None,
) -> np.ndarray:
    x_new_nd, y_new_nd, boundary_path = extract_image_boundary(
        img_path=img_path,
        len_scale=len_scale,
        horizontal_length=horizontal_length_m,
        vertical_length=vertical_length_m,
    )

    # If column_res parameters are provided, use the column-based approach
    if column_res_x_m is not None or column_res_y_m is not None or column_res_m is not None:
        if column_res_x_m is not None:
            dx_nd = float(column_res_x_m) / float(len_scale)
        elif column_res_m is not None:
            dx_nd = float(column_res_m) / float(len_scale)
        else:
            raise ValueError("Either column_res_x_m or column_res_m must be provided")

        if column_res_y_m is not None:
            dy_nd = float(column_res_y_m) / float(len_scale)
        elif column_res_m is not None:
            dy_nd = float(column_res_m) / float(len_scale)
        else:
            raise ValueError("Either column_res_y_m or column_res_m must be provided")

        if not (dx_nd > 0.0 and dy_nd > 0.0):
            raise ValueError("column_res_x_m and column_res_y_m (or column_res_m) must be > 0")

        if step_nd is not None and step_nd > 0.0:
            dy_nd = round(dy_nd / step_nd) * step_nd

        zigzag = create_path_columns(x_new_nd, y_new_nd, boundary_path, dx_nd=dx_nd, dy_nd=dy_nd)
    else:
        # Legacy n-based approach
        if n is None:
            n = 100
        zigzag = _create_path_legacy(x_new_nd, y_new_nd, boundary_path, n=n)

    W = np.array(zigzag, dtype=float)
    if anchor_to_first and W.shape[0] >= 1:
        W[:, 0] -= W[0, 0]
        W[:, 1] -= W[0, 1]
    return W


