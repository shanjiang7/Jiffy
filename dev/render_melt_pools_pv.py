#!/usr/bin/env pvbatch
"""
ParaView prototype renders for the "one wall-clock instant, N melt pools" hero.

Scene A (pv_pool_zoom.png): one real comoving-window pool, volume-rendered.
Scene B (pv_layer.png): full bull layer — rank-colored path tube + the 8
rank-local pool isosurfaces at their true positions (scaled up POOL_SCALE x
for build-scale visibility; disclosed in the caption).

Run:  module load gcc/15.1.0 cuda/13.0 openmpi/5.0.9 paraview_osmesa/5.13.3
      pvbatch dev/render_melt_pools_pv.py
"""
import json

from paraview.simple import *  # noqa: F401,F403

OUT = "outputs/proto_melt_pools"
N_RANKS = 8
POOL_SCALE = 5.0
H_MM = 0.030
HALF = 16 * H_MM

# turbo-ish anchor colors for 8 ranks
TURBO8 = [(0.19, 0.07, 0.23), (0.16, 0.42, 0.89), (0.10, 0.80, 0.72),
          (0.47, 0.99, 0.21), (0.83, 0.88, 0.10), (0.98, 0.54, 0.10),
          (0.84, 0.20, 0.02), (0.48, 0.02, 0.01)]

info = json.load(open(f"{OUT}/ranks.json"))
mids = {}
for r, d in info.items():
    mids[int(r)] = d

# Laser positions are recoverable from the vtk ORIGIN, so read them there.
def pool_center(path):
    for line in open(path):
        if line.startswith("ORIGIN"):
            _, x, y, z = line.split()
            return float(x) + HALF, float(y) + HALF, 0.0
    raise RuntimeError(f"no ORIGIN in {path}")


# ---------------- Scene A: single-pool volume render ----------------
view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1600, 1200]
view.UseColorPaletteForBackground = 0
view.Background = [0.04, 0.04, 0.07]
view.OrientationAxesVisibility = 0

pool = LegacyVTKReader(FileNames=[f"{OUT}/rank_03.vtk"])
rep = Show(pool, view)
rep.SetRepresentationType("Volume")
ColorBy(rep, ("POINTS", "excessT"))
lut = GetColorTransferFunction("excessT")
lut.RGBPoints = [0.0, 0.0, 0.0, 0.0,
                 60.0, 0.35, 0.05, 0.28,
                 180.0, 0.9, 0.25, 0.05,
                 350.0, 1.0, 0.65, 0.05,
                 520.0, 1.0, 1.0, 0.85]
opa = GetOpacityTransferFunction("excessT")
opa.Points = [0.0, 0.0, 0.5, 0.0,
              25.0, 0.02, 0.5, 0.0,
              120.0, 0.25, 0.5, 0.0,
              520.0, 0.95, 0.5, 0.0]

cx, cy, cz = pool_center(f"{OUT}/rank_03.vtk")

# The comoving-domain box: outline of the same window (second reader so the
# volume rep and the outline rep can coexist).
box = LegacyVTKReader(FileNames=[f"{OUT}/rank_03.vtk"])
brep = Show(box, view)
brep.SetRepresentationType("Outline")
brep.AmbientColor = [0.75, 0.75, 0.8]
brep.LineWidth = 2.0

view.CameraFocalPoint = [cx, cy - 0.3, -0.25]
view.CameraPosition = [cx + 1.0, cy - 1.25, 0.55]
view.CameraViewUp = [0, 0, 1]
Render()
SaveScreenshot(f"{OUT}/pv_pool_zoom.png", view)
print(f"[ok] {OUT}/pv_pool_zoom.png")
Hide(pool, view)
Hide(box, view)

# ---------------- Scene B: full layer, 8 pools + path ----------------
view.Background = [1.0, 1.0, 1.0]

path = LegacyVTKReader(FileNames=[f"{OUT}/path.vtk"])
tube = Tube(Input=path)
tube.Radius = 0.045
prep = Show(tube, view)
ColorBy(prep, ("POINTS", "rank"))
plut = GetColorTransferFunction("rank", prep, separate=True)
prep.UseSeparateColorMap = 1
rgb = []
for i, c in enumerate(TURBO8):
    rgb += [float(i)] + list(c)
plut.RGBPoints = rgb
prep.Opacity = 0.45
prep.LookupTable = plut

for r in range(N_RANKS):
    f = f"{OUT}/rank_{r:02d}.vtk"
    src = LegacyVTKReader(FileNames=[f])
    con = Contour(Input=src)
    con.ContourBy = ["POINTS", "excessT"]
    con.Isosurfaces = [40.0, 160.0, 350.0]
    cx, cy, cz = pool_center(f)
    tr = Transform(Input=con)
    s = POOL_SCALE
    tr.Transform.Scale = [s, s, s]
    tr.Transform.Translate = [cx * (1 - s), cy * (1 - s), 0.0]
    crep = Show(tr, view)
    ColorBy(crep, None)
    crep.DiffuseColor = list(TURBO8[r])
    crep.Opacity = 0.55
    crep.Specular = 0.4

view.CameraFocalPoint = [25.0, -12.0, 0.0]
view.CameraPosition = [25.0, -34.0, 52.0]
view.CameraViewUp = [0.0, 0.0, 1.0]
Render()
SaveScreenshot(f"{OUT}/pv_layer.png", view)
print(f"[ok] {OUT}/pv_layer.png")
