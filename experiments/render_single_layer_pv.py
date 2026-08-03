#!/usr/bin/env pvbatch
"""
High-quality single-layer render: one layer of the 15-layer bull with its
rank chunks as distinct colors and each rank's comoving-domain field block
(black-body volume) at its true mid-chunk position. White background.

Run:  module load gcc/15.1.0 cuda/13.0 openmpi/5.0.9 paraview_osmesa/5.13.3
      pvbatch experiments/render_single_layer_pv.py [layer]
"""
import json
import sys

from paraview.simple import *  # noqa: F401,F403

OUT = "outputs/pv_bundle_ml15_bull"
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 7
DZ = 3.0

meta = json.load(open(f"{OUT}/pools_meta.json"))
pools = [m for m in meta if m["layer"] == LAYER]
r_lo = min(m["rank"] for m in pools)
r_hi = max(m["rank"] for m in pools)
print(f"layer {LAYER}: ranks {r_lo}..{r_hi} ({len(pools)} pools)")

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [3200, 2200]
view.UseColorPaletteForBackground = 0
view.Background = [1.0, 1.0, 1.0]
view.OrientationAxesVisibility = 0

# Path as a tube, rank LUT rescaled to THIS layer's chunk range so the
# cuts read as distinct color bands.
rank_lut = GetColorTransferFunction("rank")
rank_lut.ApplyPreset("Turbo", True)

path = LegacyVTKReader(FileNames=[f"{OUT}/path_layer_{LAYER:02d}.vtk"])
tube = Tube(Input=path)
tube.Radius = 0.06
prep = Show(tube, view)
ColorBy(prep, ("POINTS", "rank"))
prep.LookupTable = rank_lut
prep.Opacity = 0.9

t_lut = GetColorTransferFunction("T_K")
t_lut.ApplyPreset("Black-Body Radiation", True)
t_opa = GetOpacityTransferFunction("T_K")
t_opa.Points = [298.0, 0.22, 0.5, 0.0,
                800.0, 0.3, 0.5, 0.0,
                1658.0, 0.7, 0.5, 0.0,
                3000.0, 0.95, 0.5, 0.0]

for m in pools:
    src = LegacyVTKReader(FileNames=[f"{OUT}/pool_rank_{m['rank']:02d}.vtk"])
    rep = Show(src, view)
    rep.SetRepresentationType("Volume")
    ColorBy(rep, ("POINTS", "T_K"))
    rep.LookupTable = t_lut

# Pad the range so no chunk lands on turbo's dark endpoints.
rank_lut.RescaleTransferFunction(float(r_lo) - 1.2, float(r_hi) + 1.2)
rank_lut.AutomaticRescaleRangeMode = "Never"
t_lut.RescaleTransferFunction(298.0, 2000.0)
t_lut.MapControlPointsToLogSpace()
t_lut.UseLogScale = 1
t_lut.AutomaticRescaleRangeMode = "Never"

z = LAYER * DZ
view.CameraFocalPoint = [25.0, -11.0, z]
view.CameraPosition = [25.0, -34.0, z + 14.0]
view.CameraViewUp = [0.0, 0.0, 1.0]
Render()
SaveScreenshot(f"{OUT}/single_layer_{LAYER:02d}.png", view)
print(f"[ok] {OUT}/single_layer_{LAYER:02d}.png")
