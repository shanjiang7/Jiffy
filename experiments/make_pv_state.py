#!/usr/bin/env pvbatch
"""
Assemble the ParaView scene for the 15-layer x 64-rank bundle and save it as
scene.pvsm (openable locally in ParaView 5.13) plus a preview screenshot.

Run:  module load gcc/15.1.0 cuda/13.0 openmpi/5.0.9 paraview_osmesa/5.13.3
      pvbatch experiments/make_pv_state.py
"""
import glob

from paraview.simple import *  # noqa: F401,F403

OUT = "outputs/pv_bundle_ml15_bull"

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1800, 1400]
view.UseColorPaletteForBackground = 0
view.Background = [1.0, 1.0, 1.0]
view.OrientationAxesVisibility = 0

# ---- per-layer paths, colored by owning rank (shared turbo LUT) ----
rank_lut = GetColorTransferFunction("rank")
rank_lut.ApplyPreset("Turbo", True)
rank_lut.RescaleTransferFunction(0.0, 63.0)

for f in sorted(glob.glob(f"{OUT}/path_layer_*.vtk")):
    src = LegacyVTKReader(FileNames=[f], registrationName=f.split("/")[-1])
    rep = Show(src, view)
    ColorBy(rep, ("POINTS", "rank"))
    rep.LookupTable = rank_lut
    rep.LineWidth = 1.6
    rep.Opacity = 0.55

# ---- per-rank pool windows, shared black-body volume rendering ----
# Physical Kelvin: ambient 298, solidus 1658, liquidus 1723; clamp at 2500
# (source-cell conduction overshoot above that is unphysical detail).
# The standard "Black-Body Radiation" preset (black -> red -> yellow ->
# white), scaled 298..3000 K: ambient stays black, mid-range glows red,
# the melt zone (1658/1723 K) sits in the red-to-yellow transition.
t_lut = GetColorTransferFunction("T_K")
t_lut.ApplyPreset("Black-Body Radiation", True)
t_lut.RescaleTransferFunction(298.0, 2000.0)
t_lut.MapControlPointsToLogSpace()
t_lut.UseLogScale = 1
# Ambient gets nonzero opacity so each window renders as a solid dark block
# of temperature field (black-body: cold = black) on the white page, with
# the melt pool glowing inside. Values above 3000 K clip to white.
t_opa = GetOpacityTransferFunction("T_K")
t_opa.Points = [298.0, 0.22, 0.5, 0.0,
                800.0, 0.3, 0.5, 0.0,
                1658.0, 0.7, 0.5, 0.0,
                3000.0, 0.95, 0.5, 0.0]

import sys
MODE = "contour" if "--volumes" not in sys.argv else "volume"

for f in sorted(glob.glob(f"{OUT}/pool_rank_*.vtk")):
    src = LegacyVTKReader(FileNames=[f], registrationName=f.split("/")[-1])
    if MODE == "volume":
        rep = Show(src, view)
        rep.SetRepresentationType("Volume")
        ColorBy(rep, ("POINTS", "T_K"))
        rep.LookupTable = t_lut
        # 1 mm boxes seen from ~70 mm away: shrink the opacity unit distance
        # so the small pools still accumulate visible opacity.
        rep.ScalarOpacityUnitDistance = 0.06
    else:
        # Laptop-friendly: nested translucent isosurfaces instead of 64
        # volume-ray-cast actors (which re-render every frame and freeze
        # small GPUs).
        con = Contour(Input=src, registrationName=f.split("/")[-1] + ":iso")
        con.ContourBy = ["POINTS", "T_K"]
        # Warm halo / solidus (melt front) / liquidus — physical levels.
        con.Isosurfaces = [800.0, 1658.0, 1723.0]
        rep = Show(con, view)
        ColorBy(rep, ("POINTS", "T_K"))
        rep.LookupTable = t_lut
        rep.Opacity = 0.5
        rep.Specular = 0.3
        # Wireframe box of the sampled window so all 64 domains are
        # locatable at overview distance.
        orep = Show(src, view)
        orep.SetRepresentationType("Outline")
        orep.AmbientColor = [0.35, 0.35, 0.4]
        orep.LineWidth = 1.2

# Lock LUT ranges AFTER all Show() calls: ParaView auto-rescales shared LUTs
# to the last-shown dataset's range otherwise (which clamps every earlier
# layer's colors).
rank_lut.RescaleTransferFunction(0.0, 63.0)
rank_lut.AutomaticRescaleRangeMode = "Never"
t_lut.RescaleTransferFunction(298.0, 2000.0)
t_lut.MapControlPointsToLogSpace()
t_lut.UseLogScale = 1
t_lut.AutomaticRescaleRangeMode = "Never"

# ---- camera: oblique view of the exploded stack ----
view.CameraFocalPoint = [25.0, -12.0, 21.0]
view.CameraPosition = [25.0, -75.0, 55.0]
view.CameraViewUp = [0.0, 0.0, 1.0]
Render()

STATE = "scene.pvsm" if MODE == "contour" else "scene_volumes.pvsm"
SaveScreenshot(f"{OUT}/preview.png", view)
servermanager.SaveState(f"{OUT}/{STATE}")

# Zoomed verification shot on one mid-stack pool (camera state in the .pvsm
# stays the wide view above).
import json
meta = json.load(open(f"{OUT}/pools_meta.json"))
m = [d for d in meta if d["rank"] == 30][0]
px, py, pz = m["x_mm"], m["y_mm"], m["layer"] * 3.0
view.CameraFocalPoint = [px, py - 0.3, pz - 0.2]
view.CameraPosition = [px + 2.2, py - 2.6, pz + 1.4]
view.CameraViewUp = [0, 0, 1]
Render()
SaveScreenshot(f"{OUT}/preview_zoom.png", view)
print(f"[ok] {OUT}/scene.pvsm + preview.png + preview_zoom.png")
