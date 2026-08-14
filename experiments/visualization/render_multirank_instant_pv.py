#!/usr/bin/env pvbatch
"""
Headless ParaView rendering of the multi-rank instant scene (pvbatch).

Loads <run>/vtk_instant/{domains/rank_*.vtk, paths.vtk}, applies the figure
styling (transparent-ambient volume rendering on the black-body scale, thin
domain outlines, traced/untraced path split), renders PNGs from a set of
camera angles, and saves a .pvsm state that carries every style for local
ParaView editing.

Usage:  pvbatch render_multirank_instant_pv.py <vtk_instant_dir> <out_dir>
"""
import glob
import os
import sys

from paraview.simple import (  # noqa
    LegacyVTKReader, Outline, Contour, Tube, Show, Hide, Render,
    GetActiveViewOrCreate, ColorBy, GetColorTransferFunction,
    GetOpacityTransferFunction, SaveScreenshot, SaveState,
    HideScalarBarIfNotNeeded,
)

MM = 1e-3
T_AMBIENT, T_CLIP = 298.0, 2600.0


def main() -> None:
    src_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    view = GetActiveViewOrCreate("RenderView")
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    view.ViewSize = [2400, 1800]

    # --- temperature transfer functions (shared by all domains) ------------
    ctf = GetColorTransferFunction("full_K")
    ctf.ApplyPreset("Black-Body Radiation", True)
    ctf.RescaleTransferFunction(T_AMBIENT, T_CLIP)
    otf = GetOpacityTransferFunction("full_K")
    otf.Points = [
        T_AMBIENT, 0.0, 0.5, 0.0,
        330.0, 0.03, 0.5, 0.0,
        500.0, 0.10, 0.5, 0.0,
        900.0, 0.25, 0.5, 0.0,
        1300.0, 0.45, 0.5, 0.0,
        1658.0, 0.75, 0.5, 0.0,
        T_CLIP, 0.95, 0.5, 0.0,
    ]

    # --- the 16 moving domains: hot glow + thin outline --------------------
    SHELLS = [  # (T iso, opacity, RGB — explicit black-body-style colors)
        (1000.0, 0.22, (0.45, 0.08, 0.03)),
        (1400.0, 0.50, (0.85, 0.25, 0.05)),
        (1658.0, 0.85, (1.00, 0.55, 0.10)),
        (2300.0, 1.00, (1.00, 0.95, 0.75)),
    ]
    for f in sorted(glob.glob(os.path.join(src_dir, "domains", "rank_*.vtk"))):
        r = LegacyVTKReader(registrationName=os.path.basename(f), FileNames=[f])
        for tval, alpha, rgb in SHELLS:
            c = Contour(registrationName=f"{os.path.basename(f)}_T{int(tval)}", Input=r)
            c.ContourBy = ["POINTS", "full_K"]
            c.Isosurfaces = [tval]
            cs = Show(c, view)
            cs.ColorArrayName = ["POINTS", ""]
            cs.AmbientColor = list(rgb)
            cs.DiffuseColor = list(rgb)
            cs.Ambient = 0.45
            cs.Opacity = alpha
        box = Outline(registrationName=os.path.basename(f) + "_box", Input=r)
        outl = Show(box, view)
        outl.ColorArrayName = ["POINTS", ""]
        outl.AmbientColor = [0.45, 0.45, 0.45]
        outl.DiffuseColor = [0.45, 0.45, 0.45]
        outl.LineWidth = 1.2
        outl.Opacity = 0.8

    # --- paths: traced (tubes) vs untraced (faint hairlines) ---------------
    done = LegacyVTKReader(registrationName="paths_done.vtk",
                           FileNames=[os.path.join(src_dir, "paths_done.vtk")])
    dshow = Show(done, view)
    dshow.ColorArrayName = ["POINTS", ""]
    dshow.AmbientColor = [0.40, 0.40, 0.43]
    dshow.DiffuseColor = [0.40, 0.40, 0.43]
    dshow.LineWidth = 2.0
    dshow.Opacity = 0.9

    todo = LegacyVTKReader(registrationName="paths_todo.vtk",
                           FileNames=[os.path.join(src_dir, "paths_todo.vtk")])
    tshow = Show(todo, view)
    tshow.ColorArrayName = ["POINTS", ""]
    tshow.AmbientColor = [0.72, 0.72, 0.72]
    tshow.DiffuseColor = [0.72, 0.72, 0.72]
    tshow.LineWidth = 1.0
    tshow.Opacity = 0.3

    HideScalarBarIfNotNeeded(ctf, view)

    # --- cameras ------------------------------------------------------------
    focal = [10 * MM, -10 * MM, 7 * MM]
    cams = {
        "iso": ([27 * MM, -29 * MM, 18 * MM], [0, 0, 1]),
        "front": ([10 * MM, -46 * MM, 14 * MM], [0, 0, 1]),
        "top": ([10 * MM, -10.01 * MM, 46 * MM], [0, 1, 0]),
    }
    for name, (pos, up) in cams.items():
        view.CameraFocalPoint = focal
        view.CameraPosition = pos
        view.CameraViewUp = up
        view.CameraViewAngle = 30.0
        Render(view)
        png = os.path.join(out_dir, f"multirank_instant_{name}.png")
        SaveScreenshot(png, view, ImageResolution=[2400, 1800])
        print(f"[ok] {png}")

    state = os.path.join(out_dir, "multirank_instant.pvsm")
    SaveState(state)
    print(f"[ok] {state}")


if __name__ == "__main__":
    main()
