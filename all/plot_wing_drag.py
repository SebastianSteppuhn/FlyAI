#!/usr/bin/env python3
"""
Plot wing-only drag visualization from existing SU2 outputs.

- Reads drag/lift from history.csv
- Reads latest surface_flow*.vtu
- Removes farfield box faces using cell centers
- Plots ONLY the wing colored by a drag-related field

Requirements:
    pip install pyvista pandas
"""

import pathlib

import numpy as np
import pandas as pd
import pyvista as pv


# ----------------------------- USER SETTINGS ----------------------------------

CASE_DIR     = pathlib.Path(".")
HISTORY_FILE = CASE_DIR / "history.csv"
SURFACE_GLOB = "surface_flow*.vtu"
OUTPUT_IMAGE = CASE_DIR / "wing_drag.png"


# ----------------------- DRAG / LIFT FROM history.csv ------------------------


def read_drag_from_history():
    """Get CD and CL from history.csv, robust to spacing/quotes in headers."""
    if not HISTORY_FILE.exists():
        raise FileNotFoundError(f"{HISTORY_FILE} not found")

    df = pd.read_csv(HISTORY_FILE)
    print("history.csv columns:", list(df.columns))

    def norm(name: str) -> str:
        return name.strip().strip('"').strip().upper()

    norm_to_orig = {norm(c): c for c in df.columns}

    # DRAG (CD)
    drag_candidates = ["DRAG", "CD", "CD_TOTAL", "C_D", "CD_SUM", "C_D_SUM"]
    drag_col = None
    for key in drag_candidates:
        if key in norm_to_orig:
            drag_col = norm_to_orig[key]
            break
    if drag_col is None:
        for orig in df.columns:
            n = norm(orig)
            if "DRAG" in n or n.startswith("CD"):
                drag_col = orig
                break
    if drag_col is None:
        raise KeyError("No drag-like column in history.csv")

    # LIFT (CL) – optional
    lift_candidates = ["LIFT", "CL", "CL_TOTAL", "C_L", "CL_SUM", "C_L_SUM"]
    lift_col = None
    for key in lift_candidates:
        if key in norm_to_orig:
            lift_col = norm_to_orig[key]
            break
    if lift_col is None:
        for orig in df.columns:
            n = norm(orig)
            if "LIFT" in n or n.startswith("CL"):
                lift_col = orig
                break

    cd = float(df[drag_col].iloc[-1])
    cl = float(df[lift_col].iloc[-1]) if lift_col is not None else None

    print(f"Using drag column: {drag_col!r} -> CD = {cd:.6f}")
    if cl is not None:
        print(f"Using lift column: {lift_col!r} -> CL = {cl:.6f}")
    print()
    return cd, cl


# ----------------------------- WING GEOMETRY ----------------------------------


def load_surface_polydata():
    """
    Load the latest surface_flow*.vtu and return a PolyData surface.

    SU2 SURFACE_PARAVIEW gives an UnstructuredGrid; we extract the surface
    and triangulate it so we can treat it as a clean PolyData mesh.
    """
    vtus = sorted(CASE_DIR.glob(SURFACE_GLOB))
    if not vtus:
        raise FileNotFoundError(f"No {SURFACE_GLOB} found in {CASE_DIR}")
    surface_file = vtus[-1]
    print(f"Reading surface data from {surface_file}")

    grid = pv.read(surface_file)
    # Ensure we end up with PolyData with cell-based surface
    if isinstance(grid, pv.PolyData):
        surface = grid
    else:
        surface = grid.extract_surface()
    surface = surface.triangulate()

    print("Surface type:", type(surface).__name__,
          "| n_cells:", surface.n_cells, "| n_points:", surface.n_points)
    print("Available fields on surface:", surface.array_names)
    return surface


def strip_farfield_box_cells(mesh: pv.PolyData, margin_ratio: float = 1e-3) -> pv.PolyData:
    """
    Keep only cells whose centers are strictly inside the global bounding box
    by a small margin. "Box" faces have centers exactly on the bbox planes, so
    they get removed. The wing (inside the box) is kept.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    Lx = xmax - xmin
    Ly = ymax - ymin
    Lz = zmax - zmin

    # margins: small fraction of box size to avoid numerical issues
    mx = max(margin_ratio * Lx, 1e-12)
    my = max(margin_ratio * Ly, 1e-12)
    mz = max(margin_ratio * Lz, 1e-12)

    centers = mesh.cell_centers().points  # shape (n_cells, 3)

    inside = (
        (centers[:, 0] > xmin + mx) & (centers[:, 0] < xmax - mx) &
        (centers[:, 1] > ymin + my) & (centers[:, 1] < ymax - my) &
        (centers[:, 2] > zmin + mz) & (centers[:, 2] < zmax - mz)
    )

    keep_ids = np.where(inside)[0]
    print(f"Total surface cells: {mesh.n_cells}, "
          f"keeping {int(keep_ids.size)} internal (wing) cells.")

    if keep_ids.size == 0:
        raise RuntimeError(
            "After removing bbox cells, no cells remain.\n"
            "Either the wing touches the box everywhere, or something is off "
            "with the mesh extents."
        )

    return mesh.extract_cells(keep_ids)


def pick_drag_field(mesh: pv.PolyData) -> str:
    """
    Choose a scalar field suitable for drag visualization.
    Prefer Skin_Friction_Coefficient, then Pressure_Coefficient.
    """
    preferred = ["Skin_Friction_Coefficient", "Pressure_Coefficient"]
    for name in preferred:
        if name in mesh.array_names:
            return name

    # fallback: anything that looks drag-ish
    for name in mesh.array_names:
        u = name.upper()
        if "FRICTION" in u or "COEFF" in u or "PRESSURE" in u:
            return name

    raise KeyError("No drag-related scalar field found on the surface mesh.")


# ------------------------------- PLOTTING -------------------------------------


def plot_wing_only(cd: float, output_image: pathlib.Path):
    full_surface = load_surface_polydata()
    wing = strip_farfield_box_cells(full_surface)
    field = pick_drag_field(wing)

    print(f"Coloring wing by field: {field}")

    values = np.asarray(wing[field])

    # Aggressive color scaling using percentiles
    vmin = float(np.percentile(values, 5))
    vmax = float(np.percentile(values, 95))
    if vmin == vmax:
        vmin = float(values.min())
        vmax = float(values.max())

    print(f"Color scale limits: vmin={vmin:.3e}, vmax={vmax:.3e}")

    plotter = pv.Plotter(off_screen=True)

    # Higher resolution window
    plotter.window_size = (2560, 1440)   # <--- change to (3840, 2160) for “4K” etc.

    plotter.add_mesh(
        wing,
        scalars=field,
        show_edges=False,
        clim=(vmin, vmax),
    )
    plotter.add_axes()
    plotter.enable_anti_aliasing()
    plotter.add_title(f"{field} on wing  (CD = {cd:.4f})")

    xmin, xmax, ymin, ymax, zmin, zmax = wing.bounds
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    dx = max(xmax - xmin, 1e-9)

    plotter.camera_position = [
        (cx - 3.0 * dx, cy, cz + (zmax - zmin)),  # camera
        (cx, cy, cz),                             # look-at
        (0, 0, 1),                                # up
    ]

    plotter.show(screenshot=str(output_image))
    print(f"Saved wing-only drag image to {output_image}")



#-------------------- MAIN ---------------------------------------


def main():
    cd, cl = read_drag_from_history()
    plot_wing_only(cd, OUTPUT_IMAGE)


if __name__ == "__main__":
    main()
