#!/usr/bin/env python3
"""
Run SU2 on a wing mesh, extract drag (CD, CL) and plot the wing colored
by a drag-related surface field (skin friction or pressure coefficient).

Requirements:
    pip install pyvista pandas

Make sure:
    - run.cfg is in the same folder as this script
    - MESH_FILENAME in run.cfg points to your mesh.su2
"""

import subprocess
import pathlib
from collections import deque

import pandas as pd
import pyvista as pv


# ----------------------------- USER SETTINGS ----------------------------------

CASE_DIR   = pathlib.Path("")          # folder with run.cfg, history.csv, surface_flow*.vtu
SU2_BINARY = "SU2_CFD"                  # or full path to SU2_CFD if needed
CFG_FILE   = CASE_DIR / "run.cfg"

HISTORY_FILE  = CASE_DIR / "history.csv"
SURFACE_GLOB  = "surface_flow*.vtu"
OUTPUT_IMAGE  = CASE_DIR / "wing_drag.png"
SU2_LOG       = CASE_DIR / "su2_out.log"


# ---------------------------- SU2 RUNNER --------------------------------------


def tail(filename, n=80):
    """Return last n lines of a text file."""
    dq = deque(maxlen=n)
    with open(filename, "r") as f:
        for line in f:
            dq.append(line.rstrip("\n"))
    return list(dq)


def run_su2():
    """
    Run SU2_CFD on run.cfg, capture output in su2_out.log.
    If SU2 crashes, print the last lines of the log and raise.
    """
    print(f"Running {SU2_BINARY} {CFG_FILE} ...")
    with open(SU2_LOG, "w") as log:
        result = subprocess.run(
            [SU2_BINARY, str(CFG_FILE)],
            cwd=CASE_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    if result.returncode != 0:
        print("\n[ERROR] SU2_CFD returned non-zero exit status.")
        print(f"Return code: {result.returncode}")
        print(f"Full SU2 output is in: {SU2_LOG}")
        print("\n--- Last 80 lines of SU2 output ---")
        for line in tail(SU2_LOG, n=80):
            print(line)
        print("--- End of SU2 output tail ---\n")
        raise RuntimeError("SU2_CFD failed. See messages above and su2_out.log for details.")

    print("SU2 run finished successfully.\n")


# ----------------------------- DRAG FROM history.csv --------------------------


def read_drag_from_history():
    """
    Read drag and lift from SU2's history.csv, robust to weird spacing/quotes
    in column names, e.g. '       \"CD\"       '.

    Uses the last row of:
        CD  (drag coefficient)
        CL  (lift coefficient)
        CEff (efficiency), if present
    """
    if not HISTORY_FILE.exists():
        raise FileNotFoundError(f"Could not find {HISTORY_FILE}")

    df = pd.read_csv(HISTORY_FILE)
    print("history.csv columns:", list(df.columns))

    def norm(name: str) -> str:
        # strip spaces + quotes, then upper-case
        return name.strip().strip('"').strip().upper()

    norm_to_orig = {}
    for c in df.columns:
        norm_to_orig[norm(c)] = c

    # drag: CD / DRAG variants
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
        raise KeyError(
            "Could not find a drag-like column in history.csv.\n"
            "Columns found: " + ", ".join(df.columns)
        )

    # lift: CL / LIFT variants
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

    # efficiency: CEFF if present
    eff_col = None
    for orig in df.columns:
        n = norm(orig)
        if "CEFF" in n:
            eff_col = orig
            break

    cd = float(df[drag_col].iloc[-1])
    cl = float(df[lift_col].iloc[-1]) if lift_col is not None else None
    ce = float(df[eff_col].iloc[-1]) if eff_col is not None else None

    print(f"Using drag column: {drag_col!r}  -> CD = {cd:.6f}")
    if cl is not None:
        print(f"Using lift column: {lift_col!r}  -> CL = {cl:.6f}")
    if ce is not None:
        print(f"Using efficiency column: {eff_col!r}  -> CEff = {ce:.6f}")
    print()

    return cd, cl, ce


# -------------------------- SURFACE MESH & FIELDS -----------------------------


def load_surface_mesh():
    """
    Read the latest surface_flow*.vtu and choose a drag-related field.
    For your SU2 build, typical fields are:
      - 'Skin_Friction_Coefficient'
      - 'Pressure_Coefficient'
    """
    vtus = sorted(CASE_DIR.glob(SURFACE_GLOB))
    if not vtus:
        raise FileNotFoundError(
            f"No {SURFACE_GLOB} files found. "
            "Check OUTPUT_FILES, SURFACE_FILENAME and MARKER_PLOTTING in run.cfg."
        )

    surface_file = vtus[-1]
    print(f"Reading surface data from {surface_file}")
    mesh = pv.read(surface_file)
    print("Available fields on surface:", mesh.array_names)

    # Prefer skin friction (viscous drag) first, then pressure coefficient
    preferred_order = [
        "Skin_Friction_Coefficient",
        "Pressure_Coefficient",
    ]

    field = None
    for cand in preferred_order:
        if cand in mesh.array_names:
            field = cand
            break

    if field is None:
        # Fallback: anything mentioning FRICTION or COEFF
        for name in mesh.array_names:
            nu = name.upper()
            if "FRICTION" in nu or "COEFF" in nu:
                field = name
                break

    if field is None:
        raise KeyError(
            "Could not find a drag-related scalar in surface data.\n"
            "Open the .vtu in ParaView, check array names, and choose one "
            "like 'Pressure_Coefficient' or 'Skin_Friction_Coefficient'."
        )

    print(f"Coloring wing by field: {field}\n")
    return mesh, field


# ------------------------------ PLOTTING --------------------------------------


def plot_colored_wing(mesh, field, cd, output_image: pathlib.Path):
    """
    Use PyVista to plot the wing colored by 'field' and save a PNG.
    """
    plotter = pv.Plotter(off_screen=True)

    plotter.add_mesh(
        mesh,
        scalars=field,
        show_edges=False,
    )
    plotter.add_axes()
    plotter.enable_anti_aliasing()
    plotter.add_title(f"{field} on wing  (CD = {cd:.4f})")

    # Simple camera: look from upstream, slightly above
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    dx = xmax - xmin

    plotter.camera_position = [
        (cx - 3.0 * dx, cy, cz + (zmax - zmin)),  # camera location
        (cx, cy, cz),                             # look-at point
        (0, 0, 1),                                # up-vector
    ]

    plotter.show(screenshot=str(output_image))
    print(f"Saved colored wing image to {output_image}")


# ------------------------------ MAIN ------------------------------------------


def main():
    # 1) Run SU2
    run_su2()

    # 2) Drag / lift / efficiency from history.csv
    cd, cl, ce = read_drag_from_history()

    # 3) Load surface and choose a drag-related field
    mesh, field = load_surface_mesh()

    # 4) Plot and save image
    plot_colored_wing(mesh, field, cd, OUTPUT_IMAGE)


if __name__ == "__main__":
    main()
