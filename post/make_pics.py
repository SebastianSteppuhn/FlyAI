# post/make_pics.py
import os, numpy as np
import pyvista as pv

# headless-friendly
try:
    pv.start_xvfb()
except Exception:
    pass
pv.global_theme.smooth_shading = True

VOL = os.path.join("..", "run", "vol_solution.vtu")
SUR = os.path.join("..", "run", "surface.vtu")

vol = pv.read(VOL)
surf_any = pv.read(SUR)

def ensure_speed(d):
    if "Velocity" in d.array_names and "|U|" not in d.array_names:
        U = d["Velocity"]
        if hasattr(U, "shape") and U.ndim == 2:
            d["|U|"] = np.linalg.norm(U[:, :3], axis=1)

def save_pic(mesh, scalars, fname, overlay=None):
    p = pv.Plotter(off_screen=True)
    if scalars in mesh.array_names:
        p.add_mesh(mesh, scalars=scalars)
    else:
        p.add_mesh(mesh, color=True)
    if overlay is not None:
        p.add_mesh(overlay, color=True, opacity=0.25)
    p.add_axes()
    p.show(screenshot=fname)
    p.close()

def extract_wing(surface_like):
    """Return only internal wing faces (remove outer box planes)."""
    if isinstance(surface_like, pv.MultiBlock):
        surface = surface_like.combine()
    else:
        surface = surface_like

    Xmin, Xmax, Ymin, Ymax, Zmin, Zmax = surface.bounds
    L = max(Xmax - Xmin, Ymax - Ymin, Zmax - Zmin)
    tol = 1e-6 * L

    centers = surface.cell_centers().points
    on_plane = (
        (np.abs(centers[:, 0] - Xmin) < tol) |
        (np.abs(centers[:, 0] - Xmax) < tol) |
        (np.abs(centers[:, 1] - Ymin) < tol) |
        (np.abs(centers[:, 1] - Ymax) < tol) |
        (np.abs(centers[:, 2] - Zmin) < tol) |
        (np.abs(centers[:, 2] - Zmax) < tol)
    )
    keep_ids = np.nonzero(~on_plane)[0]
    wing_usg = surface.extract_cells(keep_ids)

    # Convert to a real surface PolyData for normals/offsets
    wing_poly = wing_usg.extract_surface().triangulate()
    return wing_poly if wing_poly.n_cells else surface.extract_surface().triangulate()

wing = extract_wing(surf_any)

# Make sure volume has |U| etc.
ensure_speed(vol)

# --- 1) Pressure & Cp on the wing ---
# sample pressure onto wing if needed
if "Pressure" not in wing.array_names:
    wing = wing.sample(vol)

rho = 1.225   # match your cfg
Uinf = 5.0    # match your MARKER_INLET speed
p_ref = 0.0   # outlet gauge pressure

if "Pressure" in wing.array_names and "Cp" not in wing.array_names:
    q = 0.5 * rho * Uinf * Uinf
    wing["Cp"] = (wing["Pressure"] - p_ref) / q

save_pic(wing, "Pressure", "wing_pressure.png")
if "Cp" in wing.array_names:
    save_pic(wing, "Cp", "wing_cp.png")

# --- 2) Near-wall speed (offset the wing along normals, then sample volume) ---
ensure_speed(vol)
Lbox = max(vol.bounds[1]-vol.bounds[0],
           vol.bounds[3]-vol.bounds[2],
           vol.bounds[5]-vol.bounds[4])
eps = 0.01 * Lbox  # 1% of box size

# Force POINT normals only (no cell normals, no vertex splitting)
wing_n = wing.compute_normals(point_normals=True,
                              cell_normals=False,
                              auto_orient_normals=True,
                              consistent_normals=True,
                              split_vertices=False,
                              inplace=False)

# Get point normals robustly
normals = None
for key in ("Normals", "PointNormals", "vtkNormals"):
    if key in wing_n.point_data:
        normals = wing_n.point_data[key]
        break

if normals is not None and normals.shape[0] == wing_n.points.shape[0]:
    # Happy path: offset every vertex along its point normal
    pts_off = wing_n.points + eps * normals
    wing_off = pv.PolyData(pts_off, wing_n.faces)
else:
    # Fallback: offset cell centers along cell normals
    centers = wing_n.cell_centers().points
    cnormals = wing_n.cell_normals  # always matches number of cells
    pts_off = centers + eps * cnormals
    wing_off = pv.PolyData(pts_off)

# Sample the volume on the offset shell and plot |U|
sampled = wing_off.sample(vol)
ensure_speed(sampled)
if "|U|" in sampled.array_names:
    save_pic(sampled, "|U|", "wing_nearwall_speed.png")

# --- 3) Mid-plane slice (assumes +X freestream; change normal if not) ---
xmid = 0.5 * (vol.bounds[0] + vol.bounds[1])
slice_x = vol.slice(normal=(1, 0, 0), origin=(xmid, 0, 0))
fld = "Pressure" if "Pressure" in slice_x.array_names else ("|U|" if "|U|" in slice_x.array_names else None)
save_pic(slice_x, fld, "slice.png", overlay=wing)

# --- 4) Streamlines (if vectors available) ---
if "Velocity" in vol.array_names:
    xmin, xmax, ymin, ymax, zmin, zmax = vol.bounds
    seed = pv.Cube(center=(xmin - 0.05*(xmax-xmin), 0, 0),
                   x_length=0.01*(xmax-xmin),
                   y_length=0.6*(ymax-ymin),
                   z_length=0.6*(zmax-zmin))
    try:
        lines = vol.streamlines(vectors="Velocity", source=seed, max_time=2.0, initial_step_length=0.01)
        p = pv.Plotter(off_screen=True)
        p.add_mesh(lines)
        p.add_mesh(wing, color=True, opacity=0.3)
        p.add_axes()
        p.show(screenshot="streamlines.png")
        p.close()
    except Exception as e:
        print("Streamlines failed:", e)

print("Saved: wing_pressure.png, wing_cp.png, wing_nearwall_speed.png, slice.png, streamlines.png")
