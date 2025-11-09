import numpy as np
import pyvista as pv

# ---------- EDIT THESE ----------
VOL_FILE   = "vol_solution.vtu"
SURF_FILE  = "surface.vtu"
OUTPUT_PNG = "flow_like_reference.png"
VEL_NAME   = "Velocity"             # from your printout
SURF_SCALAR = "Pressure_Coefficient"  # for coloring the body
# -------------------------------

# load volume and surface
vol  = pv.read(VOL_FILE)
surf = pv.read(SURF_FILE)

# --- split surface into farfield and body by geometry ---
sxmin, sxmax, symin, symax, szmin, szmax = surf.bounds
Lbox = max(sxmax - sxmin, symax - symin, szmax - szmin, 1e-9)
tol  = 1e-3 * Lbox

centers = surf.cell_centers().points
x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]

on_xmin = np.isclose(x, sxmin, atol=tol)
on_xmax = np.isclose(x, sxmax, atol=tol)
on_ymin = np.isclose(y, symin, atol=tol)
on_ymax = np.isclose(y, symax, atol=tol)
on_zmin = np.isclose(z, szmin, atol=tol)
on_zmax = np.isclose(z, szmax, atol=tol)

mask_far  = on_xmin | on_xmax | on_ymin | on_ymax | on_zmin | on_zmax
mask_body = ~mask_far

far_cells  = np.where(mask_far)[0]
body_cells = np.where(mask_body)[0]

far  = surf.extract_cells(far_cells)   if far_cells.size  > 0 else None
body = surf.extract_cells(body_cells)  if body_cells.size > 0 else surf

print(f"Surface cells: total={surf.n_cells}, body={body.n_cells}, farfield={far.n_cells if far else 0}")

# --- set the velocity as active vectors ---
vel = vol.point_data[VEL_NAME]
vol.set_active_vectors(VEL_NAME)

# --- seed a PLANE of streamlines upstream (flow -x -> +x) ---
xmin, xmax, ymin, ymax, zmin, zmax = vol.bounds
Lx   = xmax - xmin
xseed = xmin + 0.05 * Lx           # a bit inside inlet plane

ny, nz = 25, 15                    # grid of seed points
ys = np.linspace(ymin, ymax, ny)
zs = np.linspace(zmin, zmax, nz)
Y, Z = np.meshgrid(ys, zs, indexing="ij")
X = np.full_like(Y, xseed)

points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
seeds = pv.PolyData(points)

streams = vol.streamlines_from_source(
    seeds,
    vectors=VEL_NAME,
    integration_direction="forward",  # with the flow
    max_time=3.0 * Lx,                # let them travel across the box
    initial_step_length=0.01 * Lx,
)

print("Streamlines: points =", streams.n_points, "cells =", streams.n_cells)

# --- plot: body + streamlines, no farfield ---
p = pv.Plotter()

# aircraft body colored by Cp (or fallback white)
if SURF_SCALAR in body.point_data:
    p.add_mesh(body, scalars=SURF_SCALAR, opacity=1.0, smooth_shading=True)
else:
    p.add_mesh(body, color="white", opacity=1.0, smooth_shading=True)

# streamlines in grey
if streams.n_points > 0 and streams.n_cells > 0:
    p.add_mesh(streams, color="gray", line_width=1.5)
else:
    print("WARNING: streamlines are empty; check velocity field or integration settings.")

# camera: look downstream (from -x towards +x)
xmid = 0.5 * (xmin + xmax)
ymid = 0.5 * (ymin + ymax)
zmid = 0.5 * (zmin + zmax)

camera_pos = [
    (xmin - 1.0 * Lx, ymid, zmid),  # eye (upstream)
    (xmid, ymid, zmid),             # look-at
    (0.0, 0.0, 1.0),                # up direction (z)
]
p.camera_position = camera_pos

p.add_axes()
p.show_grid()
p.show(screenshot=OUTPUT_PNG)
