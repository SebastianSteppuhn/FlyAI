import pyvista as pv
import numpy as np

# ------------ EDIT THESE 3 LINES ------------
VOL_FILE   = "vol_solution.vtu"
SURF_FILE  = "surface.vtu"
VEL_NAME   = "Velocity"   # set to your velocity array name
OUTPUT_PNG = "flow_streamlines.png"
# --------------------------------------------

# load data
vol  = pv.read(VOL_FILE)
surf = pv.read(SURF_FILE)

# clip surface to the central region to get rid of most of the farfield box
sxmin, sxmax, symin, symax, szmin, szmax = surf.bounds
Lx, Ly, Lz = sxmax - sxmin, symax - symin, szmax - szmin
surf_core = surf.clip_box(
    bounds=(
        sxmin + 0.15 * Lx, sxmax - 0.15 * Lx,
        symin + 0.15 * Ly, symax - 0.15 * Ly,
        szmin + 0.15 * Lz, szmax - 0.15 * Lz,
    ),
    invert=False,
)
if surf_core.n_points == 0:  # if clipping removed everything, fall back
    surf_core = surf

# set velocity as active vectors
vel = vol[VEL_NAME]
vol.set_active_vectors(VEL_NAME)

# seeds near inlet so flow is roughly from -x to +x
xmin, xmax, ymin, ymax, zmin, zmax = vol.bounds
Lx = xmax - xmin
xseed = xmin + 0.02 * Lx
zmid  = 0.5 * (zmin + zmax)

streams = vol.streamlines(
    vectors=VEL_NAME,
    integration_direction="forward",
    pointa=(xseed, ymin, zmid),
    pointb=(xseed, ymax, zmid),
    n_points=150,
)

# plot
p = pv.Plotter()
p.add_mesh(surf_core, color="white", opacity=1.0, smooth_shading=True)

if streams.n_points > 0:
    p.add_mesh(streams, color="gray", line_width=2.0)
else:
    print("WARNING: no streamlines generated – seeds may be in a solid or zero-velocity region")

p.add_axes()
p.show_grid()
p.camera_position = "iso"
p.show(screenshot=OUTPUT_PNG)
