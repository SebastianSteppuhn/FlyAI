# post/plot_arrows.py
# Robust streamlines + arrow glyphs (PyVista/VTK version-safe)
import os, numpy as np, pyvista as pv

# ---- set this to your freestream axis for camera & seeding ----
FLOW_AXIS = "x"   # 'x' | 'y' | 'z'
RHO, UINF, P_REF = 1.225, 5.0, 0.0

VOL = os.path.join("..","run", "vol_solution.vtu")
SUR = os.path.join("..","run", "surface.vtu")

# headless-safe
try: pv.start_xvfb()
except Exception: pass
pv.global_theme.smooth_shading = True

# ------------ load ----------
vol = pv.read(VOL)
surf_any = pv.read(SUR)

# ------------ wing extraction (hide outer box) ----------
def extract_wing(surface_like):
    surface = surface_like.combine() if isinstance(surface_like, pv.MultiBlock) else surface_like
    # prefer a block named like 'walls'
    if isinstance(surface_like, pv.MultiBlock):
        for key in surface_like.keys():
            if key and "wall" in key.lower():
                return surface_like[key].extract_surface().triangulate()
    Xmin,Xmax,Ymin,Ymax,Zmin,Zmax = surface.bounds
    L=max(Xmax-Xmin, Ymax-Ymin, Zmax-Zmin); tol=1e-6*L
    centers = surface.cell_centers().points
    on_plane = (
        (np.abs(centers[:,0]-Xmin)<tol) | (np.abs(centers[:,0]-Xmax)<tol) |
        (np.abs(centers[:,1]-Ymin)<tol) | (np.abs(centers[:,1]-Ymax)<tol) |
        (np.abs(centers[:,2]-Zmin)<tol) | (np.abs(centers[:,2]-Zmax)<tol)
    )
    keep = np.nonzero(~on_plane)[0]
    wing_usg = surface.extract_cells(keep)
    return wing_usg.extract_surface().triangulate()

wing = extract_wing(surf_any)

# ------------ velocity ------------
def pick_velocity_name(d):
    for n in ("Velocity","velocity","U","Vel"):
        if n in d.array_names:
            arr = np.asarray(d[n])
            if arr.ndim==2 and arr.shape[1]>=3:
                return n
    return None

vname = pick_velocity_name(vol)
if vname is None:
    raise SystemExit("No 3-component velocity field in volume VTU. Ensure VOLUME_OUTPUT includes SOLUTION.")

def ensure_speed(ds):
    if vname in ds.array_names and "|U|" not in ds.array_names:
        U = np.asarray(ds[vname])[:, :3]
        ds["|U|"] = np.linalg.norm(U, axis=1)

ensure_speed(vol)
# sample Pressure for Cp if available
if "Pressure" not in wing.array_names:
    try: wing = wing.sample(vol)
    except Exception: pass
if "Pressure" in wing.array_names:
    wing["Cp"] = (wing["Pressure"] - P_REF)/(0.5*RHO*UINF*UINF)

# ------------ seeding geometry (plane upstream) ----------
b = vol.bounds
xmin,xmax,ymin,ymax,zmin,zmax = b
dx,dy,dz = xmax-xmin, ymax-ymin, zmax-zmin
center = ((xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2)

if FLOW_AXIS == "x":
    seed_center = (xmin - 0.05*dx, center[1], center[2]); seed_dir=(1,0,0)
    i_size, j_size = 0.6*dy, 0.6*dz
elif FLOW_AXIS == "y":
    seed_center = (center[0], ymin - 0.05*dy, center[2]); seed_dir=(0,1,0)
    i_size, j_size = 0.6*dx, 0.6*dz
else:
    seed_center = (center[0], center[1], zmin - 0.05*dz); seed_dir=(0,0,1)
    i_size, j_size = 0.6*dx, 0.6*dy

seed = pv.Plane(center=seed_center, direction=seed_dir,
                i_size=i_size, j_size=j_size, i_resolution=25, j_resolution=25)

# ------------ robust streamlines (API-stable) ------------
vol2 = vol.copy()
vol2.set_active_vectors(vname)
# step_unit must be 'cl' (cell length) or 'l' (length); use 'cl' for stability
lines = vol2.streamlines_from_source(
    source=seed,
    integrator_type=4,                # RK4 (safe)
    integration_direction='forward',
    step_unit='cl',                   # << was the error
    initial_step_length=0.5,          # in cell-length units
    min_step_length=0.01,
    max_step_length=1.0,
    max_time=5.0,                     # increase if you want longer lines
    terminal_speed=0.0
)

# Color by speed on the lines
if vname in lines.array_names:
    U = np.asarray(lines[vname])[:, :3]
    lines["|U|"] = np.linalg.norm(U, axis=1)

# ------------ arrow glyphs ------------
# Downsample to avoid too many arrows
on_ratio = max(int(lines.n_points/800), 1)
pts = lines.mask_points(on_ratio=on_ratio)

# Use line tangents if vectors missing on points
vec_for_glyph = vname if vname in pts.array_names else None
if vec_for_glyph is None:
    try:
        lines_with_tan = lines.compute_tangents()
        pts = lines_with_tan.mask_points(on_ratio=on_ratio)
        vec_for_glyph = "FrenetTangent"
    except Exception:
        vec_for_glyph = None

arrow_scale = 0.02 * max(dx,dy,dz)
if vec_for_glyph:
    pts.set_active_vectors(vec_for_glyph)
arrows = pts.glyph(scale=False, factor=arrow_scale, geom=pv.Arrow())

# ------------ camera ------------
dist = 2.2 * max(dx,dy,dz)
if   FLOW_AXIS=="x": cpos=[(center[0]-dist, center[1], center[2]), center, (0,0,1)]
elif FLOW_AXIS=="y": cpos=[(center[0], center[1]-dist, center[2]), center, (0,0,1)]
else:                cpos=[(center[0], center[1], center[2]+dist), center, (0,1,0)]

# ------------ render: 3D streamlines with arrows ------------
p = pv.Plotter(off_screen=True, window_size=(1600,1000))
p.add_mesh(lines, scalars="|U|" if "|U|" in lines.array_names else None, line_width=1.5)
p.add_mesh(arrows, color="black")
p.add_mesh(wing, color="white", opacity=0.35, smooth_shading=True)
p.camera_position = cpos
p.add_axes()
p.show(screenshot="streamlines_arrows.png")
p.close()

# ------------ render: mid-plane slice with arrows ------------
if FLOW_AXIS == "x":
    origin=(0.5*(xmin+xmax), 0, 0); normal=(1,0,0)
elif FLOW_AXIS == "y":
    origin=(0, 0.5*(ymin+ymax), 0); normal=(0,1,0)
else:
    origin=(0, 0, 0.5*(zmin+zmax)); normal=(0,0,1)

slice_vol = vol.slice(normal=normal, origin=origin)
# sample vectors on a thin plane and glyph directly (no streamline API needed)
grid = pv.Plane(center=origin, direction=normal, i_size=i_size, j_size=j_size, i_resolution=40, j_resolution=16)
sampled = grid.sample(vol)
if vname in sampled.array_names:
    U = np.asarray(sampled[vname])[:, :3]
    sampled["|U|"] = np.linalg.norm(U, axis=1)
    sampled.set_active_vectors(vname)
    arr2 = sampled.glyph(scale=False, factor=0.015*max(dx,dy,dz), geom=pv.Arrow())
else:
    arr2 = None

p = pv.Plotter(off_screen=True, window_size=(1600,1000))
p.add_mesh(slice_vol, scalars="|U|" if "|U|" in slice_vol.array_names else None, opacity=0.6)
if arr2 is not None:
    p.add_mesh(arr2, color="black")
p.add_mesh(wing, color="white", opacity=0.25)
p.camera_position = cpos
p.add_axes()
p.show(screenshot="slice_arrows.png")
p.close()

# optional: reference Cp on wing
if "Cp" in wing.array_names:
    q = pv.Plotter(off_screen=True, window_size=(1200,800))
    q.add_mesh(wing, scalars="Cp")
    q.camera_position = cpos
    q.add_axes()
    q.show(screenshot="wing_cp.png")
    q.close()

print("Saved: streamlines_arrows.png, slice_arrows.png", "(+ wing_cp.png)" if "Cp" in wing.array_names else "")
