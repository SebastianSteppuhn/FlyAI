# post/plot_turbulence_3d.py
import os
import numpy as np
import pyvista as pv

# ---- user knobs (match your solver setup) ----
FLOW_AXIS = "x"   # 'x'|'y'|'z' – for default camera orientation
RHO       = 1.225 # kg/m^3
UINF      = 5.0   # m/s
P_REF     = 0.0   # Pa (gauge at outlet)
VOL_PATH  = os.path.join("../..", "run", "vol_solution.vtu")
SUR_PATH  = os.path.join("../..", "run", "surface.vtu")
# ----------------------------------------------

# headless-safe
try:
    pv.start_xvfb()
except Exception:
    pass
pv.global_theme.smooth_shading = True

# ------------ load data ------------
vol = pv.read(VOL_PATH)
surf_any = pv.read(SUR_PATH)

def extract_wing(surface_like: pv.DataSet) -> pv.PolyData:
    """Return wing surface only: prefer 'walls' block; else strip box planes."""
    if isinstance(surface_like, pv.MultiBlock):
        # prefer block named like 'walls'
        for key in surface_like.keys():
            if key and "wall" in key.lower():
                return surface_like[key].extract_surface().triangulate()
        surface = surface_like.combine()
    else:
        surface = surface_like

    # remove farfield planes (faces on domain box)
    Xmin, Xmax, Ymin, Ymax, Zmin, Zmax = surface.bounds
    L  = max(Xmax-Xmin, Ymax-Ymin, Zmax-Zmin)
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
    keep = np.nonzero(~on_plane)[0]
    wing_usg = surface.extract_cells(keep)
    wing_pd  = wing_usg.extract_surface().triangulate()
    return wing_pd if wing_pd.n_cells else surface.extract_surface().triangulate()

wing = extract_wing(surf_any)

# ------------ pick turbulence scalar ------------
def pick_turbulence_name(dset: pv.DataSet):
    prefs = [
        ("TurbulentViscosity", "μ_t"),
        ("Turbulent_Viscosity","μ_t"),
        ("MuT","μ_t"), ("mut","μ_t"),
        ("TurbulentKE","k"), ("k","k"),
        ("SpecificDissipationRate","ω"), ("omega","ω"),
    ]
    for name, label in prefs:
        if name in dset.array_names:
            return name, label
    return None, None

turb_name, turb_label = pick_turbulence_name(vol)

# Fallback: derive Q-criterion from Velocity if no turbulence field found
if turb_name is None and "Velocity" in vol.array_names:
    try:
        # compute_derivative is robust across pyvista versions for vorticity/Q
        grad = vol.compute_derivative(vorticity=True, qcriterion=True)
        if "Q-criterion" in grad.array_names:
            vol["Q-criterion"] = grad["Q-criterion"]
            turb_name, turb_label = "Q-criterion", "Q"
        elif "vorticity" in grad.array_names:
            v = grad["vorticity"]
            vol["|ω|"] = np.linalg.norm(v[:, :3], axis=1)
            turb_name, turb_label = "|ω|", "|ω|"
    except Exception:
        pass

if turb_name is None:
    raise SystemExit("No turbulence field found (μ_t/k/ω) and could not derive Q. "
                     "Make sure VOLUME_OUTPUT includes TURBULENCE or rerun case_turbulent.cfg.")

# ------------ helper to compute Cp on wing (nice overlay ref) ------------
if "Pressure" not in wing.array_names:
    try:
        wing = wing.sample(vol)
    except Exception:
        pass
if "Pressure" in wing.array_names:
    q = 0.5 * RHO * UINF * UINF
    wing["Cp"] = (wing["Pressure"] - P_REF) / q

# ------------ iso-surface(s) ------------
vals = np.asarray(vol[turb_name])
vals = vals[np.isfinite(vals)]
if vals.size == 0:
    raise SystemExit(f"Turbulence field '{turb_name}' has no finite values.")

# Choose robust iso levels (tune if needed)
iso_hi = float(np.quantile(vals, 0.90))
iso_lo = float(np.quantile(vals, 0.75)) if np.quantile(vals, 0.75) < iso_hi else 0.5 * iso_hi
# Prevent degenerate zero iso
if iso_hi <= 0.0 and vals.max() > 0.0:
    iso_hi = 0.1 * float(vals.max())

# Build iso(s)
iso1 = vol.contour(isosurfaces=[iso_hi], scalars=turb_name)
iso2 = vol.contour(isosurfaces=[iso_lo], scalars=turb_name) if iso_lo > 0 else None

# Optional light decimation for speed (comment out if you want full detail)
try:
    iso1 = iso1.decimate(0.3, volume_preservation=True)
    if iso2 is not None:
        iso2 = iso2.decimate(0.3, volume_preservation=True)
except Exception:
    pass

# ------------ camera default based on flow axis ------------
bounds = vol.bounds
center = ((bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2)
dx = bounds[1]-bounds[0]; dy = bounds[3]-bounds[2]; dz = bounds[5]-bounds[4]
dist = 2.2 * max(dx, dy, dz)

if FLOW_AXIS == "x":
    cpos = [(center[0]-dist, center[1], center[2]), center, (0,0,1)]
elif FLOW_AXIS == "y":
    cpos = [(center[0], center[1]-dist, center[2]), center, (0,0,1)]
else:
    cpos = [(center[0], center[1], center[2]+dist), center, (0,1,0)]

# ------------ render single-iso image ------------
p = pv.Plotter(off_screen=True, window_size=(1600, 1000))
p.add_mesh(iso1, scalars=turb_name, name="turb_iso_hi")
p.add_mesh(wing, color="white", smooth_shading=True, opacity=0.35)
if "Cp" in wing.array_names:
    # add faint Cp for context (optional)
    pass
p.camera_position = cpos
p.add_axes()
p.show(screenshot="turbulence_3d.png")
p.close()

# ------------ render multi-iso image (hi + lo) ------------
p = pv.Plotter(off_screen=True, window_size=(1600, 1000))
# low iso faint
if iso2 is not None:
    p.add_mesh(iso2, scalars=turb_name, opacity=0.35)
# high iso solid
p.add_mesh(iso1, scalars=turb_name)
p.add_mesh(wing, color="white", smooth_shading=True, opacity=0.35)
p.camera_position = cpos
p.add_axes()
p.show(screenshot="turbulence_3d_multi.png")
p.close()

# Reference: Cp on wing
if "Cp" in wing.array_names:
    p = pv.Plotter(off_screen=True, window_size=(1200, 800))
    p.add_mesh(wing, scalars="Cp")
    p.camera_position = cpos
    p.add_axes()
    p.show(screenshot="wing_cp.png")
    p.close()

print(f"Saved: turbulence_3d.png, turbulence_3d_multi.png, {'wing_cp.png' if 'Cp' in wing.array_names else ''}")
