# post/turbulence_viz.py
# Robust turbulence viz + plane quiver with projected arrows.
import os, numpy as np, pyvista as pv

# ----------- USER KNOBS -----------
FLOW_AXIS = "x"        # for default camera on iso image
LOOK_FROM = "+y"       # for the XZ plane view ("+y" or "-y")
RHO, UINF, P_REF = 1.225, 5.0, 0.0
VOL = os.path.join("..","run", "vol_solution.vtu")
SUR = os.path.join("..","run", "surface.vtu")

# Plane sampling & arrows
PAD = 0.15             # padding around wing chord/thickness in plane
GRID_RES = (70, 28)    # (Nx, Nz) arrow grid resolution
ARROW_FACTOR = 0.02    # arrow size ~ fraction of (max plane extent)
# ----------------------------------

try: pv.start_xvfb()
except Exception: pass
pv.global_theme.smooth_shading = True

# -------- helpers --------
def pick_vec(ds):
    for n in ("Velocity","velocity","U","Vel"):
        if n in ds.array_names and ds[n].ndim == 2 and ds[n].shape[1] >= 3:
            return n
    return None

def ensure_speed(ds, vname):
    if vname and vname in ds.array_names and "|U|" not in ds.array_names:
        U = np.asarray(ds[vname])[:, :3]
        ds["|U|"] = np.linalg.norm(U, axis=1)

def extract_wing(surface_like):
    if isinstance(surface_like, pv.MultiBlock):
        # prefer a block named walls
        for k in surface_like.keys():
            if k and "wall" in k.lower():
                return surface_like[k].extract_surface().triangulate()
        surf = surface_like.combine()
    else:
        surf = surface_like
    # strip farfield planes
    Xmin,Xmax,Ymin,Ymax,Zmin,Zmax = surf.bounds
    L = max(Xmax-Xmin, Ymax-Ymin, Zmax-Zmin); tol = 1e-6*L
    centers = surf.cell_centers().points
    on_plane = ((np.abs(centers[:,0]-Xmin)<tol)|(np.abs(centers[:,0]-Xmax)<tol)|
                (np.abs(centers[:,1]-Ymin)<tol)|(np.abs(centers[:,1]-Ymax)<tol)|
                (np.abs(centers[:,2]-Zmin)<tol)|(np.abs(centers[:,2]-Zmax)<tol))
    keep = np.nonzero(~on_plane)[0]
    wing = surf.extract_cells(keep).extract_surface().triangulate()
    return wing if wing.n_cells else surf.extract_surface().triangulate()

def cam_from_bounds(bounds, axis, dist_mult=1.2):
    xmin,xmax,ymin,ymax,zmin,zmax = bounds
    cx,cy,cz = 0.5*(xmin+xmax), 0.5*(ymin+ymax), 0.5*(zmin+zmax)
    dx,dy,dz = xmax-xmin, ymax-ymin, zmax-zmin
    dist = dist_mult*max(dx,dy,dz)
    if axis=="x": return [(cx-dist,cy,cz),(cx,cy,cz),(0,0,1)]
    if axis=="y": return [(cx,cy-dist,cz),(cx,cy,cz),(0,0,1)]
    return [(cx,cy,cz+dist),(cx,cy,cz),(0,1,0)]

def project_to_plane(vecs, normal):
    n = np.asarray(normal, float)
    n = n/np.linalg.norm(n)
    # remove normal component: v_t = v - (v·n)n
    dot = np.einsum('ij,j->i', vecs, n)
    return vecs - np.outer(dot, n)

def pick_turb_field(ds):
    prefs=[("TurbulentViscosity","μ_t"),("Turbulent_Viscosity","μ_t"),
           ("MuT","μ_t"),("mut","μ_t"),
           ("TurbulentKE","k"),("k","k"),
           ("SpecificDissipationRate","ω"),("omega","ω")]
    for n,l in prefs:
        if n in ds.array_names:
            return n,l
    return None,None

def subsample(points, vectors, max_pts):
    n = points.shape[0]
    if n <= max_pts: return points, vectors
    idx = np.linspace(0, n-1, max_pts, dtype=int)
    return points[idx], vectors[idx]

# -------- load data --------
vol = pv.read(VOL)
surf_any = pv.read(SUR)
wing = extract_wing(surf_any)
vname = pick_vec(vol)
if not vname: raise SystemExit("No 3-component velocity in volume VTU.")
ensure_speed(vol, vname)

# Cp on wing (handy reference)
try:
    if "Pressure" not in wing.array_names:
        wing = wing.sample(vol)
    if "Pressure" in wing.array_names:
        wing["Cp"] = (wing["Pressure"] - P_REF)/(0.5*RHO*UINF*UINF)
except Exception:
    pass

# ===================== 1) 3D turbulence iso =====================
turb_name, turb_label = pick_turb_field(vol)
if turb_name is None:
    # derive Q or |ω|
    try:
        grad = vol.compute_derivative(scalars=vname, vorticity=True, qcriterion=True)
        if "Q-criterion" in grad.array_names:
            vol["Q-criterion"] = grad["Q-criterion"]; turb_name="Q-criterion"; turb_label="Q"
        elif "vorticity" in grad.array_names:
            v = np.asarray(grad["vorticity"])[:, :3]
            vol["|omega|"] = np.linalg.norm(v, axis=1); turb_name="|omega|"; turb_label="|ω|"
    except Exception:
        pass

if turb_name:
    vals = np.asarray(vol[turb_name]); vals = vals[np.isfinite(vals)]
    if vals.size:
        iso_val = float(np.quantile(vals, 0.90))
        if iso_val <= 0 and vals.max()>0: iso_val = 0.1*float(vals.max())
        try:
            iso = vol.contour(isosurfaces=[iso_val], scalars=turb_name)
            try: iso = iso.decimate(0.35, volume_preservation=True)
            except Exception: pass
            p = pv.Plotter(off_screen=True, window_size=(1500,950))
            p.add_mesh(iso, scalars=turb_name)
            p.add_mesh(wing, color="white", opacity=0.35, smooth_shading=True)
            p.camera_position = cam_from_bounds(iso.bounds, FLOW_AXIS)
            p.add_axes()
            p.show(screenshot="turb_iso.png"); p.close()
        except Exception as e:
            print("Iso failed:", e)

# Wing Cp (or Pressure)
p = pv.Plotter(off_screen=True, window_size=(1200,800))
if "Cp" in wing.array_names:
    p.add_mesh(wing, scalars="Cp")
elif "Pressure" in wing.array_names:
    p.add_mesh(wing, scalars="Pressure")
else:
    p.add_mesh(wing, color="white")
p.camera_position = cam_from_bounds(wing.bounds, FLOW_AXIS)
p.add_axes(); p.show(screenshot="wing_cp.png"); p.close()

# ===================== 2) XZ plane quiver (projected) =====================
# Build XZ plane at wing mid-span, normal +Y
wxmin,wxmax,wymin,wymax,wzmin,wzmax = wing.bounds
cx, cz, ym = 0.5*(wxmin+wxmax), 0.5*(wzmin+wzmax), 0.5*(wymin+wymax)
sx, sz = max(wxmax-wxmin,1e-9), max(wzmax-wzmin,1e-9)
extent_x, extent_z = sx*(1+2*PAD), sz*(1+2*PAD)
origin = (cx, ym, cz); normal = (0,1,0)

# Background slice colored by |U|
slice_vol = vol.slice(normal=normal, origin=origin)
if vname not in slice_vol.array_names:
    slice_vol = slice_vol.sample(vol)
ensure_speed(slice_vol, vname)

# Regular grid on the plane → sample vectors, then project to plane
grid = pv.Plane(center=origin, direction=normal,
                i_size=extent_x, j_size=extent_z,
                i_resolution=GRID_RES[0], j_resolution=GRID_RES[1])
sampled = grid.sample(vol)
U = np.asarray(sampled[vname])[:, :3]
U_tan = project_to_plane(U, normal)                 # << core fix
# Optional: clamp absurd magnitudes for visibility
mag = np.linalg.norm(U_tan, axis=1)
m95 = np.quantile(mag[mag>0], 0.95) if np.any(mag>0) else 1.0
U_tan = np.where(mag[:,None]>m95, U_tan*(m95/np.maximum(mag,1e-12))[:,None], U_tan)

# Build arrow glyphs (no implicit scaling; orientation from tangential vectors)
pts = pv.PolyData(sampled.points)
pts["U_tan"] = U_tan
pts.set_active_vectors("U_tan")
arrows = pts.glyph(scale=False, factor=ARROW_FACTOR*max(extent_x,extent_z), geom=pv.Arrow())

# Camera: look along ±Y, tight on plane extents
dist = 1.05*max(extent_x, extent_z)
cam = ((cx, ym - dist, cz), (cx, ym, cz), (0,0,1)) if LOOK_FROM=="+y" \
      else ((cx, ym + dist, cz), (cx, ym, cz), (0,0,1))

p = pv.Plotter(off_screen=True, window_size=(1500,950))
p.add_mesh(slice_vol, scalars="|U|" if "|U|" in slice_vol.array_names else None, opacity=0.65)
p.add_mesh(arrows, color="black")
p.add_mesh(wing, color="white", opacity=0.25, smooth_shading=True)
p.camera_position = cam
p.add_axes()
p.show(screenshot="xz_quiver.png"); p.close()

print("Saved: turb_iso.png (if available), xz_quiver.png, wing_cp.png")
