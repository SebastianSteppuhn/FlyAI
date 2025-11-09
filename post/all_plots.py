# post/all_plots_tight.py
# Same outputs as all_plots.py, but with tight auto-camera around wing/feature bounds.
import os, csv, numpy as np, pyvista as pv

# -------------------- USER KNOBS --------------------
FLOW_AXIS = "x"     # 'x'|'y'|'z' (freestream)
RHO, UINF, P_REF = 1.225, 5.0, 0.0
VOL_PATH = os.path.join("..","run","vol_solution.vtu")
SUR_PATH = os.path.join("..","run", "surface.vtu")
HIS_PATH = os.path.join("..","run","history.csv")
# Camera padding around target bounds (fraction of target size)
PAD = 0.10
# How far the camera sits from the target (smaller = closer)
DIST_MULT = 1.1
# ---------------------------------------------------

try: pv.start_xvfb()
except Exception: pass
pv.global_theme.smooth_shading = True

# ---------- helpers ----------
def pick_velocity_name(d):
    for n in ("Velocity","velocity","U","Vel"):
        if n in d.array_names:
            arr = np.asarray(d[n])
            if arr.ndim==2 and arr.shape[1]>=3:
                return n
    return None

def ensure_speed(d, vname):
    if vname and vname in d.array_names and "|U|" not in d.array_names:
        U = np.asarray(d[vname])[:, :3]
        d["|U|"] = np.linalg.norm(U, axis=1)

def extract_wing(surface_like):
    if isinstance(surface_like, pv.MultiBlock):
        for key in surface_like.keys():
            if key and "wall" in key.lower():
                return surface_like[key].extract_surface().triangulate()
        surf = surface_like.combine()
    else:
        surf = surface_like
    Xmin,Xmax,Ymin,Ymax,Zmin,Zmax = surf.bounds
    L = max(Xmax-Xmin, Ymax-Ymin, Zmax-Zmin)
    tol = 1e-6*L
    centers = surf.cell_centers().points
    on_plane = (
        (np.abs(centers[:,0]-Xmin)<tol)|(np.abs(centers[:,0]-Xmax)<tol)|
        (np.abs(centers[:,1]-Ymin)<tol)|(np.abs(centers[:,1]-Ymax)<tol)|
        (np.abs(centers[:,2]-Zmin)<tol)|(np.abs(centers[:,2]-Zmax)<tol)
    )
    wing_usg = surf.extract_cells(np.nonzero(~on_plane)[0])
    wing_pd = wing_usg.extract_surface().triangulate()
    return wing_pd if wing_pd.n_cells else surf.extract_surface().triangulate()

def union_bounds(*bbs):
    xs = [b[0] for b in bbs]; Xs=[b[1] for b in bbs]
    ys = [b[2] for b in bbs]; Ys=[b[3] for b in bbs]
    zs = [b[4] for b in bbs]; Zs=[b[5] for b in bbs]
    return (min(xs), max(Xs), min(ys), max(Ys), min(zs), max(Zs))

def fit_camera(bounds, axis, pad=PAD, dist_mult=DIST_MULT):
    xmin,xmax,ymin,ymax,zmin,zmax = bounds
    dx,dy,dz = max(xmax-xmin,1e-12), max(ymax-ymin,1e-12), max(zmax-zmin,1e-12)
    cx,cy,cz = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2
    # pad the target box
    dxp, dyp, dzp = dx*(1+pad), dy*(1+pad), dz*(1+pad)
    # camera distance scaled to padded target size
    dist = dist_mult * max(dxp, dyp, dzp)
    if axis=="x": pos=(cx - dist, cy, cz); up=(0,0,1)
    elif axis=="y": pos=(cx, cy - dist, cz); up=(0,0,1)
    else: pos=(cx, cy, cz + dist); up=(0,1,0)
    return [pos, (cx,cy,cz), up]

def save_pic(mesh, scalars, fname, overlay=None, cam_bounds=None, opacity=None):
    p = pv.Plotter(off_screen=True, window_size=(1600,1000))
    opts={}
    if opacity is not None: opts["opacity"]=opacity
    if scalars and scalars in mesh.array_names:
        p.add_mesh(mesh, scalars=scalars, **opts)
    else:
        p.add_mesh(mesh, color=True, **opts)
    if overlay is not None:
        p.add_mesh(overlay, color="white", opacity=0.35, smooth_shading=True)
    if cam_bounds is None: cam_bounds = mesh.bounds
    p.camera_position = fit_camera(cam_bounds, FLOW_AXIS)
    p.add_axes()
    p.show(screenshot=fname)
    p.close()

def pick_turb_field(d):
    prefs=[("TurbulentViscosity","μ_t"),("Turbulent_Viscosity","μ_t"),
           ("MuT","μ_t"),("mut","μ_t"),
           ("TurbulentKE","k"),("k","k"),
           ("SpecificDissipationRate","ω"),("omega","ω")]
    for n,l in prefs:
        if n in d.array_names: return n,l
    return None,None

def subsample_points(points: np.ndarray, vectors: np.ndarray, max_pts: int):
    n = points.shape[0]
    if n <= max_pts: return points, vectors
    idx = np.linspace(0, n-1, max_pts, dtype=int)
    return points[idx], vectors[idx]

# ---------- load ----------
vol = pv.read(VOL_PATH)
surf_any = pv.read(SUR_PATH)
wing = extract_wing(surf_any)

vname = pick_velocity_name(vol)
if not vname:
    raise SystemExit("No 3-component velocity field in volume VTU. Ensure VOLUME_OUTPUT includes SOLUTION.")
ensure_speed(vol, vname)

# sample Pressure onto wing if needed
if "Pressure" not in wing.array_names:
    try: wing = wing.sample(vol)
    except Exception: pass
if "Pressure" in wing.array_names:
    wing["Cp"] = (wing["Pressure"] - P_REF)/(0.5*RHO*UINF*UINF)

# ---------- Wing plots (camera = wing bounds) ----------
save_pic(wing, "Cp" if "Cp" in wing.array_names else "Pressure",
         "wing_cp.png", cam_bounds=wing.bounds)
save_pic(wing, "Pressure" if "Pressure" in wing.array_names else None,
         "wing_pressure.png", cam_bounds=wing.bounds)
# -------- slice + arrows (XZ plane at mid-span) --------
b = vol.bounds
xmin,xmax,ymin,ymax,zmin,zmax = b
dx,dy,dz = xmax-xmin, ymax-ymin, zmax-zmin

# XZ plane  -> normal = +Y, origin at y_mid; center x,z for nice framing
cx = 0.5*(xmin+xmax)
ym = 0.5*(ymin+ymax)
cz = 0.5*(zmin+zmax)
origin = (cx, ym, cz)
normal = (0, 1, 0)

# sampling/glyph plane that lies in XZ
i_size, j_size = 0.7*dx, 0.7*dz
slice_vol = vol.slice(normal=normal, origin=origin)

grid = pv.Plane(center=origin, direction=normal,
                i_size=i_size, j_size=j_size,
                i_resolution=45, j_resolution=18)

sampled = grid.sample(vol)
if vname in sampled.array_names:
    U = np.asarray(sampled[vname])[:, :3]
    sampled["|U|"] = np.linalg.norm(U, axis=1)
    # subsample arrows for clarity
    pts_arr, vec_arr = subsample_points(sampled.points, U, max_pts=1200)
    pts_pd = pv.PolyData(pts_arr); pts_pd[vname] = vec_arr; pts_pd.set_active_vectors(vname)
    arrows = pts_pd.glyph(scale=False, factor=0.015*max(dx,dy,dz), geom=pv.Arrow())
else:
    arrows = None

cam_bounds = union_bounds(slice_vol.bounds, wing.bounds)
p = pv.Plotter(off_screen=True, window_size=(1600,1000))
p.add_mesh(slice_vol, scalars="|U|" if "|U|" in slice_vol.array_names else None, opacity=0.65)
if arrows is not None: p.add_mesh(arrows, color="black")
p.add_mesh(wing, color="white", opacity=0.25)
p.camera_position = fit_camera(cam_bounds, FLOW_AXIS)
p.add_axes()
p.show(screenshot="slice_speed_arrows.png")
p.close()

# ---------- Streamlines + arrows (camera = union(lines, wing), fallback to wing) ----------
center = ((xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2)
if FLOW_AXIS=="x":
    seed_center = (xmin - 0.05*dx, center[1], center[2]); seed_dir=(1,0,0)
    i_s, j_s = 0.6*dy, 0.6*dz
elif FLOW_AXIS=="y":
    seed_center = (center[0], ymin - 0.05*dy, center[2]); seed_dir=(0,1,0)
    i_s, j_s = 0.6*dx, 0.6*dz
else:
    seed_center = (center[0], center[1], zmin - 0.05*dz); seed_dir=(0,0,1)
    i_s, j_s = 0.6*dx, 0.6*dy

seed = pv.Plane(center=seed_center, direction=seed_dir, i_size=i_s, j_size=j_s,
                i_resolution=28, j_resolution=28)

vol2 = vol.copy(); vol2.set_active_vectors(vname)
try:
    lines = vol2.streamlines_from_source(
        source=seed, integrator_type=4, integration_direction='forward',
        step_unit='cl', initial_step_length=0.5, min_step_length=0.01,
        max_step_length=1.0, max_time=5.0, terminal_speed=0.0
    )
except Exception:
    lines = pv.PolyData()

if vname in lines.array_names:
    U = np.asarray(lines[vname])[:, :3]
    lines["|U|"] = np.linalg.norm(U, axis=1)

# build arrow glyphs by sampling velocity at streamline points
if lines.n_points > 0:
    pts_all = lines.points
    pts_pd = pv.PolyData(pts_all)
    pts_s  = vol.sample(pts_pd)
    if vname in pts_s.array_names:
        Upts = np.asarray(pts_s[vname])[:, :3]
        pts_sub, U_sub = subsample_points(pts_all, Upts, max_pts=2000)
        arr_pd = pv.PolyData(pts_sub); arr_pd[vname] = U_sub; arr_pd.set_active_vectors(vname)
        arrows3d = arr_pd.glyph(scale=False, factor=0.02*max(dx,dy,dz), geom=pv.Arrow())
    else:
        arrows3d = None
else:
    arrows3d = None

cam_bounds = union_bounds(lines.bounds if lines.n_points>0 else wing.bounds, wing.bounds)
p = pv.Plotter(off_screen=True, window_size=(1600,1000))
if lines.n_cells > 0:
    p.add_mesh(lines, scalars="|U|" if "|U|" in lines.array_names else None, line_width=1.5)
if arrows3d is not None:
    p.add_mesh(arrows3d, color="black")
p.add_mesh(wing, color="white", opacity=0.35, smooth_shading=True)
p.camera_position = fit_camera(cam_bounds, FLOW_AXIS)
p.add_axes()
p.show(screenshot="streamlines_arrows.png")
p.close()

# ---------- Turbulence/Q iso (camera = union(iso, wing)) ----------
turb_name,_ = pick_turb_field(vol)
if turb_name is None and vname in vol.array_names:
    try:
        grad = vol.compute_derivative(scalars=vname, vorticity=True, qcriterion=True)
        if "Q-criterion" in grad.array_names:
            vol["Q-criterion"] = grad["Q-criterion"]; turb_name = "Q-criterion"
        elif "vorticity" in grad.array_names:
            v = np.asarray(grad["vorticity"])[:, :3]
            vol["|omega|"] = np.linalg.norm(v, axis=1); turb_name = "|omega|"
    except Exception:
        pass

if turb_name:
    vals = np.asarray(vol[turb_name]); vals = vals[np.isfinite(vals)]
    if vals.size>0:
        hi = float(np.quantile(vals, 0.90))
        if hi <= 0 and vals.max() > 0: hi = 0.1*float(vals.max())
        try:
            iso = vol.contour(isosurfaces=[hi], scalars=turb_name)
            try: iso = iso.decimate(0.3, volume_preservation=True)
            except Exception: pass
            cam_bounds = union_bounds(iso.bounds, wing.bounds)
            save_pic(iso, turb_name, "turb_iso.png", overlay=wing, cam_bounds=cam_bounds)
        except Exception as e:
            print("Iso-surface failed:", e)

# ---------- Near-wall speed (camera = wing bounds) ----------
try:
    wing_poly = wing.extract_surface().triangulate()
    wing_n = wing_poly.compute_normals(point_normals=True, cell_normals=False,
                                       auto_orient_normals=True, consistent_normals=True,
                                       split_vertices=False, inplace=False)
    Lbox = max(dx,dy,dz); eps = 0.01*Lbox
    normals = None
    for key in ("Normals","PointNormals","vtkNormals"):
        if key in wing_n.point_data:
            normals = np.asarray(wing_n.point_data[key]); break
    if normals is not None and normals.shape[0] == wing_n.points.shape[0]:
        pts_off = wing_n.points + eps * normals
        wing_off = pv.PolyData(pts_off, wing_n.faces)
    else:
        centers = wing_n.cell_centers().points
        cnorm   = wing_n.cell_normals
        pts_off = centers + eps * cnorm
        wing_off = pv.PolyData(pts_off)
    sampled = wing_off.sample(vol)
    if vname in sampled.array_names:
        U = np.asarray(sampled[vname])[:, :3]
        sampled["|U|"] = np.linalg.norm(U, axis=1)
        save_pic(sampled, "|U|", "nearwall_speed.png", overlay=wing, cam_bounds=wing.bounds)
except Exception as e:
    print("Near-wall speed failed:", e)

# ---------- history plot (if present) ----------
if os.path.exists(HIS_PATH):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        its, cl, cd = [], [], []
        with open(HIS_PATH, newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    its.append(int(row.get("Iter", row.get("iter", row.get("INNER_ITER", 0)))))
                    cl.append(float(row.get("CL", row.get("C_L", np.nan))))
                    cd.append(float(row.get("CD", row.get("C_D", np.nan))))
                except Exception:
                    continue
        if its and (any(np.isfinite(cl)) or any(np.isfinite(cd))):
            plt.figure(figsize=(8,5))
            if any(np.isfinite(cl)): plt.plot(its, cl, label="CL")
            if any(np.isfinite(cd)): plt.plot(its, cd, label="CD")
            plt.xlabel("Iteration"); plt.ylabel("Coeff."); plt.legend(); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig("history.png", dpi=160); plt.close()
    except Exception as e:
        print("history plot failed:", e)

print("Saved: wing_cp.png, wing_pressure.png, slice_speed_arrows.png, streamlines_arrows.png, turb_iso.png (if available), nearwall_speed.png (if successful), history.png (if present)")
