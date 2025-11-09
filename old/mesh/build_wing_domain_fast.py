# Solid STEP -> farfield box -> boolean cut -> tag inlet/outlet/walls/farfield
# -> fast tetra mesher -> export SU2 only (skip CGNS for speed)
import gmsh, multiprocessing
from math import fabs

# ---- USER SETTINGS ----
STEP_PATH      = "../../geo/wing_solid.stp"  # solid STEP
FLOW_AXIS      = "x"                  # 'x'|'y'|'z'
PRESET         = "ULTRA_~0p6_1p2M"    # <- new super-coarse preset (aims ~0.6–1.2M)

# Exports
SU2_FILENAME   = "mesh.su2"

# Presets in multiples of Lref = max wing bbox extent
PRESETS = {
    # safer & faster (aim ~0.6–1.2M)
    "ULTRA_~0p6_1p2M": {
        "UPSTREAM": 2.0, "DOWNSTREAM": 6.0, "HALF_ORT1": 1.2, "HALF_ORT2": 1.2,
        "LC_NEAR_FRAC": 1/250,   # near-wall size ~ Lref/250  (coarser)
        "LC_FAR_FRAC":  1/2,     # farfield size ~ Lref/2     (MUCH coarser)
        "D_NEAR_FRAC":  1/40,    # within Lref/40 use near size
        "D_FAR_FRAC":   1/4,     # beyond Lref/4 use far size
    },
    # medium (aim ~1–2M)
    "MID_~1_2M": {
        "UPSTREAM": 3.0, "DOWNSTREAM": 8.0, "HALF_ORT1": 1.8, "HALF_ORT2": 1.8,
        "LC_NEAR_FRAC": 1/400,
        "LC_FAR_FRAC":  1/3,
        "D_NEAR_FRAC":  1/50,
        "D_FAR_FRAC":   1/4,
    }
}

def bb(dim, tag): return gmsh.model.occ.getBoundingBox(dim, tag)

def is_plane(face, ax, pos, tol):
    b = gmsh.model.occ.getBoundingBox(2, face)
    i = {"x":0,"y":1,"z":2}[ax]
    return abs(b[i]-pos) <= tol and abs(b[i+3]-pos) <= tol

gmsh.initialize()
gmsh.model.add("wing_ext_fast")

# speed tweaks
gmsh.option.setNumber("General.NumThreads", max(1, multiprocessing.cpu_count()-1))
gmsh.option.setNumber("Mesh.Algorithm3D", 10)          # HXT (fast)
gmsh.option.setNumber("Mesh.Optimize", 0)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
gmsh.option.setNumber("Mesh.ElementOrder", 1)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.SaveAll", 0)

# import solid
gmsh.model.occ.importShapes(STEP_PATH)
gmsh.model.occ.synchronize()
vols = gmsh.model.occ.getEntities(3)
if not vols:
    gmsh.finalize(); raise RuntimeError("STEP has no solid volume (needs solid B-Rep).")

# wing bbox / Lref
Bs = [bb(3,v[1]) for v in vols]
xmin=min(b[0] for b in Bs); ymin=min(b[1] for b in Bs); zmin=min(b[2] for b in Bs)
xmax=max(b[3] for b in Bs); ymax=max(b[4] for b in Bs); zmax=max(b[5] for b in Bs)
xc=0.5*(xmin+xmax); yc=0.5*(ymin+ymax); zc=0.5*(zmin+zmax)
Lx=xmax-xmin; Ly=ymax-ymin; Lz=zmax-zmin; Lref=max(Lx,Ly,Lz,1e-9)

P = PRESETS[PRESET]
up=P["UPSTREAM"]*Lref; dn=P["DOWNSTREAM"]*Lref; h1=P["HALF_ORT1"]*Lref; h2=P["HALF_ORT2"]*Lref

# farfield box
if   FLOW_AXIS=="x": X0,X1=xmin-up, xmax+dn; Y0,Y1=yc-h1, yc+h1; Z0,Z1=zc-h2, zc+h2
elif FLOW_AXIS=="y": Y0,Y1=ymin-up, ymax+dn; X0,X1=xc-h1, xc+h1; Z0,Z1=zc-h2, zc+h2
else:                Z0,Z1=zmin-up, zmax+dn; X0,X1=xc-h1, xc+h1; Y0,Y1=yc-h2, yc+h2
box = gmsh.model.occ.addBox(X0,Y0,Z0, X1-X0, Y1-Y0, Z1-Z0)
gmsh.model.occ.synchronize()

# fluid = box \ wing
fluid,_ = gmsh.model.occ.cut([(3,box)], vols, removeObject=True, removeTool=False)
gmsh.model.occ.synchronize()
if not fluid: gmsh.finalize(); raise RuntimeError("Boolean cut failed.")
fluid_vol = fluid[0][1]

# classify faces
faces=[t for (d,t) in gmsh.model.getBoundary([(3,fluid_vol)], oriented=False, recursive=False) if d==2]
Ldom=max(abs(X1-X0),abs(Y1-Y0),abs(Z1-Z0)); tol=1e-8*(Ldom if Ldom>0 else 1.0)
inlet,outlet,far=[],[],[]
for s in faces:
    if FLOW_AXIS=="x" and is_plane(s,"x",X0,tol): inlet.append(s);  continue
    if FLOW_AXIS=="x" and is_plane(s,"x",X1,tol): outlet.append(s); continue
    if FLOW_AXIS=="y" and is_plane(s,"y",Y0,tol): inlet.append(s);  continue
    if FLOW_AXIS=="y" and is_plane(s,"y",Y1,tol): outlet.append(s); continue
    if FLOW_AXIS=="z" and is_plane(s,"z",Z0,tol): inlet.append(s);  continue
    if FLOW_AXIS=="z" and is_plane(s,"z",Z1,tol): outlet.append(s); continue
    if (is_plane(s,"x",X0,tol) or is_plane(s,"x",X1,tol) or
        is_plane(s,"y",Y0,tol) or is_plane(s,"y",Y1,tol) or
        is_plane(s,"z",Z0,tol) or is_plane(s,"z",Z1,tol)): far.append(s)
in_ids,set_out,set_far=set(inlet),set(outlet),set(far)
walls=[s for s in faces if s not in in_ids|set_out|set_far]

# physicals
gmsh.model.addPhysicalGroup(3,[fluid_vol],name="fluid")
if inlet:  gmsh.model.addPhysicalGroup(2,inlet, name="inlet")
if outlet: gmsh.model.addPhysicalGroup(2,outlet,name="outlet")
if walls:  gmsh.model.addPhysicalGroup(2,walls, name="walls")
if far:    gmsh.model.addPhysicalGroup(2,far,   name="farfield")

# size field
h_near=Lref*P["LC_NEAR_FRAC"]; h_far=Lref*P["LC_FAR_FRAC"]
d_near=Lref*P["D_NEAR_FRAC"];  d_far=Lref*P["D_FAR_FRAC"]

gmsh.model.mesh.field.add("Distance",1)
gmsh.model.mesh.field.setNumbers(1,"FacesList", walls if walls else [t for (d,t) in gmsh.model.occ.getEntities(2)])
gmsh.model.mesh.field.add("Threshold",2)
gmsh.model.mesh.field.setNumber(2,"IField",1)
gmsh.model.mesh.field.setNumber(2,"LcMin",h_near)
gmsh.model.mesh.field.setNumber(2,"LcMax",h_far)
gmsh.model.mesh.field.setNumber(2,"DistMin",d_near)
gmsh.model.mesh.field.setNumber(2,"DistMax",d_far)
gmsh.model.mesh.field.setAsBackgroundMesh(2)

# generate & export
gmsh.model.mesh.generate(3)
gmsh.write(SU2_FILENAME)
print(f"[{PRESET}] Exported SU2: {SU2_FILENAME}")
print("Faces -> inlet:",len(inlet),"outlet:",len(outlet),"walls:",len(walls),"farfield:",len(far))
gmsh.finalize()
