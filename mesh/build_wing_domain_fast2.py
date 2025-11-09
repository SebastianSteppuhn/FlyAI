# Solid STEP -> small farfield box -> boolean cut -> tag inlet/outlet/walls/farfield
# -> distance-to-wing sizing (NOT to the box) -> fast tet mesh -> export SU2 only
import gmsh, multiprocessing
from math import fabs

# ---------- USER SETTINGS ----------
STEP_PATH    = "../geo/wing_solid.stp"   # solid STEP
FLOW_AXIS    = "x"                  # 'x'|'y'|'z'
PRESET       = "SMALL_~1_2M"     # target ~0.5–1.2M
SU2_FILENAME = "../mesh/mesh2.su2"
# -----------------------------------

PRESETS = {
    # very small box + very coarse farfield
    "TINY_~0p5_1p2M": {
        "UP": 1.5, "DN": 3.5, "H1": 1.0, "H2": 1.0,  # box extents in Lref
        "LC_NEAR": 1/180, "LC_FAR": 1/1, "D_NEAR": 1/60, "D_FAR": 1/6
    },
    # slightly larger (~1–2M)
    "SMALL_~1_2M": {
        "UP": 2.5, "DN": 6.0, "H1": 1.5, "H2": 1.5,
        "LC_NEAR": 1/300, "LC_FAR": 1/2, "D_NEAR": 1/50, "D_FAR": 1/5
    }
}

def bb(dim, tag): return gmsh.model.occ.getBoundingBox(dim, tag)
def on_plane(face, ax, pos, tol):
    b = gmsh.model.occ.getBoundingBox(2, face); i = {"x":0,"y":1,"z":2}[ax]
    return abs(b[i]-pos) <= tol and abs(b[i+3]-pos) <= tol

gmsh.initialize()
gmsh.model.add("wing_ext_fast2")

# speed/low-mem tweaks
gmsh.option.setNumber("General.NumThreads", max(1, multiprocessing.cpu_count()-1))
gmsh.option.setNumber("Mesh.Algorithm3D", 10)          # HXT if available
gmsh.option.setNumber("Mesh.Optimize", 0)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
gmsh.option.setNumber("Mesh.ElementOrder", 1)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.SaveAll", 0)

# import wing
gmsh.model.occ.importShapes(STEP_PATH)
gmsh.model.occ.synchronize()
vols = gmsh.model.occ.getEntities(3)
if not vols:
    gmsh.finalize(); raise RuntimeError("STEP has no solid volume (needs solid B-Rep).")

# wing bbox & ref length
Bs = [bb(3, v[1]) for v in vols]
xmin=min(b[0] for b in Bs); ymin=min(b[1] for b in Bs); zmin=min(b[2] for b in Bs)
xmax=max(b[3] for b in Bs); ymax=max(b[4] for b in Bs); zmax=max(b[5] for b in Bs)
xc=0.5*(xmin+xmax); yc=0.5*(ymin+ymax); zc=0.5*(zmin+zmax)
Lx=xmax-xmin; Ly=ymax-ymin; Lz=zmax-zmin; Lref=max(Lx,Ly,Lz,1e-9)

P = PRESETS[PRESET]
up=P["UP"]*Lref; dn=P["DN"]*Lref; h1=P["H1"]*Lref; h2=P["H2"]*Lref

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
    if FLOW_AXIS=="x" and on_plane(s,"x",X0,tol): inlet.append(s);  continue
    if FLOW_AXIS=="x" and on_plane(s,"x",X1,tol): outlet.append(s); continue
    if FLOW_AXIS=="y" and on_plane(s,"y",Y0,tol): inlet.append(s);  continue
    if FLOW_AXIS=="y" and on_plane(s,"y",Y1,tol): outlet.append(s); continue
    if FLOW_AXIS=="z" and on_plane(s,"z",Z0,tol): inlet.append(s);  continue
    if FLOW_AXIS=="z" and on_plane(s,"z",Z1,tol): outlet.append(s); continue
    if (on_plane(s,"x",X0,tol) or on_plane(s,"x",X1,tol) or
        on_plane(s,"y",Y0,tol) or on_plane(s,"y",Y1,tol) or
        on_plane(s,"z",Z0,tol) or on_plane(s,"z",Z1,tol)):
        far.append(s)
in_ids,set_out,set_far=set(inlet),set(outlet),set(far)
walls=[s for s in faces if s not in in_ids|set_out|set_far]

# physicals
gmsh.model.addPhysicalGroup(3,[fluid_vol],name="fluid")
if inlet:  gmsh.model.addPhysicalGroup(2,inlet, name="inlet")
if outlet: gmsh.model.addPhysicalGroup(2,outlet,name="outlet")
if walls:  gmsh.model.addPhysicalGroup(2,walls, name="walls")
if far:    gmsh.model.addPhysicalGroup(2,far,   name="farfield")

# -------- KEY FIX: size by distance to ORIGINAL WING SURFACES (not box) --------
wing_faces = [t for (d,t) in gmsh.model.getBoundary(vols, oriented=False, recursive=False) if d==2]
h_near=Lref*P["LC_NEAR"]; h_far=Lref*P["LC_FAR"]; d_near=Lref*P["D_NEAR"]; d_far=Lref*P["D_FAR"]

gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "FacesList", wing_faces)  # distance to wing only
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "IField", 1)
gmsh.model.mesh.field.setNumber(2, "LcMin", h_near)
gmsh.model.mesh.field.setNumber(2, "LcMax", h_far)
gmsh.model.mesh.field.setNumber(2, "DistMin", d_near)
gmsh.model.mesh.field.setNumber(2, "DistMax", d_far)
gmsh.model.mesh.field.setAsBackgroundMesh(2)

# mesh & export
gmsh.model.mesh.generate(3)
gmsh.write(SU2_FILENAME)
print(f"[{PRESET}] SU2 mesh written to: {SU2_FILENAME}")
print("Faces -> inlet:",len(inlet),"outlet:",len(outlet),"walls:",len(walls),"farfield:",len(far))
gmsh.finalize()
