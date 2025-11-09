import gmsh, numpy as np

gmsh.initialize()
gmsh.model.add("bfs_from_step")

# --- Import your STEP file (same directory as this script) ---
# Use either importShapes (OCC CAD kernel) or merge; importShapes is preferred.
gmsh.model.occ.importShapes("geometry.step")   # <- your file
gmsh.model.occ.synchronize()

# Optional: global mesh size
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.03)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.06)

# --- Tag physical groups ---
# All fluid surfaces (2D) -> one region
surfs = [s for (dim, s) in gmsh.model.getEntities(dim=2)]
gmsh.model.addPhysicalGroup(2, surfs, name="fluid")

# Find global x-extent to detect inlet/outlet edges by position
xmin = +1e300; xmax = -1e300
for s in surfs:
    x1,y1,z1, x2,y2,z2 = gmsh.model.getBoundingBox(2, s)
    xmin = min(xmin, x1); xmax = max(xmax, x2)

# Classify boundary curves (1D) by their bbox relative to xmin/xmax
curves = [c for (dim, c) in gmsh.model.getEntities(dim=1)]
inlet, outlet, walls = [], [], []
tol = 1e-6

for c in curves:
    x1,y1,z1, x2,y2,z2 = gmsh.model.getBoundingBox(1, c)
    if abs(x1 - xmin) < tol and abs(x2 - xmin) < tol:
        inlet.append(c)
    elif abs(x1 - xmax) < tol and abs(x2 - xmax) < tol:
        outlet.append(c)
    else:
        walls.append(c)

if not inlet or not outlet:
    raise RuntimeError("Could not detect inlet/outlet from STEP. Check model orientation.")

gmsh.model.addPhysicalGroup(1, inlet,  name="inlet")
gmsh.model.addPhysicalGroup(1, outlet, name="outlet")
gmsh.model.addPhysicalGroup(1, walls,  name="walls")

# --- Generate 2D mesh and write .msh for meshio/dolfinx ---
gmsh.model.mesh.generate(2)
gmsh.write("bfs.msh")
gmsh.finalize()
