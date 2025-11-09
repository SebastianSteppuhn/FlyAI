# Solid STEP -> farfield box (relative to chord) -> boolean cut
# -> tag inlet/outlet/walls/farfield -> distance-based size field
# -> tetra mesh -> export CGNS + SU2

import gmsh
from math import fabs

# -------------------- USER KNOBS --------------------
STEP_PATH      = "../geo/wing_solid.stp"   # manifold solid STEP
FLOW_AXIS      = "x"                  # 'x'|'y'|'z' (freestream direction)

# Choose one: "MID_~1p5M" or "LOW_~0p8M"
PRESET         = "MID_~1p5M"

# Angle of attack (deg) only affects solver, not mesh; here for book-keeping.
AOA_DEG        = 0.0

# Exports
CGNS_FILENAME  = "../mesh/mesh.cgns"
SU2_FILENAME   = "../mesh/mesh.su2"
# ----------------------------------------------------

# Presets are expressed in multiples of Lref = max bbox extent of the wing
PRESETS = {
    "MID_~1p5M": {
        # Farfield extents (multipliers of Lref)
        "UPSTREAM": 5.0,   # inlet is 5 chords upstream of nose
        "DOWNSTREAM": 12.0,# outlet 12 chords downstream of tail
        "HALF_ORT1": 2.5,  # half-height (span-normal) ~2.5 chords
        "HALF_ORT2": 2.5,  # half-width  ~2.5 chords

        # Size field (Lc near wall, farfield & blending distances)
        "LC_NEAR_FRAC": 1/800,   # near-wall cell size ~ Lref/800
        "LC_FAR_FRAC":  1/6,     # farfield cell size ~ Lref/6
        "D_NEAR_FRAC":  1/60,    # within Lref/60 use LC_NEAR
        "D_FAR_FRAC":   1/5,     # beyond Lref/5 use LC_FAR
    },
    "LOW_~0p8M": {
        "UPSTREAM": 4.0,
        "DOWNSTREAM": 10.0,
        "HALF_ORT1": 2.0,
        "HALF_ORT2": 2.0,
        "LC_NEAR_FRAC": 1/600,
        "LC_FAR_FRAC":  1/4,     # coarser farfield
        "D_NEAR_FRAC":  1/50,
        "D_FAR_FRAC":   1/4,
    }
}

def occ_bb(dim, tag):
    return gmsh.model.occ.getBoundingBox(dim, tag)  # xmin,ymin,zmin,xmax,ymax,zmax

def is_plane_at_face(face_tag, axis, pos, tol):
    b = occ_bb(2, face_tag)
    idx = {"x":0, "y":1, "z":2}[axis]
    return fabs(b[idx] - pos) <= tol and fabs(b[idx+3] - pos) <= tol

def main():
    gmsh.initialize()
    gmsh.model.add("wing_ext_flow")

    # Import solid STEP
    gmsh.model.occ.importShapes(STEP_PATH)
    gmsh.model.occ.synchronize()

    vols = gmsh.model.occ.getEntities(3)
    if not vols:
        gmsh.finalize()
        raise RuntimeError("STEP has no solid volume. Re-export as manifold solid B-Rep.")

    # Wing bbox & reference length
    Bs = [occ_bb(3, v[1]) for v in vols]
    xmin = min(b[0] for b in Bs); ymin = min(b[1] for b in Bs); zmin = min(b[2] for b in Bs)
    xmax = max(b[3] for b in Bs); ymax = max(b[4] for b in Bs); zmax = max(b[5] for b in Bs)
    xc, yc, zc = 0.5*(xmin+xmax), 0.5*(ymin+ymax), 0.5*(zmin+zmax)
    Lx, Ly, Lz = xmax-xmin, ymax-ymin, zmax-zmin
    Lref = max(Lx, Ly, Lz, 1e-9)

    P = PRESETS[PRESET]
    upstream   = P["UPSTREAM"]   * Lref
    downstream = P["DOWNSTREAM"] * Lref
    half_1     = P["HALF_ORT1"]  * Lref
    half_2     = P["HALF_ORT2"]  * Lref

    # Build farfield box, positioned relative to wing bbox
    if FLOW_AXIS == "x":
        X0, X1 = xmin - upstream, xmax + downstream
        Y0, Y1 = yc - half_1, yc + half_1
        Z0, Z1 = zc - half_2, zc + half_2
        box = gmsh.model.occ.addBox(X0, Y0, Z0, X1 - X0, Y1 - Y0, Z1 - Z0)
    elif FLOW_AXIS == "y":
        Y0, Y1 = ymin - upstream, ymax + downstream
        X0, X1 = xc - half_1, xc + half_1
        Z0, Z1 = zc - half_2, zc + half_2
        box = gmsh.model.occ.addBox(X0, Y0, Z0, X1 - X0, Y1 - Y0, Z1 - Z0)
    else:
        Z0, Z1 = zmin - upstream, zmax + downstream
        X0, X1 = xc - half_1, xc + half_1
        Y0, Y1 = yc - half_2, yc + half_2
        box = gmsh.model.occ.addBox(X0, Y0, Z0, X1 - X0, Y1 - Y0, Z1 - Z0)

    gmsh.model.occ.synchronize()

    # Fluid = box \ wing
    fluid, _ = gmsh.model.occ.cut([(3, box)], vols, removeObject=True, removeTool=False)
    gmsh.model.occ.synchronize()
    if not fluid:
        gmsh.finalize()
        raise RuntimeError("Boolean cut failed. Enlarge box or check STEP integrity.")
    fluid_vol = fluid[0][1]

    # Classify boundary faces
    faces = [t for (d, t) in gmsh.model.getBoundary([(3, fluid_vol)], oriented=False, recursive=False) if d == 2]
    Ldom = max(fabs(X1 - X0), fabs(Y1 - Y0), fabs(Z1 - Z0))
    tol  = 1e-8 * (Ldom if Ldom > 0 else 1.0)

    inlet_s, outlet_s, far_s = [], [], []
    for s in faces:
        if   FLOW_AXIS == "x" and is_plane_at_face(s, "x", X0, tol): inlet_s.append(s);  continue
        elif FLOW_AXIS == "x" and is_plane_at_face(s, "x", X1, tol): outlet_s.append(s); continue
        elif FLOW_AXIS == "y" and is_plane_at_face(s, "y", Y0, tol): inlet_s.append(s);  continue
        elif FLOW_AXIS == "y" and is_plane_at_face(s, "y", Y1, tol): outlet_s.append(s); continue
        elif FLOW_AXIS == "z" and is_plane_at_face(s, "z", Z0, tol): inlet_s.append(s);  continue
        elif FLOW_AXIS == "z" and is_plane_at_face(s, "z", Z1, tol): outlet_s.append(s); continue
        # remaining outer faces:
        if (is_plane_at_face(s, "x", X0, tol) or is_plane_at_face(s, "x", X1, tol) or
            is_plane_at_face(s, "y", Y0, tol) or is_plane_at_face(s, "y", Y1, tol) or
            is_plane_at_face(s, "z", Z0, tol) or is_plane_at_face(s, "z", Z1, tol)):
            far_s.append(s)

    in_ids, out_ids, far_ids = set(inlet_s), set(outlet_s), set(far_s)
    walls = [s for s in faces if s not in in_ids | out_ids | far_ids]

    # Physicals
    gmsh.model.addPhysicalGroup(3, [fluid_vol], name="fluid")
    if inlet_s:  gmsh.model.addPhysicalGroup(2, inlet_s,  name="inlet")
    if outlet_s: gmsh.model.addPhysicalGroup(2, outlet_s, name="outlet")
    if walls:    gmsh.model.addPhysicalGroup(2, walls,    name="walls")
    if far_s:    gmsh.model.addPhysicalGroup(2, far_s,    name="farfield")

    # Size field (distance to walls)
    h_near = Lref * P["LC_NEAR_FRAC"]
    h_far  = Lref * P["LC_FAR_FRAC"]
    d_near = Lref * P["D_NEAR_FRAC"]
    d_far  = Lref * P["D_FAR_FRAC"]

    gmsh.model.mesh.field.add("Distance", 1)
    if walls:
        gmsh.model.mesh.field.setNumbers(1, "FacesList", walls)
    else:
        # fallback: all wing faces (if classification missed walls)
        wing_faces = [t for (d,t) in gmsh.model.occ.getEntities(2)]
        gmsh.model.mesh.field.setNumbers(1, "FacesList", wing_faces)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", h_near)
    gmsh.model.mesh.field.setNumber(2, "LcMax", h_far)
    gmsh.model.mesh.field.setNumber(2, "DistMin", d_near)
    gmsh.model.mesh.field.setNumber(2, "DistMax", d_far)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    # Keep farfield coarse
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)  # linear tets (lower memory)
    gmsh.option.setNumber("Mesh.SaveAll", 0)

    # Mesh & export
    gmsh.model.mesh.generate(3)
    gmsh.write(CGNS_FILENAME)
    gmsh.write(SU2_FILENAME)

    print(f"[{PRESET}] Exported:")
    print("  CGNS:", CGNS_FILENAME)
    print("  SU2 :", SU2_FILENAME)
    print("Faces -> inlet:", len(inlet_s), "outlet:", len(outlet_s),
          "walls:", len(walls), "farfield:", len(far_s))
    gmsh.finalize()

if __name__ == "__main__":
    main()
