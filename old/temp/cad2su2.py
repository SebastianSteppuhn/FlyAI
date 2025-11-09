#!/usr/bin/env python3
"""
CAD -> Mesh -> SU2 CFD (conda-only)
- Accepts STEP (.step/.stp) or STL (.stl)
- Builds farfield box, subtracts geometry, names BCs (farfield, wall)
- Writes mesh.msh AND mesh.su2 directly from Gmsh
- Runs SU2, prints CL/CD/L/D + residual drop

Env needs: gmsh, python-gmsh, su2, numpy, pandas
"""

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------- SU2 helpers ---------------------------

def write_su2_cfg(cfg_path, mesh_su2, mach, aoa_deg, reynolds, ref_len, inviscid, iters):
    solver = "EULER" if inviscid else "RANS"
    turb   = "NONE" if inviscid else "SA"
    wall   = "MARKER_EULER= ( wall )" if inviscid else "MARKER_HEATFLUX= ( wall, 0.0 )"

    cfg = f"""%
% Auto-generated SU2 config (v8+ friendly)
MATH_PROBLEM= DIRECT
SOLVER= {solver}
KIND_TURB_MODEL= {turb}

MESH_FILENAME= {mesh_su2.name}

% Freestream
MACH_NUMBER= {mach}
AOA= {aoa_deg}
SIDESLIP_ANGLE= 0.0
GAMMA_VALUE= 1.4
FREESTREAM_TEMPERATURE= 288.15
FREESTREAM_PRESSURE= 101325.0
REYNOLDS_NUMBER= {reynolds}
REYNOLDS_LENGTH= {ref_len}

% Boundaries (from physical names)
MARKER_FAR= ( farfield )
{wall}

% Numerics
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 5.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.5, 2.0, 1.1, 50.0 )
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1e-6
LINEAR_SOLVER_ITER= 50
CONV_NUM_METHOD_FLOW= ROE
VENKAT_LIMITER_COEFF= 0.05
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN

% Convergence & output
ITER= {iters}
CONV_RESIDUAL_MINVAL= 1e-10
CONV_STARTITER= 10
SCREEN_OUTPUT= (ITER, RMS_RES, LIFT, DRAG, MACH, AOA)
HISTORY_OUTPUT= (ITER, RMS_RES, CL, CD, CMZ)
CONV_FILENAME= history   % name for history file (history.csv by default)
WRT_CON_FREQ= 1          % write every iteration
"""
    cfg_path.write_text(cfg)


def run_su2(cfg_path: Path, workdir: Path):
    exe = shutil.which("SU2_CFD")
    if not exe:
        raise RuntimeError("SU2_CFD not found in PATH (activate your conda env).")
    res = subprocess.run([exe, cfg_path.name], cwd=workdir, stdout=sys.stdout, stderr=sys.stderr)
    if res.returncode != 0:
        raise RuntimeError("SU2_CFD returned non-zero exit code.")


def summarize_history(history_csv: Path) -> str:
    if not history_csv.exists():
        return "No history.csv produced – SU2 may have failed early."

    df = None
    for kwargs in ({}, {"sep": ";"}, {"delim_whitespace": True}):
        try:
            df = pd.read_csv(history_csv, **kwargs)
            break
        except Exception:
            pass
    if df is None:
        return "Could not parse history.csv."

    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    c_iter = pick("ITER", "Iter", "Iteration")
    c_cl = pick("CL", "Lift_Coefficient", "LIFT")
    c_cd = pick("CD", "Drag_Coefficient", "DRAG")
    c_res = pick("RMS_RES", "RMS_Density", "RMS_RESIDUAL")

    lines, resF = [], None
    last = df.iloc[-1]

    if c_cl and c_cd:
        CL = float(last[c_cl]); CD = float(last[c_cd])
        lines.append(f"Final coefficients: CL = {CL:.4f}, CD = {CD:.4f}, L/D = {CL/CD:.2f}")
        if len(df) > 20:
            tail = df.tail(20)
            dCL = tail[c_cl].max() - tail[c_cl].min()
            dCD = tail[c_cd].max() - tail[c_cd].min()
            lines.append(f"Stability (last 20 iters): ΔCL = {dCL:.4e}, ΔCD = {dCD:.4e}")

    if c_res:
        res0 = float(df.iloc[0][c_res])
        resF = float(last[c_res])
        drop = np.log10(max(res0, 1e-99)) - np.log10(max(resF, 1e-99))
        lines.append(f"Residual drop: {drop:.1f} orders (final {c_res} = {resF:.2e})")

    if c_iter:
        lines.append(f"Iterations run: {int(last[c_iter])}")

    verdict = "✅ Converged-looking solution" if (resF is not None and resF < 1e-5) else "⚠️ May need more iters / refinement"
    return verdict + "\n" + "\n".join(lines)


# --------------------------- mesher core ---------------------------

def _bbox_from_all_entities(gmsh):
    # Robust bbox across any available entities (0..3D)
    have_any = False
    xmn, ymn, zmn = +1e300, +1e300, +1e300
    xmx, ymx, zmx = -1e300, -1e300, -1e300
    for dim in (3, 2, 1, 0):
        for (d, tag) in gmsh.model.getEntities(dim):
            bx = gmsh.model.getBoundingBox(d, tag)
            xmn, ymn, zmn = min(xmn, bx[0]), min(ymn, bx[1]), min(zmn, bx[2])
            xmx, ymx, zmx = max(xmx, bx[3]), max(ymx, bx[4]), max(zmx, bx[5])
            have_any = True
    if not have_any:
        raise RuntimeError("No entities present after import.")
    L = max(xmx - xmn, ymx - ymn, zmx - zmn)
    if not np.isfinite([xmn, ymn, zmn, xmx, ymx, zmx]).all() or L <= 0:
        raise RuntimeError("Degenerate bounding box.")
    return (xmn, ymn, zmn, xmx, ymx, zmx, L)


def _add_phys(gmsh, name, dim, entities):
    tag = gmsh.model.addPhysicalGroup(dim, [e[1] for e in entities])
    gmsh.model.setPhysicalName(dim, tag, name)
    return tag


def _geo_make_box(gmsh, x0, y0, z0, dx, dy, dz):
    """
    Build a hexahedral box in the GEO kernel:
    - returns (vol_tag, surf_entities_list)
    """
    # Points
    p = {}
    p[0] = gmsh.model.geo.addPoint(x0,       y0,       z0)
    p[1] = gmsh.model.geo.addPoint(x0+dx,    y0,       z0)
    p[2] = gmsh.model.geo.addPoint(x0+dx,    y0+dy,    z0)
    p[3] = gmsh.model.geo.addPoint(x0,       y0+dy,    z0)
    p[4] = gmsh.model.geo.addPoint(x0,       y0,       z0+dz)
    p[5] = gmsh.model.geo.addPoint(x0+dx,    y0,       z0+dz)
    p[6] = gmsh.model.geo.addPoint(x0+dx,    y0+dy,    z0+dz)
    p[7] = gmsh.model.geo.addPoint(x0,       y0+dy,    z0+dz)

    # Helper to make a rectangular plane surface from 4 points
    def rect(a,b,c,d):
        l1 = gmsh.model.geo.addLine(a, b)
        l2 = gmsh.model.geo.addLine(b, c)
        l3 = gmsh.model.geo.addLine(c, d)
        l4 = gmsh.model.geo.addLine(d, a)
        loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        return gmsh.model.geo.addPlaneSurface([loop])

    # Six faces
    s_bottom = rect(p[0], p[1], p[2], p[3])
    s_top    = rect(p[4], p[5], p[6], p[7])
    s_front  = rect(p[0], p[1], p[5], p[4])
    s_back   = rect(p[3], p[2], p[6], p[7])
    s_left   = rect(p[0], p[4], p[7], p[3])
    s_right  = rect(p[1], p[2], p[6], p[5])

    gmsh.model.geo.synchronize()

    # Surface loop / volume
    sl = gmsh.model.geo.addSurfaceLoop([s_bottom, s_top, s_front, s_back, s_left, s_right])
    vol = gmsh.model.geo.addVolume([sl])
    gmsh.model.geo.synchronize()

    # Collect the surfaces (dim=2) belonging to the box volume
    box_surfs = gmsh.model.getBoundary([(3, vol)], oriented=False, combined=False)
    return vol, box_surfs


def build_mesh_from_cad(
    cad_path: Path,
    msh_path: Path,
    su2_path: Path,
    near_ratio=0.02,
    far_ratio=0.5,
    box_upstream=5.0,
    box_downstream=15.0,
    box_side=10.0,
    box_topbottom=10.0,
):
    """
    Creates a 3D external mesh with physical groups:
      - farfield (outer box)
      - wall (aircraft)
      - fluid (volume)
    Writes both .msh and .su2 directly from Gmsh.
    Returns reference length L (overall size of the geometry).
    """
    import gmsh

    ext = cad_path.suffix.lower()
    if ext not in [".stp", ".step", ".stl"]:
        raise RuntimeError("Unsupported CAD: use .step/.stp or .stl")

    gmsh.initialize()
    gmsh.model.add("ext_flow")

    # Make sure all entities get written, and names are kept
    gmsh.option.setNumber("Mesh.SaveAll", 1)

    try:
        if ext in [".stp", ".step"]:
            # ---------- STEP path (OpenCASCADE booleans) ----------
            gmsh.model.occ.importShapes(str(cad_path))
            gmsh.model.occ.synchronize()

            # Try to get volumes; if none, heal & attempt making one from faces
            vols = gmsh.model.occ.getEntities(3)
            if not vols:
                gmsh.model.occ.healShapes()
                gmsh.model.occ.removeAllDuplicates()
                gmsh.model.occ.synchronize()
                vols = gmsh.model.occ.getEntities(3)
                if not vols:
                    faces = [tag for (_, tag) in gmsh.model.occ.getEntities(2)]
                    if faces:
                        try:
                            loop = gmsh.model.occ.addSurfaceLoop(faces)
                            gmsh.model.occ.addVolume([loop])
                            gmsh.model.occ.synchronize()
                            vols = gmsh.model.occ.getEntities(3)
                        except Exception:
                            pass
            if not vols:
                raise RuntimeError("STEP has no watertight solids (needs SOLID export).")

            # Ref length from bbox
            xmn, ymn, zmn, xmx, ymx, zmx, L = _bbox_from_all_entities(gmsh)

            # Farfield box (OCC)
            xc = 0.5 * (xmn + xmx)
            x0 = xc - box_upstream * L
            y0 = ymn - box_side * L
            z0 = zmn - box_topbottom * L
            box = gmsh.model.occ.addBox(
                x0,
                y0,
                z0,
                (box_upstream + box_downstream) * L,
                (box_side * 2) * L,
                (box_topbottom * 2) * L,
            )
            gmsh.model.occ.synchronize()

            # Farfield surfaces before cut
            box_surfs = gmsh.model.getBoundary([(3, box)], oriented=False, combined=False)

            # Aircraft solids (exclude the box)
            all_solids = gmsh.model.occ.getEntities(3)
            aircraft_solids = [(d, t) for (d, t) in all_solids if t != box]
            if not aircraft_solids:
                raise RuntimeError("No 3D solids found in STEP after import.")

            # Cut: box \ aircraft
            fluid, _ = gmsh.model.occ.cut([(3, box)], aircraft_solids, removeTool=False)
            gmsh.model.occ.synchronize()
            if not fluid:
                raise RuntimeError("Boolean cut failed (geometry may be non-watertight).")

            fluid_vol = fluid[0]
            fluid_surfs = gmsh.model.getBoundary([fluid_vol], oriented=False, combined=False)

            # Classify wall vs farfield
            box_set = set(box_surfs)
            wall_surfs = [s for s in fluid_surfs if s not in box_set]

            # Sizing
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", L * near_ratio)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", L * far_ratio)

            # Mild near-wall refinement
            try:
                gmsh.model.mesh.field.add("Distance", 1)
                gmsh.model.mesh.field.setNumbers(1, "SurfacesList", [s[1] for s in wall_surfs])
                gmsh.model.mesh.field.setNumber(1, "NumPointsPerCurve", 100)

                gmsh.model.mesh.field.add("Threshold", 2)
                gmsh.model.mesh.field.setNumber(2, "InField", 1)
                gmsh.model.mesh.field.setNumber(2, "SizeMin", L * near_ratio)
                gmsh.model.mesh.field.setNumber(2, "SizeMax", L * far_ratio)
                gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1 * L)
                gmsh.model.mesh.field.setNumber(2, "DistMax", 2.0 * L)
                gmsh.model.mesh.field.setAsBackgroundMesh(2)
            except Exception:
                pass

            # Physical groups
            _add_phys(gmsh, "farfield", 2, box_surfs)
            _add_phys(gmsh, "wall", 2, wall_surfs)
            _add_phys(gmsh, "fluid", 3, [fluid_vol])

            # Mesh + direct writes
            gmsh.model.mesh.generate(3)
            gmsh.write(str(msh_path))
            gmsh.write(str(su2_path))
            return L

        else:
            # ---------- STL path (GEO kernel + manual box) ----------
            gmsh.merge(str(cad_path))

            # Reconstruct CAD surfaces from STL triangles
            angle = math.radians(40.0)            # crease angle for patching
            includeBoundary = True
            forceParametrizablePatches = True
            curveAngle = math.radians(60.0)

            gmsh.model.mesh.classifySurfaces(
                angle, includeBoundary, forceParametrizablePatches, curveAngle
            )
            gmsh.model.mesh.createGeometry()      # creates surfaces from mesh
            gmsh.model.geo.synchronize()

            faces = [tag for (_, tag) in gmsh.model.getEntities(2)]
            if not faces:
                raise RuntimeError("No surfaces could be reconstructed from STL.")

            # Attempt to make a closed volume from all faces
            try:
                sl = gmsh.model.geo.addSurfaceLoop(faces)
                obj_vol = gmsh.model.geo.addVolume([sl])
                gmsh.model.geo.synchronize()
            except Exception as e:
                raise RuntimeError("STL does not form a closed, watertight shell.") from e

            # Ref length from bbox
            xmn, ymn, zmn, xmx, ymx, zmx, L = _bbox_from_all_entities(gmsh)

            # Build farfield box manually (GEO kernel)
            xc = 0.5 * (xmn + xmx)
            x0 = xc - box_upstream * L
            y0 = ymn - box_side * L
            z0 = zmn - box_topbottom * L
            dx = (box_upstream + box_downstream) * L
            dy = (box_side * 2) * L
            dz = (box_topbottom * 2) * L
            box_vol, box_surfs = _geo_make_box(gmsh, x0, y0, z0, dx, dy, dz)

            # Boolean: box \ object (GEO kernel doesn’t have .cut in some builds; switch to OCC boolean via convert)
            gmsh.model.geo.synchronize()
            gmsh.model.geo.synchronize()
            # Convert GEO to OCC and perform boolean there for robustness
            gmsh.model.occ.importShapes(gmsh.write())
            gmsh.model.occ.synchronize()

            # Re-identify volumes (box first, then object) by size heuristic
            vols = gmsh.model.occ.getEntities(3)
            vols_sorted = sorted(vols, key=lambda e: gmsh.model.occ.getCenterOfMass(e[0], e[1])[0])
            # As a robust fallback, just do a fresh OCC box and OCC import of STL is complex; keeping STL path optional.
            raise RuntimeError("STL path not supported in this minimal fix. Use STEP, or send the STL and I’ll wire a robust boolean.")

    finally:
        gmsh.finalize()


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="STEP/STL -> mesh -> SU2 (external aerodynamics)")
    ap.add_argument("--cad", required=True, help="Path to .step/.stp or .stl")
    ap.add_argument("--mach", type=float, default=0.2, help="Freestream Mach number")
    ap.add_argument("--aoa", type=float, default=2.0, help="Angle of attack [deg]")
    ap.add_argument("--re", type=float, default=5e6, help="Reynolds number")
    ap.add_argument("--iters", type=int, default=800, help="SU2 iterations")
    ap.add_argument("--inviscid", action="store_true", help="Use Euler (no viscosity/turbulence)")
    ap.add_argument("--near", type=float, default=0.02, help="Near-wall size ratio   (L*near)")
    ap.add_argument("--far", type=float, default=0.5, help="Far-field size ratio    (L*far)")
    ap.add_argument("--box-up", type=float, default=5.0, help="Domain upstream size [×L]")
    ap.add_argument("--box-down", type=float, default=15.0, help="Domain downstream size [×L]")
    ap.add_argument("--box-side", type=float, default=10.0, help="Domain lateral size [×L]")
    ap.add_argument("--box-z", type=float, default=10.0, help="Domain top/bottom size [×L]")
    ap.add_argument("--outdir", default="run_cad", help="Output directory")
    args = ap.parse_args()

    work = Path(args.outdir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    cad = Path(args.cad).resolve()
    msh = work / "mesh.msh"
    su2mesh = work / "mesh.su2"
    cfg = work / "case.cfg"
    hist = work / "history.csv"

    print("=== 1) Meshing CAD with Gmsh ===")
    L = build_mesh_from_cad(
        cad_path=cad,
        msh_path=msh,
        su2_path=su2mesh,
        near_ratio=args.near,
        far_ratio=args.far,
        box_upstream=args.box_up,
        box_downstream=args.box_down,
        box_side=args.box_side,
        box_topbottom=args.box_z,
    )
    print(f"  wrote: {msh}")
    print(f"  wrote: {su2mesh}")

    print("=== 2) Writing SU2 config ===")
    write_su2_cfg(cfg, su2mesh, args.mach, args.aoa, args.re, L, args.inviscid, args.iters)
    print(f"  wrote: {cfg}")

    print("=== 3) Running SU2 ===")
    run_su2(cfg, work)

    print("=== 4) Post-process ===")
    print("\n" + summarize_history(hist))


if __name__ == "__main__":
    main()
